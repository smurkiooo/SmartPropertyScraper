"""
Сборщик объявлений о недвижимости с cian.ru.

Используется публичный JSON POST-эндпоинт поиска api.cian.ru, который
вызывает фронтенд сайта при обычном поиске. Возвращает структурированные
поля, текстовое описание автора и ссылки на фотографии.

Пример запуска:
    uv run python -m realty_scraper.cian \
        --region 1 --deal-type sale \
        --pages 20 --out data/processed/listings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

SEARCH_URL = "https://api.cian.ru/search-offers/v2/search-offers-desktop/"
MAIN_URL = "https://www.cian.ru/"
PAGE_SIZE = 20

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

API_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin": "https://www.cian.ru",
    "Referer": "https://www.cian.ru/kupit-kvartiru/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}


class CaptchaError(Exception):
    """Сервер ответил капчей вместо данных."""

# Типы сделок для Cian jsonQuery._type
DEAL_TYPE_MAP = {
    ("sale", "flat"): "flatsale",
    ("rent", "flat"): "flatrent",
    ("sale", "room"): "roomsale",
    ("rent", "room"): "roomrent",
    ("sale", "house"): "suburbansale",
    ("rent", "house"): "suburbanrent",
}

logger = logging.getLogger("realty_scraper.cian")


@dataclass
class Listing:
    """Плоское представление одного объявления для CSV."""

    offer_id: str
    url: str
    deal_type: str
    object_type: str
    price: float | None
    price_per_m2: float | None
    area_total: float | None
    area_living: float | None
    area_kitchen: float | None
    rooms: int | None
    floor: int | None
    floors_total: int | None
    address: str
    city: str
    underground: str
    latitude: float | None
    longitude: float | None
    year_built: int | None
    house_material: str
    has_balcony: bool | None
    description: str
    image_urls: list[str] = field(default_factory=list)

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["image_urls"] = ";".join(self.image_urls)
        row["description"] = (
            (self.description or "").replace("\r", " ").replace("\n", " ").strip()
        )
        return row


def _safe(d: Any, *keys: str | int, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if cur is None:
            return default
        try:
            cur = cur[k] if isinstance(k, int) else cur.get(k)
        except (KeyError, AttributeError, IndexError, TypeError):
            return default
    return cur if cur is not None else default


def parse_offer(raw: dict[str, Any]) -> Listing:
    """Достаёт поля из сырого JSON-объекта объявления Cian."""

    offer_id = str(raw.get("id") or raw.get("cianId") or "")
    url = raw.get("fullUrl") or raw.get("url") or f"https://www.cian.ru/sale/flat/{offer_id}/"

    # Цена и цена за м² — в bargainTerms
    bargain = raw.get("bargainTerms") or {}
    price = bargain.get("price")
    price_per_m2 = bargain.get("pricePerMeter")

    building = raw.get("building") or {}

    geo = raw.get("geo") or {}
    coords = geo.get("coordinates") or {}
    address_parts = geo.get("address") or []
    undergrounds = geo.get("undergrounds") or []

    # Строка адреса из частей без типа "location" (это просто "Москва")
    address_str = ", ".join(
        p.get("shortName") or p.get("name") or ""
        for p in address_parts
        if p.get("type") not in ("location",)
    )
    city = next(
        (
            p.get("shortName") or p.get("name") or ""
            for p in address_parts
            if p.get("type") == "location"
        ),
        "",
    )
    # Ближайшее метро — первый элемент undergrounds
    underground = _safe(undergrounds, 0, "name", default="")

    images: list[str] = []
    for photo in raw.get("photos") or []:
        u = photo.get("fullUrl") or photo.get("url") or photo.get("miniUrl")
        if u:
            images.append(u)

    return Listing(
        offer_id=offer_id,
        url=url,
        deal_type=str(raw.get("dealType") or ""),
        object_type=str(raw.get("objectType") or raw.get("category") or ""),
        price=price,
        price_per_m2=price_per_m2,
        area_total=raw.get("totalArea") or raw.get("area"),
        area_living=raw.get("livingArea"),
        area_kitchen=raw.get("kitchenArea"),
        rooms=raw.get("roomsCount"),
        floor=raw.get("floorNumber"),
        floors_total=_safe(building, "floorsCount"),
        address=address_str,
        city=city,
        underground=underground,
        latitude=coords.get("lat"),
        longitude=coords.get("lng"),
        year_built=_safe(building, "buildYear"),
        house_material=str(_safe(building, "materialType") or ""),
        has_balcony=raw.get("hasBalcony"),
        description=str(raw.get("description") or "").strip(),
        image_urls=images,
    )


def _build_json_query(
    *,
    region_id: int,
    deal_type: str,
    offer_type: str,
    page: int,
) -> dict[str, Any]:
    query_type = DEAL_TYPE_MAP.get((deal_type, offer_type), f"flat{deal_type}")
    return {
        "jsonQuery": {
            "_type": query_type,
            "engine_version": {"type": "term", "value": 2},
            "region": {"type": "terms", "value": [region_id]},
            "page": {"type": "term", "value": page},
        }
    }


class CianClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        self._session_ready = False

    def __enter__(self) -> "CianClient":
        self._init_session()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.close()

    def _init_session(self) -> None:
        """Посещаем главную страницу и страницу поиска, чтобы получить
        сессионные cookies и выглядеть как браузер."""
        logger.info("Инициализируем сессию на cian.ru…")
        try:
            self._client.get(MAIN_URL)
            time.sleep(random.uniform(1.5, 3.0))
            self._client.get("https://www.cian.ru/kupit-kvartiru/")
            time.sleep(random.uniform(1.5, 2.5))
        except httpx.HTTPError as exc:
            logger.warning("Не удалось инициализировать сессию: %s", exc)
        self._session_ready = True

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=3, min=5, max=60),
        retry=retry_if_exception_type((httpx.HTTPError, CaptchaError)),
    )
    def fetch_page(
        self,
        *,
        region_id: int,
        deal_type: str,
        offer_type: str,
        page: int,
    ) -> dict[str, Any]:
        if not self._session_ready:
            self._init_session()

        body = _build_json_query(
            region_id=region_id,
            deal_type=deal_type,
            offer_type=offer_type,
            page=page,
        )
        resp = self._client.post(
            SEARCH_URL,
            json=body,
            headers=API_HEADERS,
        )

        # Captcha редирект: Циан возвращает 302 → /cian-captcha/
        if "captcha" in str(resp.url).lower():
            logger.warning("Получена капча (страница %s) — пауза и повтор…", page)
            time.sleep(random.uniform(15, 30))
            raise CaptchaError(f"captcha redirect on page {page}")

        resp.raise_for_status()

        # На случай если ответ — HTML (не JSON)
        ct = resp.headers.get("content-type", "")
        if "json" not in ct and not resp.content.strip().startswith(b"{"):
            logger.warning("Неожиданный content-type: %s, страница %s", ct, page)
            raise CaptchaError(f"non-JSON response on page {page}")

        return resp.json()


def iter_listings(
    client: CianClient,
    *,
    region_id: int,
    deal_type: str,
    offer_type: str,
    pages: int,
    sleep_range: tuple[float, float] = (2.0, 4.5),
) -> Iterable[tuple[Listing, dict[str, Any]]]:
    for page_num in tqdm(range(1, pages + 1), desc="pages"):
        try:
            payload = client.fetch_page(
                region_id=region_id,
                deal_type=deal_type,
                offer_type=offer_type,
                page=page_num,
            )
        except CaptchaError as exc:
            logger.error("Капча не пройдена после всех ретраев: %s", exc)
            logger.error("Попробуйте запустить скрипт позже или уменьшите --pages.")
            break
        except httpx.HTTPStatusError as exc:
            logger.warning("HTTP %s на странице %s: %s", exc.response.status_code, page_num, exc)
            if exc.response.status_code in (403, 429):
                logger.info("Ограничение сервера — пауза 30-60 сек…")
                time.sleep(random.uniform(30, 60))
            continue
        except httpx.HTTPError as exc:
            logger.warning("страница %s не загружена: %s", page_num, exc)
            continue

        data = payload.get("data") or {}
        items = data.get("offersSerialized") or []

        if not items:
            logger.info("пустой результат на странице %s — останавливаемся", page_num)
            break

        for raw in items:
            try:
                yield parse_offer(raw), raw
            except Exception as exc:
                logger.warning("не смогли распарсить объявление: %s", exc)

        time.sleep(random.uniform(*sleep_range))


CSV_FIELDS = [
    "offer_id",
    "url",
    "deal_type",
    "object_type",
    "price",
    "price_per_m2",
    "area_total",
    "area_living",
    "area_kitchen",
    "rooms",
    "floor",
    "floors_total",
    "address",
    "city",
    "underground",
    "latitude",
    "longitude",
    "year_built",
    "house_material",
    "has_balcony",
    "description",
    "image_urls",
]


def run(args: argparse.Namespace) -> None:
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_jsonl = Path(args.raw) if args.raw else out_csv.with_suffix(".jsonl")
    raw_jsonl.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_written = 0

    with (
        CianClient() as client,
        out_csv.open("w", encoding="utf-8", newline="") as csv_fh,
        raw_jsonl.open("w", encoding="utf-8") as raw_fh,
    ):
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for listing, raw in iter_listings(
            client,
            region_id=args.region,
            deal_type=args.deal_type,
            offer_type=args.offer_type,
            pages=args.pages,
        ):
            if not listing.offer_id or listing.offer_id in seen:
                continue
            seen.add(listing.offer_id)
            writer.writerow(listing.to_csv_row())
            raw_fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info("сохранено %s объявлений → %s", n_written, out_csv)
    print(f"OK: {n_written} объявлений → {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--region",
        type=int,
        default=1,
        help="ID региона Cian (1=Москва, 2=СПб, 4580=Екб и т.д.)",
    )
    p.add_argument("--deal-type", dest="deal_type", default="sale", choices=["sale", "rent"])
    p.add_argument(
        "--offer-type",
        dest="offer_type",
        default="flat",
        choices=["flat", "room", "house"],
    )
    p.add_argument("--pages", type=int, default=20, help="сколько страниц по 20 объявлений")
    p.add_argument("--out", default="data/processed/listings.csv")
    p.add_argument("--raw", default=None, help="путь к JSONL с сырыми объектами")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
