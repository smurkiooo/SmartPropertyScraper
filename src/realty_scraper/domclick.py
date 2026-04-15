"""
Сборщик объявлений о вторичной недвижимости с domclick.ru.

Используется публичный JSON-эндпоинт offers-service, который вызывает
фронтенд сайта при поиске. Скрипт собирает структурированные признаки,
текстовое описание автора и ссылки на изображения, после чего сохраняет
данные в CSV (поля + описание) и JSONL (полный сырой ответ).

Пример запуска:
    uv run python -m realty_scraper.domclick \
        --region 81 --deal-type sale --category living \
        --offer-type flat --pages 20 --out data/processed/listings.csv
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

API_URL = "https://offers-service.domclick.ru/research/v5/offers/"
PAGE_SIZE = 20

DEFAULT_HEADERS = {
    # Притворяемся обычным браузером — без этого endpoint иногда отдаёт 403.
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru,en;q=0.9",
    "Origin": "https://domclick.ru",
    "Referer": "https://domclick.ru/",
}

logger = logging.getLogger("realty_scraper.domclick")


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
    region: str
    city: str
    latitude: float | None
    longitude: float | None
    year_built: int | None
    house_material: str
    has_balcony: bool | None
    description: str
    image_urls: list[str] = field(default_factory=list)

    def to_csv_row(self) -> dict[str, Any]:
        row = asdict(self)
        # CSV не любит списки — склеиваем url-ы через точку с запятой
        row["image_urls"] = ";".join(self.image_urls)
        # описание экранируем от переносов, чтобы не ломать строки CSV
        row["description"] = (self.description or "").replace("\r", " ").replace("\n", " ").strip()
        return row


def _safe_get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def parse_offer(raw: dict[str, Any]) -> Listing:
    """Достаёт поля из сырого JSON-объекта объявления Domclick."""

    offer_id = str(raw.get("id") or raw.get("offer_id") or "")
    slug = raw.get("slug") or offer_id
    url = f"https://domclick.ru/card/{slug}" if slug else ""

    object_info = raw.get("object_info") or {}
    house = raw.get("house") or {}
    address = raw.get("address") or {}
    coords = address.get("position") or {}

    images: list[str] = []
    for photo in raw.get("photos") or []:
        # У domclick фото лежат либо как готовый url, либо шаблон с {size}
        u = photo.get("url") or photo.get("link")
        if not u:
            continue
        if "{size}" in u:
            u = u.replace("{size}", "1024x768")
        images.append(u)

    return Listing(
        offer_id=offer_id,
        url=url,
        deal_type=str(raw.get("deal_type") or ""),
        object_type=str(raw.get("object_type") or raw.get("offer_type") or ""),
        price=_safe_get(raw, "price_info", "price"),
        price_per_m2=_safe_get(raw, "price_info", "square_price"),
        area_total=object_info.get("area"),
        area_living=object_info.get("living_area"),
        area_kitchen=object_info.get("kitchen_area"),
        rooms=object_info.get("rooms"),
        floor=object_info.get("floor"),
        floors_total=house.get("floors") or object_info.get("floors"),
        address=address.get("display_name") or address.get("name") or "",
        region=_safe_get(address, "subject", "display_name", default=""),
        city=_safe_get(address, "locality", "display_name", default=""),
        latitude=coords.get("lat"),
        longitude=coords.get("lon") or coords.get("lng"),
        year_built=house.get("build_year"),
        house_material=str(house.get("housing_type") or house.get("material") or ""),
        has_balcony=object_info.get("has_balcony"),
        description=str(raw.get("description") or "").strip(),
        image_urls=images,
    )


class DomclickClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            http2=False,
            follow_redirects=True,
        )

    def __enter__(self) -> "DomclickClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=2, max=20),
        retry=retry_if_exception_type((httpx.HTTPError,)),
    )
    def fetch_page(
        self,
        *,
        region_id: int,
        deal_type: str,
        category: str,
        offer_type: str,
        offset: int,
    ) -> dict[str, Any]:
        params = {
            "address": f"region={region_id}",
            "deal_type": deal_type,
            "category": category,
            "offer_type": offer_type,
            "offset": offset,
            "limit": PAGE_SIZE,
            "sort": "qi",
            "sort_dir": "desc",
        }
        resp = self._client.get(API_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def iter_listings(
    client: DomclickClient,
    *,
    region_id: int,
    deal_type: str,
    category: str,
    offer_type: str,
    pages: int,
    sleep_range: tuple[float, float] = (0.7, 1.6),
) -> Iterable[tuple[Listing, dict[str, Any]]]:
    for page_idx in tqdm(range(pages), desc="pages"):
        offset = page_idx * PAGE_SIZE
        try:
            payload = client.fetch_page(
                region_id=region_id,
                deal_type=deal_type,
                category=category,
                offer_type=offer_type,
                offset=offset,
            )
        except httpx.HTTPError as exc:
            logger.warning("страница offset=%s провалилась: %s", offset, exc)
            continue

        # Domclick возвращает либо {"result": {"items": [...]}}, либо {"items": [...]}
        items = (
            payload.get("items")
            or _safe_get(payload, "result", "items", default=[])
            or []
        )
        if not items:
            logger.info("пустая страница offset=%s — останавливаемся", offset)
            break

        for raw in items:
            try:
                yield parse_offer(raw), raw
            except Exception as exc:  # pragma: no cover — оборонительная ветка
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
    "region",
    "city",
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
        DomclickClient() as client,
        out_csv.open("w", encoding="utf-8", newline="") as csv_fh,
        raw_jsonl.open("w", encoding="utf-8") as raw_fh,
    ):
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for listing, raw in iter_listings(
            client,
            region_id=args.region,
            deal_type=args.deal_type,
            category=args.category,
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
    p.add_argument("--region", type=int, default=81, help="ID региона domclick (81 = Москва)")
    p.add_argument("--deal-type", default="sale", choices=["sale", "rent"])
    p.add_argument("--category", default="living")
    p.add_argument(
        "--offer-type",
        dest="offer_type",
        default="flat",
        choices=["flat", "room", "house", "townhouse", "land"],
    )
    p.add_argument("--pages", type=int, default=20, help="сколько страниц по 20 объявлений собрать")
    p.add_argument("--out", default="data/processed/listings.csv")
    p.add_argument("--raw", default=None, help="путь к JSONL с сырыми объектами (по умолчанию рядом с CSV)")
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
