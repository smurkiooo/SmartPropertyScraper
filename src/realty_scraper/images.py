"""
Скачивает картинки по HTTP-ссылкам из CSV скрэйпера, заливает их
в S3/MinIO и записывает новый CSV, где `image_urls` заменены на
`image_uris` — S3-пути вида `s3://realty/images/<offer_id>/<NN>.jpg`.

Пример:
    uv run python -m realty_scraper.images \
        --in data/processed/listings.csv \
        --out data/processed/listings_s3.csv \
        --bucket realty --workers 8 --max-per-offer 10
"""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .s3_utils import (
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    ensure_bucket,
    get_s3_client,
    object_exists,
)

logger = logging.getLogger("realty_scraper.images")

IMG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.cian.ru/",
}


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
def _download(url: str, client: httpx.Client) -> tuple[bytes, str]:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.content, resp.headers.get("content-type", "image/jpeg")


def _upload(s3_client: Any, bucket: str, key: str, data: bytes, content_type: str) -> None:
    s3_client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def process_offer(
    *,
    offer_id: str,
    urls: list[str],
    http_client: httpx.Client,
    s3_client: Any,
    bucket: str,
    max_per_offer: int,
    skip_existing: bool,
) -> list[str]:
    """Скачивает до max_per_offer фото и заливает в S3; возвращает S3 URI."""
    uris: list[str] = []
    for i, url in enumerate(urls[:max_per_offer]):
        url = (url or "").strip()
        if not url:
            continue
        key = f"images/{offer_id}/{i:03d}.jpg"
        if skip_existing and object_exists(s3_client, bucket, key):
            uris.append(f"s3://{bucket}/{key}")
            continue
        try:
            data, ct = _download(url, http_client)
            _upload(s3_client, bucket, key, data, ct)
            uris.append(f"s3://{bucket}/{key}")
        except Exception as exc:  # сетевой сбой, 404 и т.п.
            logger.warning("offer=%s img=%s %s", offer_id, i, exc)
    return uris


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client(endpoint_url=args.endpoint)
    ensure_bucket(s3, args.bucket)
    logger.info("бакет %s/%s готов", args.endpoint, args.bucket)

    rows = list(csv.DictReader(in_path.open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]
    logger.info("обрабатываем %s объявлений", len(rows))

    if not rows:
        logger.warning("входной CSV пуст")
        return

    # Колонка image_urls заменится на image_uris
    fieldnames = [f for f in rows[0].keys() if f != "image_urls"] + ["image_uris"]

    http_client = httpx.Client(headers=IMG_HEADERS, timeout=30.0, follow_redirects=True)

    try:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                future_to_row: dict[Any, dict[str, str]] = {}
                for row in rows:
                    offer_id = row.get("offer_id", "")
                    urls = [u for u in (row.get("image_urls") or "").split(";") if u]
                    fut = pool.submit(
                        process_offer,
                        offer_id=offer_id,
                        urls=urls,
                        http_client=http_client,
                        s3_client=s3,
                        bucket=args.bucket,
                        max_per_offer=args.max_per_offer,
                        skip_existing=args.skip_existing,
                    )
                    future_to_row[fut] = row

                for fut in tqdm(
                    as_completed(future_to_row),
                    total=len(future_to_row),
                    desc="offers",
                ):
                    row = future_to_row[fut]
                    try:
                        uris = fut.result()
                    except Exception as exc:
                        logger.warning("offer=%s: %s", row.get("offer_id"), exc)
                        uris = []
                    out_row = {k: v for k, v in row.items() if k != "image_urls"}
                    out_row["image_uris"] = ";".join(uris)
                    writer.writerow(out_row)
    finally:
        http_client.close()

    logger.info("готово → %s", out_path)
    print(f"OK: {len(rows)} объявлений обработано → {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="data/processed/listings100_cleaned.csv")
    p.add_argument("--out", default="data/processed/listings_s3.csv")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--workers", type=int, default=8, help="одновременных загрузок")
    p.add_argument("--max-per-offer", dest="max_per_offer", type=int, default=10,
                   help="сколько фото максимум на одно объявление")
    p.add_argument("--limit", type=int, default=0, help="обработать только первые N объявлений (0 = все)")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                   help="не заливать фото, если такой ключ уже есть в бакете")
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
