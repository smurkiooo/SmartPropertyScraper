"""
Финальная обработка данных:

1. читает CSV от скрэйпера (после этапа `images.py` — с колонкой `image_uris`),
2. приводит типы (числа, nullable-инты, bool),
3. валидирует через Pandera (`realty_scraper.schema.LISTING_SCHEMA`),
4. отсеивает аномалии (строки, не прошедшие проверки) и логирует их,
5. сохраняет результат в .parquet.

Пример:
    uv run python -m realty_scraper.clean \
        --in data/processed/listings_s3.csv \
        --out data/processed/listings.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from .schema import LISTING_SCHEMA

logger = logging.getLogger("realty_scraper.clean")

#offer_id,url,deal_type,object_type,price,area_total,rooms,floor,floors_total,address,city,underground,latitude,longitude,description,image_uris


NUMERIC_COLS = [
    "price",
    "area_total",
    "latitude",
    "longitude",
]
NULLABLE_INT_COLS = ["rooms", "floor", "floors_total"]
STR_COLS = [
    "offer_id",
    "url",
    "deal_type",
    "object_type",
    "address",
    "city",
    "underground",
    "description",
    "image_uris",
]


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит колонки к ожидаемым типам (NaN при ошибке)."""
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in NULLABLE_INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def validate_and_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    try:
        validated = LISTING_SCHEMA.validate(df, lazy=True)
        return validated, None
    except SchemaErrors as errs:
        failures = errs.failure_cases.copy()
       
        bad_idx = (
            failures["index"].dropna().astype("int64").unique().tolist()
            if "index" in failures.columns
            else []
        )
        logger.warning(
            "обнаружено %s строк с ошибками; детали по колонкам:",
            len(bad_idx),
        )
        if "column" in failures.columns:
            by_col = (
                failures.groupby("column", dropna=False)["failure_case"]
                .count()
                .sort_values(ascending=False)
            )
            for col, n in by_col.items():
                logger.warning("  %s: %s нарушений", col, n)
        cleaned = df.drop(index=bad_idx).reset_index(drop=True)
        try:
            validated = LISTING_SCHEMA.validate(cleaned, lazy=False)
        except SchemaError as exc:
            logger.error("повторная валидация упала: %s", exc)
            raise
        return validated, failures


def run(args: argparse.Namespace) -> None:
    in_path = Path(args.inp)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("читаем %s", in_path)
    df = pd.read_csv(in_path, dtype=str, keep_default_na=True, na_values=[""])
    logger.info("исходных строк: %s", len(df))

    df = coerce_types(df)

    valid_df, failures = validate_and_filter(df)
    logger.info("после валидации осталось: %s строк (отсеяно %s)",
                len(valid_df), len(df) - len(valid_df))

    if args.failures_out and failures is not None and not failures.empty:
        fpath = Path(args.failures_out)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        failures.to_csv(fpath, index=False)
        logger.info("детальный список ошибок → %s", fpath)

    valid_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("итог → %s (%s строк, %s колонок)", out_path, len(valid_df), len(valid_df.columns))
    print(f"OK: {len(valid_df)} строк → {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="data/processed/listings_s3.csv")
    p.add_argument("--out", default="data/processed/listings.parquet")
    p.add_argument(
        "--failures-out",
        dest="failures_out",
        default="data/processed/listings_failures.csv",
        help="CSV с описанием всех выявленных аномалий (для ревью)",
    )
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
