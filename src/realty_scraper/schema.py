from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


MIN_PRICE = 100_000.0
MAX_PRICE = 3_000_000_000.0


MIN_AREA_TOTAL = 8.0
MAX_AREA_TOTAL = 1_000.0
MIN_AREA_PART = 1.0        
MAX_AREA_PART = 500.0


MIN_ROOMS = 0
MAX_ROOMS = 20


MIN_FLOOR = 1
MAX_FLOOR = 100

# от Чёрного моря до Арктики)
MIN_LAT, MAX_LAT = 41.0, 82.0
MIN_LON, MAX_LON = 19.0, 180.0


MIN_YEAR, MAX_YEAR = 1700, 2035
MIN_PRICE_PER_M2 = 1_000.0
MAX_PRICE_PER_M2 = 10_000_000.0


LISTING_SCHEMA = DataFrameSchema(
    columns={

        "offer_id": Column(
            str,
            checks=Check.str_matches(r"^\d+$"),
            unique=True,
            nullable=False,
        ),
        "url": Column(str, Check.str_startswith("http"), nullable=False),
        "deal_type": Column(str, Check.isin(["sale", "rent"]), nullable=False),
        "object_type": Column(str, nullable=True),


        "price": Column(
            float,
            checks=Check.in_range(MIN_PRICE, MAX_PRICE),
            nullable=False,
        ),
    
        "area_total": Column(
            float,
            checks=Check.in_range(MIN_AREA_TOTAL, MAX_AREA_TOTAL),
            nullable=False,
        ),
     


        "rooms": Column(
            "Int64",
            checks=Check.in_range(MIN_ROOMS, MAX_ROOMS),
            nullable=True,
        ),
        "floor": Column(
            "Int64",
            checks=Check.in_range(MIN_FLOOR, MAX_FLOOR),
            nullable=True,
        ),
        "floors_total": Column(
            "Int64",
            checks=Check.in_range(MIN_FLOOR, MAX_FLOOR),
            nullable=True,
        ),


        "address": Column(str, Check.str_length(min_value=1), nullable=False),
        "city": Column(str, nullable=True),
        "underground": Column(str, nullable=True),


        "latitude": Column(
            float,
            checks=Check.in_range(MIN_LAT, MAX_LAT),
            nullable=False,
        ),
        "longitude": Column(
            float,
            checks=Check.in_range(MIN_LON, MAX_LON),
            nullable=False,
        ),

        "description": Column(str, nullable=True),
        "image_uris": Column(
            str,
            checks=Check(
                lambda s: s.fillna("").apply(
                    lambda v: v == "" or all(u.startswith("s3://") for u in v.split(";") if u)
                ),
                error="image_uris должен содержать только s3:// URI через ';'",
            ),
            nullable=True,
        ),
    },

    checks=[
        Check(
            lambda df: (
                df["floor"].isna()
                | df["floors_total"].isna()
                | (df["floor"] <= df["floors_total"])
            ),
            error="floor > floors_total",
        )],
    coerce=True,
    strict=False,
)


__all__ = ["LISTING_SCHEMA"]
