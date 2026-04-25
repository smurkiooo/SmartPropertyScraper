"""
Index listings_s3.csv into ChromaDB.

Usage:
    uv run python -m realty_scraper.index [--csv PATH] [--batch-size N] [--chroma-host HOST]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from realty_scraper.text_utils import clean_description

COLLECTION_NAME = "listings"
DEFAULT_CSV = Path("data/processed/listings_s3.csv")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_BATCH = 32


def _load_model() -> SentenceTransformer:
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    model.max_seq_length = 4096
    return model


def _load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"offer_id", "url", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["description"] = df["description"].fillna("").apply(clean_description)
    df = df[df["description"].str.len() > 0].reset_index(drop=True)
    print(f"Loaded {len(df)} listings with non-empty descriptions.")
    return df


def _metadata(row: pd.Series) -> dict:
    def _safe_num(val):
        if pd.isna(val):
            return 0
        if isinstance(val, float) and val == int(val):
            return int(val)
        return val

    return {
        "url": str(row.get("url", "")),
        "price": _safe_num(row.get("price")),
        "rooms": _safe_num(row.get("rooms")),
        "area_total": _safe_num(row.get("area_total")),
        "floor": _safe_num(row.get("floor")),
        "floors_total": _safe_num(row.get("floors_total")),
        "city": str(row.get("city", "") or ""),
        "address": str(row.get("address", "") or ""),
        "deal_type": str(row.get("deal_type", "") or ""),
        "underground": str(row.get("underground", "") or ""),
    }


def run(csv_path: Path, batch_size: int, chroma_host: str, chroma_port: int) -> None:
    df = _load_data(csv_path)
    model = _load_model()

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing = set(collection.get(include=[])["ids"])
    df = df[~df["offer_id"].astype(str).isin(existing)].reset_index(drop=True)

    if df.empty:
        print("All listings already indexed. Nothing to do.")
        return

    print(f"Indexing {len(df)} new listings in batches of {batch_size}...")

    for start in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[start : start + batch_size]

        texts = batch["description"].tolist()
        # Documents are encoded without instruction prefix (asymmetric search)
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=batch["offer_id"].astype(str).tolist(),
            embeddings=embeddings,
            documents=texts,
            metadatas=[_metadata(row) for _, row in batch.iterrows()],
        )

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} entries.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index listings into ChromaDB")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8000)
    args = parser.parse_args()

    run(args.csv, args.batch_size, args.chroma_host, args.chroma_port)


if __name__ == "__main__":
    main()
