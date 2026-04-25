"""
бэкап проиндексированных текстовых описаний в json на случай перезапуска докер контейнера

использоание:
    uv run python -m realty_scraper.backup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb

COLLECTION_NAME = "listings"
DEFAULT_BACKUP_PATH = Path("data/backups/listings_backup.json")


def run(backup_path: Path, chroma_host: str, chroma_port: int) -> None:
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_collection(COLLECTION_NAME)

    total = collection.count()
    print(f"резервное копирование {total} записей из хрома дб...")

    batch_size = 100
    all_data = {"listings": []}

    for start in range(0, total, batch_size):
        data = collection.get(offset=start, limit=batch_size, include=["documents", "metadatas"])
        for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            all_data["listings"].append({
                "id": id_,
                "description": doc,
                "metadata": meta,
            })
        print(f"  {min(start + batch_size, total)}/{total}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"резервная копия сохранена по пути: {backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup ChromaDB")
    parser.add_argument("--backup-path", type=Path, default=DEFAULT_BACKUP_PATH)
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8000)
    args = parser.parse_args()

    run(args.backup_path, args.chroma_host, args.chroma_port)


if __name__ == "__main__":
    main()
