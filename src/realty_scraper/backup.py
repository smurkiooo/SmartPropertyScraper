"""
бэкап проиндексированных текстовых описаний в json на случай перезапуска докер контейнера

использоание:
    uv run python -m realty_scraper.backup              # сохранить бэкап
    uv run python -m realty_scraper.backup --restore    # восстановить из бэкапа
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


def restore(backup_path: Path, chroma_host: str, chroma_port: int) -> None:
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    with open(backup_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    listings = all_data.get("listings", [])
    print(f"Восстановление {len(listings)} записей из бэкапа...")

    batch_size = 100
    for i in range(0, len(listings), batch_size):
        batch = listings[i : i + batch_size]
        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["description"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )
        print(f"  {min(i + batch_size, len(listings))}/{len(listings)}")

    print(f"Восстановление завершено. Коллекция '{COLLECTION_NAME}' содержит {collection.count()} записей.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup/Restore ChromaDB")
    parser.add_argument("--backup-path", type=Path, default=DEFAULT_BACKUP_PATH)
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8000)
    parser.add_argument("--restore", action="store_true", help="Restore from backup instead of creating one")
    args = parser.parse_args()

    if args.restore:
        restore(args.backup_path, args.chroma_host, args.chroma_port)
    else:
        run(args.backup_path, args.chroma_host, args.chroma_port)


if __name__ == "__main__":
    main()
