"""
Semantic search over indexed listings.

Programmatic usage:
    from realty_scraper.search import ApartmentSearch
    search = ApartmentSearch()
    results = search.query("двушка рядом с метро с ремонтом", n_results=5)

CLI usage:
    uv run python -m realty_scraper.search "запрос" [--top N] [--deal-type sale|rent]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from realty_scraper.text_utils import clean_description

COLLECTION_NAME = "listings"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@dataclass
class SearchResult:
    url: str
    score: float
    description: str
    price: int | None
    rooms: int | None
    area_total: float | None
    floor: int | None
    floors_total: int | None
    address: str
    city: str
    deal_type: str
    underground: str

    def __str__(self) -> str:
        price_str = f"{self.price:,} ₽".replace(",", " ") if self.price else "—"
        area_str = f"{self.area_total} м²" if self.area_total else "—"
        rooms_str = str(self.rooms) if self.rooms is not None else "студия/неизв."
        floor_str = (
            f"{self.floor}/{self.floors_total}"
            if self.floor and self.floors_total
            else str(self.floor or "—")
        )
        metro_str = f" · м. {self.underground}" if self.underground else ""
        return (
            f"[{self.score:.3f}] {self.url}\n"
            f"  {rooms_str} комн · {area_str} · {floor_str} эт · {price_str}\n"
            f"  {self.address}{metro_str}\n"
            f"  {self.description[:120]}..."
        )


class ApartmentSearch:
    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
    ) -> None:
        self._model: SentenceTransformer | None = None
        self._collection: Any | None = None
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
            self._model.max_seq_length = 4096
        return self._model

    def _get_collection(self) -> Any:
        if self._collection is None:
            client = chromadb.HttpClient(host=self._chroma_host, port=self._chroma_port)
            self._collection = client.get_collection(COLLECTION_NAME)
        return self._collection

    def query(
        self,
        text: str,
        n_results: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        cleaned = clean_description(text)
        model = self._get_model()

        embedding = model.encode(cleaned).tolist()

        collection = self._get_collection()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        output: list[SearchResult] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite → convert to similarity
            score = 1.0 - dist / 2.0
            output.append(
                SearchResult(
                    url=meta.get("url", ""),
                    score=score,
                    description=doc,
                    price=meta.get("price"),
                    rooms=meta.get("rooms"),
                    area_total=meta.get("area_total"),
                    floor=meta.get("floor"),
                    floors_total=meta.get("floors_total"),
                    address=meta.get("address", ""),
                    city=meta.get("city", ""),
                    deal_type=meta.get("deal_type", ""),
                    underground=meta.get("underground", ""),
                )
            )

        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Search apartments by text query")
    parser.add_argument("query", help="Search query in Russian")
    parser.add_argument("--top", type=int, default=5, help="Number of results")
    parser.add_argument("--deal-type", choices=["sale", "rent"], help="Filter by deal type")
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8000)
    args = parser.parse_args()

    filters = {"deal_type": args.deal_type} if args.deal_type else None

    search = ApartmentSearch(chroma_host=args.chroma_host, chroma_port=args.chroma_port)
    results = search.query(args.query, n_results=args.top, filters=filters)

    if not results:
        print("Ничего не найдено.")
        return

    print(f"\nТоп-{len(results)} результатов для: «{args.query}»\n" + "─" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r}")


if __name__ == "__main__":
    main()
