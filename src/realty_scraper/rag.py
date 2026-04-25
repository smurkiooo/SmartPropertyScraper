"""
Полный RAG пайплайн: семантический поиск + LLM-генерация через LM Studio.

Использование:
    uv run python -m realty_scraper.rag "двушка рядом с метро с ремонтом"
    uv run python -m realty_scraper.rag "студия с видом на реку" --top 3
    uv run python -m realty_scraper.rag "трёшка" --top 5 --deal-type sale
    uv run python -m realty_scraper.rag "пентхаус" --no-llm   # только поиск
"""

from __future__ import annotations

import argparse

from realty_scraper.llm import LMStudioClient
from realty_scraper.search import ApartmentSearch, SearchResult


MAX_LLM_RESULTS = 1


def _print_search_results(results: list[SearchResult]) -> None:
    """Выводит результаты поиска в виде краткой таблицы."""
    print(" Найденные объявления:\n" + "─" * 60)
    for i, r in enumerate(results, 1):
        price_str = f"{r.price:,} ₽".replace(",", " ") if r.price else "—"
        area_str = f"{r.area_total} м²" if r.area_total else "—"
        rooms_str = f"{r.rooms}-комн." if r.rooms else "студия"
        metro_str = f" · м. {r.underground}" if r.underground else ""
        print(
            f"{i}. [{r.score:.3f}] {rooms_str} · {area_str} · {price_str}{metro_str}\n"
            f"   {r.url}"
        )


def run(
    query: str,
    n_results: int = 3,
    deal_type: str | None = None,
    use_llm: bool = True,
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
    lm_studio_url: str = "http://localhost:1234/v1",
) -> None:
    print(f"Обрабатываю запрос...")
    filters = {"deal_type": deal_type} if deal_type else None

    search = ApartmentSearch(chroma_host=chroma_host, chroma_port=chroma_port)
    results = search.query(query, n_results=n_results, filters=filters)

    if not results:
        print("Ничего не найдено.")
        return

    _print_search_results(results)

    if not use_llm:
        return

    llm = LMStudioClient(base_url=lm_studio_url)

    print("Проверяю подключение к llm...")
    if not llm.check_connection():
        print(
            "LM Studio недоступен. Убедись что:\n"
            "   1. LM Studio запущен\n"
            "   2. Модель загружена\n"
            "   3. Сервер активен (кнопка 'Start Server' на вкладке Local Server)\n"
            f"   4. Адрес: {lm_studio_url}"
        )
        return

  
    llm_results = results[:MAX_LLM_RESULTS]
    llm.generate(query, llm_results)
    print("─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG поиск квартир с LLM-описанием"
    )
    parser.add_argument("query", help="Поисковый запрос на русском")
    parser.add_argument("--top", type=int, default=3, help="Количество результатов (макс. 5)")
    parser.add_argument("--deal-type", choices=["sale", "rent"], help="Тип сделки")
    parser.add_argument("--no-llm", action="store_true", help="Только поиск без LLM")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1")
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8000)
    args = parser.parse_args()

    n_results = min(args.top, 5)

    run(
        query=args.query,
        n_results=n_results,
        deal_type=args.deal_type,
        use_llm=not args.no_llm,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        lm_studio_url=args.lm_studio_url,
    )


if __name__ == "__main__":
    main()
