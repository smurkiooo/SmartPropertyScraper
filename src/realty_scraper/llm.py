"""
LM Studio клиент для генерации описаний квартир.

LM Studio предоставляет OpenAI-совместимый API на localhost:1234.
"""

from __future__ import annotations

from openai import OpenAI

from realty_scraper.search import SearchResult

DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"

DEFAULT_MODEL = "local-model"
DESCRIPTION_LIMIT = 700


SYSTEM_PROMPT = """\
Ты — профессиональный ассистент по подбору недвижимости в Москве. \
Тебе на вход подаётся запрос покупателя и список найденных объявлений. \
Для каждого объявления напиши 3–4 предложения на русском языке: \
выдели ключевые преимущества, объясни почему квартира подходит под запрос, \
упомяни этаж, площадь, метро и цену в удобном формате. \
Пиши живо и информативно, без лишних вводных слов.\
"""


def _format_listing(index: int, result: SearchResult) -> str:
    price_str = f"{result.price:,} ₽".replace(",", " ") if result.price else "цена не указана"
    rooms_str = f"{result.rooms}-комн." if result.rooms else "студия"
    area_str = f"{result.area_total} м²" if result.area_total else ""
    floor_str = (
        f"{result.floor} из {result.floors_total} эт."
        if result.floor and result.floors_total
        else ""
    )
    metro_str = f"м. {result.underground}" if result.underground else ""
    description = result.description[:DESCRIPTION_LIMIT]
    if len(result.description) > DESCRIPTION_LIMIT:
        description += "..."

    parts = filter(None, [rooms_str, area_str, floor_str, price_str, metro_str, result.address])
    header = " | ".join(parts)

    return f"Вариант {index}.\n{header}\nОписание: {description}"


def _build_prompt(query: str, results: list[SearchResult]) -> str:
    listings_text = "\n\n".join(
        _format_listing(i, r) for i, r in enumerate(results, 1)
    )
    return (
        f"Запрос покупателя: «{query}»\n\n"
        f"Найденные объявления:\n\n"
        f"{listings_text}\n\n"
        f"Дай краткое описание каждого варианта с учётом запроса покупателя."
    )


class LMStudioClient:
    def __init__(
        self,
        base_url: str = DEFAULT_LM_STUDIO_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key="lm-studio")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def check_connection(self) -> bool:
        """Проверяет доступность LM Studio сервера."""
        try:
            models = self._client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    def generate(self, query: str, results: list[SearchResult]) -> str:
        """
        Генерирует текстовое описание квартир по запросу.
        Возвращает стриминговый вывод прямо в консоль и итоговую строку.
        """
        prompt = _build_prompt(query, results)

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )

        collected = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                collected.append(delta)
        print()  

        return "".join(collected)
