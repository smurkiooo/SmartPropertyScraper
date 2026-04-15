# realty-scraper

Проект: сбор структурированных данных, текстовых описаний и фотографий
объявлений о недвижимости с российского портала **cian.ru**.

## Источник данных

[cian.ru](https://cian.ru) — крупнейший российский агрегатор объявлений
о покупке и аренде недвижимости. Используется публичный JSON POST-эндпоинт
`https://api.cian.ru/search-offers/v2/search-offers-desktop/`, который
вызывает фронтенд сайта при обычном поиске.

Для каждого объявления собираются три типа артефактов:

| Тип артефакта      | Поля                                                                                                                                               |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Структурированные  | `price`, `price_per_m2`, `area_total/living/kitchen`, `rooms`, `floor`, `floors_total`, `latitude`, `longitude`, `year_built`, `house_material`, … |
| Текстовое описание | `description` — свободный текст автора объявления                                                                                                  |
| Изображения        | `image_urls` — список ссылок на фото интерьеров и планировок (через `;`)                                                                           |

Готовые CSV с Kaggle **не используются** — данные собираются скрэйпером.

## Структура проекта

```
.
├── pyproject.toml          # uv-проект, зависимости
├── uv.lock
├── .gitignore              # игнорирует data/, .venv, кэш DVC
├── .dvc/                   # конфигурация DVC
├── src/
│   └── realty_scraper/
│       ├── __init__.py
│       └── cian.py         # клиент API + парсер + CLI
├── data/
│   ├── raw/                # сырые JSONL-ответы API (под DVC)
│   ├── processed/          # сводный CSV (под DVC)
│   └── images/             # скачанные фото (под DVC, ДЗ №2)
└── notebooks/              # ноутбуки для EDA
```

## Настройка окружения

Требуются [`uv`](https://docs.astral.sh/uv/) и Git.

```bash
# 1. Клонировать репозиторий
git clone <repo-url> realty-scraper && cd realty-scraper

# 2. Создать виртуальное окружение и установить зависимости
uv sync

# 3. Подтянуть данные из DVC-хранилища (если уже есть снэпшот)
uv run dvc pull
```

## Сбор данных

```bash
# ~480 объявлений о продаже квартир в Москве (по умолчанию)
uv run python -m realty_scraper.cian --pages 20 --out data/processed/listings.csv

# ~1000 объявлений
uv run python -m realty_scraper.cian --pages 50 --out data/processed/listings.csv

# Аренда квартир
uv run python -m realty_scraper.cian --deal-type rent --pages 20 --out data/processed/rent.csv

# Санкт-Петербург
uv run python -m realty_scraper.cian --region 2 --pages 20 --out data/processed/spb.csv

# Подробный лог
uv run python -m realty_scraper.cian --pages 20 -v --out data/processed/listings.csv
```

Аргументы CLI:

| Флаг            | По умолчанию                  | Описание                                       |
|-----------------|-------------------------------|------------------------------------------------|
| `--region`      | `1` (Москва)                  | ID региона cian.ru (1=Москва, 2=СПб)           |
| `--deal-type`   | `sale`                        | `sale` / `rent`                                |
| `--offer-type`  | `flat`                        | `flat`, `room`, `house`                        |
| `--pages`       | `20`                          | страниц по ~24 объявления                      |
| `--out`         | `data/processed/listings.csv` | путь к итоговому CSV                           |
| `--raw`         | рядом с CSV (`.jsonl`)        | сырые ответы API (для DVC и отладки)           |
| `-v`            | —                             | подробный лог                                  |

Скрипт инициализирует браузерную сессию (посещает главную страницу и страницу
поиска для получения cookies), делает паузы между запросами и использует
экспоненциальные ретраи через `tenacity`.

## Поля CSV

| Поле             | Описание                                      |
|------------------|-----------------------------------------------|
| `offer_id`       | ID объявления на cian.ru                      |
| `url`            | Ссылка на объявление                          |
| `deal_type`      | Тип сделки (`sale` / `rent`)                  |
| `object_type`    | Тип объекта (`flatSale`, `newBuildingFlatSale`, …) |
| `price`          | Цена, ₽                                       |
| `price_per_m2`   | Цена за м², ₽                                 |
| `area_total`     | Общая площадь, м²                             |
| `area_living`    | Жилая площадь, м²                             |
| `area_kitchen`   | Площадь кухни, м²                             |
| `rooms`          | Количество комнат                             |
| `floor`          | Этаж                                          |
| `floors_total`   | Этажей в доме                                 |
| `address`        | Адрес (район, улица, дом)                     |
| `city`           | Город                                         |
| `underground`    | Ближайшая станция метро                       |
| `latitude`       | Широта                                        |
| `longitude`      | Долгота                                       |
| `year_built`     | Год постройки                                 |
| `house_material` | Материал стен (`monolith`, `brick`, …)        |
| `has_balcony`    | Наличие балкона                               |
| `description`    | Текстовое описание от автора                  |
| `image_urls`     | URL фотографий через `;`                      |

## DVC

Тяжёлые артефакты (CSV, JSONL, фото) исключены из Git через `.gitignore`
и хранятся под DVC. После сбора данных:

```bash
uv run dvc add data/processed/listings.csv data/processed/listings.jsonl
uv run dvc push
git add data/processed/listings.csv.dvc data/processed/listings.jsonl.dvc
git commit -m "data: snapshot listings"
```

Текущее удалённое хранилище — локальная папка `~/dvc-storage-realty`.

