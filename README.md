# realty-scraper

Проект: сбор структурированных данных, текстовых описаний и фотографий
объявлений о недвижимости с российского портала **cian.ru**, загрузка фото
в локальный MinIO/S3, валидация Pandera-схемой и сохранение в parquet под
контролем DVC.

## Источник данных

[cian.ru](https://cian.ru) — крупнейший российский агрегатор объявлений
о покупке и аренде недвижимости. Используется публичный JSON POST-эндпоинт
`https://api.cian.ru/search-offers/v2/search-offers-desktop/`.

Для каждого объявления собираются три типа артефактов:

| Тип артефакта      | Поля                                                                          |
|--------------------|-------------------------------------------------------------------------------|
| Структурированные  | цена, площади, этажность, координаты, год постройки, материал стен и т.д.     |
| Текстовое описание | `description` — свободный текст автора                                        |
| Изображения        | фото интерьеров и планировок (после этапа 2 — S3 URI в MinIO)                 |



## Пайплайн

```
┌────────────┐     ┌────────────┐     ┌────────────────┐     ┌─────────────┐
│ cian.ru    │ ──▶ │ listings   │ ──▶ │ listings_s3    │ ──▶ │ listings    │
│ API        │     │ .csv (http │     │ .csv           │     │ .parquet    │
│            │     │  urls)     │     │ (s3://… URIs)  │     │ (clean)     │
└────────────┘     └────────────┘     └────────────────┘     └─────────────┘
                        │                    │                    │
                        │         ┌──────────▼────────┐           │
                        │         │ MinIO S3 bucket   │           │
                        │         │ realty/images/... │           │
                        │         └───────────────────┘           │
                        ▼                                         ▼
                      DVC                                        DVC
                 (sale snapshot)                          (clean snapshot)
```

## Структура проекта

```
.
├── pyproject.toml              # uv-проект, зависимости
├── uv.lock
├── docker-compose.yml          # MinIO + автосоздание бакетов
├── .env.example                # пример учёток MinIO и параметров S3
├── .gitignore
├── .dvc/                       # конфигурация DVC
├── src/
│   └── realty_scraper/
│       ├── __init__.py
│       ├── cian.py             # шаг 1: скрэйпер Cian API
│       ├── s3_utils.py         # boto3-клиент для MinIO
│       ├── images.py           # шаг 2: скачивание фото + заливка в S3
│       ├── schema.py           # Pandera-схема датасета
│       └── clean.py            # шаг 3: типизация + валидация + parquet
├── data/
│   ├── raw/                    # сырые JSONL-ответы API (под DVC)
│   ├── processed/              # CSV и итоговый parquet (под DVC)
│   └── images/                 # (не используется — фото в S3)
└── notebooks/                  # ноутбуки для EDA
```

## Настройка окружения

Требования: [`uv`](https://docs.astral.sh/uv/), Git, Docker Desktop.

```bash
# 1. Склонировать и установить зависимости
git clone <repo-url> realty-scraper && cd realty-scraper
uv sync

# 2. dvc-s3 ставится отдельно (конфликтует с uv resolver-ом из-за aiobotocore)
uv pip install "dvc[s3]"

# 3. Поднять MinIO (см. далее)
cp .env.example .env
docker compose up -d                    # запустит minio + создаст бакеты
# MinIO Web UI: http://localhost:9001  (minioadmin / minioadmin)
# S3 API:       http://localhost:9000
```

Контейнер `mc` автоматически создаст два бакета:
- `realty` — фото объявлений;
- `dvc-storage` — удалённое хранилище DVC.

## Использование

### Шаг 1. Сбор данных с cian.ru

```bash
uv run python -m realty_scraper.cian \
    --region 1 --deal-type sale --offer-type flat \
    --pages 20 \
    --out data/processed/listings.csv
```

Результат — `listings.csv` (структурированные поля + `image_urls` со ссылками
на `images.cdn-cian.ru`) и рядом `listings.jsonl` с сырыми ответами API.

Аргументы CLI:

| Флаг            | По умолчанию                  | Описание                                       |
|-----------------|-------------------------------|------------------------------------------------|
| `--region`      | `1` (Москва)                  | ID региона Cian (2 = СПб)                      |
| `--deal-type`   | `sale`                        | `sale` / `rent`                                |
| `--offer-type`  | `flat`                        | `flat`, `room`, `house`                        |
| `--pages`       | `20`                          | страниц по ~24 объявления                      |
| `--out`         | `data/processed/listings.csv` | путь к CSV                                     |

### Шаг 2. Заливка фото в MinIO

```bash
uv run python -m realty_scraper.images \
    --in  data/processed/listings.csv \
    --out data/processed/listings_s3.csv \
    --bucket realty \
    --workers 8 \
    --max-per-offer 10
```

Что делает:
1. Подключается к MinIO по `S3_ENDPOINT_URL` (по умолчанию `http://localhost:9000`);
2. Создаёт бакет `realty`, если его ещё нет;
3. Для каждого объявления скачивает до `--max-per-offer` фото и заливает
   в `s3://realty/images/<offer_id>/000.jpg`, `…/001.jpg`, …;
4. Записывает новый CSV, где колонка `image_urls` заменена на `image_uris`
   (`s3://realty/images/…` через `;`);
5. С флагом `--skip-existing` не перезакачивает уже залитые ключи.

Конфиденциальные параметры читаются из окружения (см. `.env.example`):
`S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_REGION`, `S3_BUCKET`.

### Шаг 3. Валидация и сохранение parquet

```bash
uv run python -m realty_scraper.clean \
    --in  data/processed/listings_s3.csv \
    --out data/processed/listings.parquet
```

Скрипт:
1. Читает CSV, приводит колонки к нужным типам (`float64`, nullable `Int64`,
   nullable `boolean`, `string`);
2. Прогоняет DataFrame через `LISTING_SCHEMA` (см. [schema.py](src/realty_scraper/schema.py));
3. Отсеивает строки, нарушившие хотя бы одну проверку; по колонкам
   выводит сводку, а все кейсы сохраняет в `listings_failures.csv`;
4. Сохраняет чистый датасет в `.parquet` (`pyarrow`, сжатие `snappy`).

## Pandera-схема

Жёсткие ограничения для строк (см. `LISTING_SCHEMA`):

| Поле             | Ограничение                                              |
|------------------|----------------------------------------------------------|
| `offer_id`       | только цифры, уникальный                                 |
| `url`            | начинается с `http`                                      |
| `deal_type`      | `sale` либо `rent`                                       |
| `price`          | 100 000 ≤ price ≤ 3 000 000 000 ₽                        |
| `area_total`     | 8 ≤ area ≤ 1 000 м²                                      |
| `rooms`          | 0 (студия) … 20                                          |
| `floor`          | 1 … 100                                                  |
| `floors_total`   | 1 … 100                                                  |
| `floor`/`floors` | `floor ≤ floors_total`                                   |
| `latitude`       | 41.0 … 82.0 (граница РФ)                                 |
| `longitude`      | 19.0 … 180.0                                             |
| `image_uris`     | пусто или только `s3://…` через `;`                      |

Запустить полный пайплайн одной командой:

```bash
make pipeline     # если есть Makefile; иначе — три uv run подряд
```

## DVC

### Настройка S3-remote в MinIO

```bash
# Удалим/переопределим старый локальный remote, если есть
uv run dvc remote remove localremote 2>/dev/null || true

uv run dvc remote add -d s3remote s3://dvc-storage
uv run dvc remote modify s3remote endpointurl http://localhost:9000
# Креды — только локально, в Git не коммитятся (.dvc/config.local)
uv run dvc remote modify --local s3remote access_key_id     minioadmin
uv run dvc remote modify --local s3remote secret_access_key minioadmin
```

### Снэпшот данных

```bash
uv run dvc add data/processed/listings.csv \
              data/processed/listings_s3.csv \
              data/processed/listings.parquet \
              data/processed/listings.jsonl

uv run dvc push              # заливка в s3://dvc-storage/...

git add data/processed/*.dvc .gitignore
git commit -m "data: clean parquet snapshot"
```

Чтобы подтянуть данные на другой машине:

```bash
docker compose up -d          # поднять MinIO
uv run dvc pull               # скачает все артефакты из dvc-storage
```

