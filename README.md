# realty-scraper

Сбор объявлений о недвижимости с **cian.ru**, загрузка фото в MinIO/S3, валидация, обучение модели предсказания цены (**CatBoost + Optuna + MLflow**) и семантический **RAG-поиск** квартир по текстовому описанию с генерацией ответов через локальную LLM.

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
┌────────────┐     ┌─────────────┐     ┌────────────────┐     ┌─────────────┐
│ cian.ru    │ ──▶ │ listings    │ ──▶ │ listings_s3    │ ──▶ │ listings    │
│ API        │     │ .csv (http  │     │ .csv           │     │ .parquet    │
│            │     │  urls)      │     │ (s3://… URIs)  │     │ (clean)     │
└────────────┘     └─────────────┘     └────────────────┘     └──────┬──────┘
                        │                    │                        │
                        │         ┌──────────▼────────┐              │
                        │         │ MinIO S3 bucket   │              ▼
                        │         │ realty/images/... │    ┌─────────────────────┐
                        │         └───────────────────┘    │ Feature Engineering │
                        ▼                                   │ + CatBoostRegressor │
                      DVC                                   │ + Optuna TPE        │
                 (raw snapshot)                             └──────────┬──────────┘
                                                                       │
                                              ┌────────────────────────┼────────────────────────┐
                                              ▼                        ▼                        ▼
                                   preprocessed_listings.csv   models/catboost_model     MLflow run
                                        (под DVC)               (локально)          (метрики + артефакты)
```

## Архитектура

```
cian.ru API
    │
    ▼
listings.csv ──▶ MinIO/S3 (фото) ──▶ listings_s3.csv
    │                                       │
    ▼                                       ▼
listings.parquet                    [RAG индексация]
    │                               text_utils.py → clean_description()
    ▼                               index.py → sentence-transformers → ChromaDB
CatBoost + Optuna                           │
    │                                       ▼
MLflow                              search.py → семантический поиск
                                            │
                                            ▼
                                    llm.py → LM Studio (локальная LLM)
                                            │
                                            ▼
                                    web.py → веб-интерфейс (FastAPI)
```

## Структура проекта

```
.
├── docker-compose.yml          # MinIO + MLflow + ChromaDB
├── src/realty_scraper/
│   ├── cian.py                 # скрапер Cian API
│   ├── images.py               # скачивание фото → MinIO
│   ├── clean.py                # валидация + parquet
│   ├── schema.py               # Pandera-схема
│   ├── train.py                # CatBoost + Optuna + MLflow
│   ├── text_utils.py           # очистка текста для эмбеддингов
│   ├── index.py                # индексация описаний → ChromaDB
│   ├── search.py               # семантический поиск
│   ├── llm.py                  # клиент LM Studio
│   ├── rag.py                  # RAG pipeline (CLI)
│   ├── web.py                  # веб-интерфейс FastAPI
│   └── backup.py               # backup ChromaDB → JSON
├── data/
│   ├── processed/
│   │   ├── listings_s3.csv     # данные с S3 URI фотографий
│   │   └── listings.parquet    # финальный датасет
│   ├── chroma/                 # ChromaDB данные (персистентный том)
│   └── backups/                # JSON-бэкапы индекса
└── models/
    └── catboost_model
```

## Настройка окружения

Требования: [`uv`](https://docs.astral.sh/uv/), Git, Docker Desktop.

```bash
git clone <repo-url> realty-scraper && cd realty-scraper
uv sync
cp .env.example .env
docker compose up -d
```

| Сервис     | Адрес                       |
|------------|-----------------------------|
| MinIO UI   | http://localhost:9001       |
| MLflow UI  | http://localhost:5001       |
| ChromaDB   | http://localhost:8000 (API) |


## Использование

### Шаг 1. Сбор данных с cian.ru

```bash
uv run python -m realty_scraper.cian \
    --region 1 --deal-type sale --offer-type flat \
    --pages 20 \
    --out data/processed/listings.csv
```

Результат — `listings.csv` (структурированные поля + `image_urls`) и `listings.jsonl` с сырыми ответами API.

| Флаг            | По умолчанию                  | Описание                          |
|-----------------|-------------------------------|-----------------------------------|
| `--region`      | `1` (Москва)                  | ID региона Cian (2 = СПб)         |
| `--deal-type`   | `sale`                        | `sale` / `rent`                   |
| `--offer-type`  | `flat`                        | `flat`, `room`, `house`           |
| `--pages`       | `20`                          | страниц по ~24 объявления         |
| `--out`         | `data/processed/listings.csv` | путь к выходному CSV              |

### Шаг 2. Заливка фото в MinIO

```bash
uv run python -m realty_scraper.images \
    --in  data/processed/listings.csv \
    --out data/processed/listings_s3.csv \
    --bucket realty \
    --workers 8 \
    --max-per-offer 10
```

Скрипт скачивает до `--max-per-offer` фото на объявление, заливает в
`s3://realty/images/<offer_id>/`, заменяет `image_urls` на `image_uris` (S3 URI через `;`).
С флагом `--skip-existing` уже залитые ключи не перезаписываются.

### Шаг 3. Валидация и сохранение parquet

```bash
uv run python -m realty_scraper.clean \
    --in  data/processed/listings_s3.csv \
    --out data/processed/listings.parquet
```

Скрипт:
1. Приводит колонки к нужным типам (`float64`, nullable `Int64`, `string`);
2. Прогоняет DataFrame через `LISTING_SCHEMA` (см. [schema.py](src/realty_scraper/schema.py));
3. Отсеивает строки с нарушениями; все кейсы сохраняет в `listings_failures.csv`;
4. Сохраняет чистый датасет в `.parquet` (`pyarrow`, сжатие `snappy`).

### Шаг 4. Обучение модели

```bash
python -m realty_scraper.train                # 50 Optuna trials (по умолчанию)
python -m realty_scraper.train --n-trials 100  # больше trials → лучше гиперпараметры
```

Пайплайн выполняет:

1. **Загрузку** `listings.parquet` (`price` → `target`);
2. **Feature Engineering** — создание новых признаков (см. ниже);
3. **Подбор гиперпараметров** — Optuna TPE (`n_trials` итераций, метрика RMSE);
4. **Обучение финальной модели** с лучшими параметрами (`CatBoostRegressor`);
5. **Оценку** метрик на train и val выборках;
6. **Логирование** в MLflow — параметры, метрики, важность признаков, история Optuna, сигнатура модели;
7. **Сохранение** модели в `models/catboost_model`.

| Флаг          | По умолчанию | Описание                             |
|---------------|--------------|--------------------------------------|
| `--n-trials`  | `50`         | Количество Optuna trials             |
| `--data`      | `data/processed/listings.parquet` | Путь к входному файлу |
| `--model-out` | `models/catboost_model` | Путь для сохранения модели  |

#### Feature Engineering

| Признак           | Тип             | Описание                                                         |
|-------------------|-----------------|------------------------------------------------------------------|
| `district`        | категориальный  | Административный округ Москвы (ЦАО, ЗАО, СВАО…), из `address`  |
| `dist_to_center`  | числовой        | Евклидово расстояние (°) от координат объекта до Кремля         |
| `num_images`      | числовой        | Количество фото в объявлении                                     |
| `floor_ratio`     | числовой        | `floor / floors_total` — относительное положение этажа          |
| `is_top_floor`    | бинарный        | Последний этаж                                                   |
| `is_ground_floor` | бинарный        | Первый этаж                                                      |
| `rooms_is_null`   | бинарный        | Флаг пропуска поля `rooms`                                       |


#### Гиперпараметры, подбираемые Optuna

| Параметр              | Диапазон        | Шкала    |
|-----------------------|-----------------|----------|
| `learning_rate`       | 0.01 — 0.30     | log      |
| `depth`               | 4 — 10          | integer  |
| `l2_leaf_reg`         | 1.0 — 10.0      | log      |
| `bagging_temperature` | 0.0 — 2.0       | linear   |
| `min_data_in_leaf`    | 1 — 50          | integer  |

#### Результаты baseline (50 trials, 1 374 объявления)

| Выборка | R²     | MAE       | RMSE      | MAPE   |
|---------|--------|-----------|-----------|--------|
| Train   | 0.9200 | 17 631 ₽  | 42 001 ₽  | 18.4%  |
| Val     | 0.7100 | 25 244 ₽  | 54 146 ₽  | 46.7%  |

Топ-признаки по важности: `area_total` (32%), `dist_to_center` (12%),
`district` (12%), `underground` (9%).

---

## RAG-поиск квартир

Семантический поиск по текстовым описаниям объявлений с генерацией ответа через локальную LLM.

**Компоненты:**
- **Эмбеддинги:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Векторная БД:** ChromaDB (Docker, порт 8000), данные в `./data/chroma/`
- **LLM:** LM Studio с любой локальной моделью (OpenAI-совместимый API на `localhost:1234`)

**Запуск:**

```bash
# 1. Индексация (однократно, ~40 мин на CPU)
uv run python -m realty_scraper.index

# 2. Веб-интерфейс → http://localhost:7860
uv run python -m realty_scraper.web

# 3. CLI без LLM
uv run python -m realty_scraper.search "двушка рядом с метро" --top 5

# 4. CLI с LLM (LM Studio должен быть запущен)
uv run python -m realty_scraper.rag "пентхаус с видом" --top 3

# 5. Backup индекса
uv run python -m realty_scraper.backup
```

**Важные детали:**
- `text_utils.py` выполняет лёгкую предобработку: удаление HTML, эмодзи, нормализацию unicode. Стоп-слова и приведение к нижнему регистру **не применяются** — это ухудшает качество трансформерных эмбеддингов.
- Метаданные в ChromaDB не принимают `None` — пропущенные числовые поля заменяются на `0`.
- `index.py` идемпотентен: повторный запуск пропускает уже проиндексированные `offer_id`.
- В LLM передаётся не более 3 объявлений (ограничение контекста 3B-модели).

---

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

---

## MLflow

MLflow-сервер поднимается вместе с MinIO через `docker compose up -d`.

- **UI:** `http://localhost:5001`
- **Backend store:** SQLite (`mlflow/mlflow.db`, монтируется в контейнер)
- **Artifact store:** `s3://realty/mlflow/` (MinIO)

Каждый запуск `train.py` создаёт MLflow run с:
- гиперпараметрами (в т.ч. лучшими параметрами Optuna);
- метриками (`train_r2`, `val_r2`, `train_mae`, `val_mae`, `train_mape`, `val_mape`, `optuna_best_val_rmse`);
- артефактами: `feature_importance.csv`, `optuna_trials.csv`;
- зарегистрированной моделью `realty-price-catboost` с сигнатурой и примером входных данных.

