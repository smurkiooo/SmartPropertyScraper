# realty-scraper

Сбор структурированных данных, текстовых описаний и фотографий объявлений о недвижимости с российского портала **cian.ru**, загрузка фото в локальный MinIO/S3, валидация Pandera-схемой, сохранение в parquet под контролем DVC и обучение модели предсказания цены на основе **CatBoostRegressor** с подбором гиперпараметров через **Optuna TPE** и логированием в **MLflow**. Фотографии квартир кодируются **CLIP ViT-B/32** и обогащают датасет визуальными признаками. Дополнительно — семантический **RAG-поиск** квартир по текстовому описанию, генерация ответов через локальную LLM и **Telegram-бот** для предсказания цены.

p.s. Все основные файлы находятся по пути **./src/realty_scraper**

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
┌────────────┐   ┌──────────────┐   ┌────────────────┐   ┌─────────────┐
│ cian.ru    │──▶│ listings.csv │──▶│ listings_s3    │──▶│ listings    │
│ API        │   │ (http urls)  │   │ .csv           │   │ .parquet    │
└────────────┘   └──────────────┘   │ (s3://… URIs)  │   │ (clean)     │
                                    └───────┬────────┘   └──────┬──────┘
                                            │                   │
                               ┌────────────▼────────┐          │
                               │ MinIO S3             │         ▼
                               │ realty/images/...    │  ┌─────────────────┐
                               └────────────┬─────────┘  │ Feature Eng.    │
                                            │            └────────┬────────┘
                                            ▼                     │
                                   ┌─────────────────┐            │
                                   │ clip_embeddings │            │
                                   │ CLIP ViT-B/32   │            │
                                   │ + PCA 512→64    │            │
                                   └────────┬────────┘            │
                                            │                     │
                                            ▼                     ▼
                                   listings_clip.parquet  ──▶  train.py
                                                                   │
                              ┌─────────────────────────┬──────────┘
                              ▼                         ▼
                       MLflow run                models/catboost_model
                  (метрики + артефакты            (локально + S3)
                   на S3/MinIO)
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
├── pyproject.toml              # uv-проект, зависимости
├── uv.lock
├── docker-compose.yml          # MinIO + MLflow + ChromaDB + автосоздание бакетов
├── .env.example                # пример учёток MinIO, S3, MLflow
├── .gitignore
├── .dvc/                       # конфигурация DVC
├── src/realty_scraper/
│   ├── __init__.py
│   ├── cian.py                 # шаг 1: скрэйпер Cian API
│   ├── s3_utils.py             # boto3-клиент для MinIO
│   ├── images.py               # шаг 2: скачивание фото + заливка в S3
│   ├── schema.py               # Pandera-схема датасета
│   ├── clean.py                # шаг 3: типизация + валидация + parquet
│   ├── clip_embeddings.py      # шаг 4: CLIP-эмбеддинги фото → listings_clip.parquet
│   ├── train.py                # шаг 5: Feature Engineering + CatBoost + MLflow
│   ├── bot.py                  # Telegram-бот для предсказания цены
│   ├── text_utils.py           # очистка текста для эмбеддингов
│   ├── index.py                # индексация описаний → ChromaDB
│   ├── search.py               # семантический поиск
│   ├── llm.py                  # клиент LM Studio
│   ├── rag.py                  # RAG pipeline (CLI)
│   ├── web.py                  # веб-интерфейс FastAPI
│   └── backup.py               # backup ChromaDB → JSON
├── data/
│   ├── raw/                    # сырые JSONL-ответы API (под DVC)
│   ├── processed/
│   │   ├── listings.csv        # структурированные поля + http image_urls
│   │   ├── listings_s3.csv     # те же поля, но image_uris → s3://…
│   │   ├── listings.parquet    # типизированный + валидированный датасет
│   │   └── listings_clip.parquet  # датасет + 512 CLIP-столбцов (clip_0…clip_511)
│   ├── chroma/                 # ChromaDB данные (персистентный том)
│   └── backups/                # JSON-бэкапы индекса
├── models/
│   └── catboost_model          # обученная модель CatBoost (бинарный формат)
├── mlflow/
│   └── mlflow.db               # SQLite backend MLflow (монтируется в Docker)
└── notebooks/                  # ноутбуки для EDA
```

## Настройка окружения

Требования: [`uv`](https://docs.astral.sh/uv/), Git, Docker Desktop.

```bash
git clone <repo-url> realty-scraper && cd realty-scraper
uv sync

# dvc-s3 ставится отдельно (конфликтует с uv resolver-ом из-за aiobotocore)
uv pip install "dvc[s3]"

# Поднять MinIO + MLflow + ChromaDB
cp .env.example .env
docker compose up -d
```

| Сервис     | Адрес                       |
|------------|-----------------------------|
| MinIO UI   | http://localhost:9001       |
| S3 API     | http://localhost:9000       |
| MLflow UI  | http://localhost:5001       |
| ChromaDB   | http://localhost:8000 (API) |

Контейнер `mc` автоматически создаст два бакета:
- `realty` — фото объявлений и артефакты MLflow;
- `dvc-storage` — удалённое хранилище DVC.


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

### Шаг 4. CLIP-эмбеддинги фотографий

```bash
uv run python -m realty_scraper.clip_embeddings
```

Скрипт обрабатывает фотографии **порциями по 50 объявлений** (контроль RAM), не накапливая все изображения в памяти одновременно:

1. Скачивает фото из MinIO параллельно (`--workers 4`);
2. Кодирует батчами через **CLIP ViT-B/32** (HuggingFace Transformers);
3. Для каждого объявления **усредняет** эмбеддинги всех фото и повторно L2-нормализует;
4. Добавляет 512 столбцов `clip_0`…`clip_511` и сохраняет `listings_clip.parquet`.

**Почему CLIP ViT-B/32:** модель обучена на парах image-text и понимает семантику интерьеров — "уютная гостиная", "свежий ремонт" — как единое целое, а не как пиксели. Это даёт более полезные признаки для задачи, чем обычные CNN-фичи.

| Флаг            | По умолчанию | Описание                                      |
|-----------------|--------------|-----------------------------------------------|
| `--workers`     | `4`          | Потоков для скачивания из S3                  |
| `--batch-size`  | `16`         | Изображений за один CLIP-батч                 |
| `--offer-chunk` | `50`         | Объявлений в порции (контролирует пик RAM)    |
| `--data`        | `listings.parquet` | Исходный датасет                        |
| `--out`         | `listings_clip.parquet` | Выходной обогащённый датасет       |

### Шаг 5. Обучение модели

```bash
# С CLIP-признаками (авто-выбор если listings_clip.parquet существует)
AWS_ACCESS_KEY_ID=minioadmin \
AWS_SECRET_ACCESS_KEY=minioadmin \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
MLFLOW_TRACKING_URI=http://localhost:5001 \
uv run python -m realty_scraper.train --n-trials 50

# Без CLIP (только базовые признаки)
uv run python -m realty_scraper.train --no-clip --n-trials 50
```

Пайплайн выполняет:

1. **Загрузку** `listings_clip.parquet` или `listings.parquet` (`price` → `target`);
2. **Feature Engineering** — создание новых признаков (см. ниже);
3. **PCA-сжатие** CLIP-признаков: 512 → 64 компоненты (объясняет ~90% дисперсии) — снижает переобучение при соотношении ~1100 samples / 523 признака;
4. **Подбор гиперпараметров** — Optuna TPE (`n_trials` итераций, метрика RMSE);
5. **Обучение финальной модели** с лучшими параметрами (`CatBoostRegressor`);
6. **Оценку** метрик на train и val выборках;
7. **Логирование** в MLflow — параметры (в т.ч. `use_clip_features`, `clip_pca_components`, `clip_explained_variance`), метрики, feature importance, история Optuna, артефакт модели;
8. **Сохранение** модели в `models/catboost_model`.

| Флаг           | По умолчанию               | Описание                                        |
|----------------|----------------------------|-------------------------------------------------|
| `--n-trials`   | `50`                       | Количество Optuna trials                        |
| `--clip`       | авто                       | Принудительно использовать CLIP-датасет         |
| `--no-clip`    | —                          | Принудительно использовать базовый датасет      |
| `--test-size`  | `0.2`                      | Доля валидации (0.1 = split 90/10)              |
| `--data`       | авто                       | Путь к parquet (переопределяет `--clip`)        |
| `--model-out`  | `models/catboost_model`    | Путь для сохранения модели                      |

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
| `clip_pca_0…63`   | числовой (×64)  | PCA-компоненты CLIP-эмбеддингов фотографий интерьера            |

#### Гиперпараметры, подбираемые Optuna

| Параметр              | Диапазон        | Шкала    |
|-----------------------|-----------------|----------|
| `learning_rate`       | 0.01 — 0.30     | log      |
| `depth`               | 4 — 10          | integer  |
| `l2_leaf_reg`         | 1.0 — 10.0      | log      |
| `bagging_temperature` | 0.0 — 2.0       | linear   |
| `min_data_in_leaf`    | 1 — 50          | integer  |

#### Результаты (50 trials, 1 374 объявления, CLIP + PCA)

| Выборка | R²     | MAE        | RMSE       | MAPE   |
|---------|--------|------------|------------|--------|
| Train   | 0.9893 | 6 182 919 ₽ | 15 366 414 ₽ | 5.39%  |
| Val     | 0.7307 | 24 954 768 ₽ | 52 180 608 ₽ | 53.19% |

Топ-признаки по важности: `area_total` (34%), `district` (9%), `clip_pca_1` (4%), `dist_to_center` (4%). CLIP-компоненты входят в топ-10, что подтверждает влияние интерьера на цену.

---

## Telegram-бот

Предсказание цены квартиры через пошаговый диалог в Telegram.

```bash
TELEGRAM_BOT_TOKEN=<token> uv run python -m realty_scraper.bot

# С MLflow-сервером (модель загружается из последнего run)
TELEGRAM_BOT_TOKEN=<token> \
MLFLOW_TRACKING_URI=http://localhost:5001 \
uv run python -m realty_scraper.bot
```

Бот собирает данные шаг за шагом: адрес → площадь → комнаты → этаж → тип объекта → метро → координаты. Затем применяет `FeatureEngineer` и возвращает предсказанную цену.

**Логика загрузки модели:** сначала ищет последний MLflow run в эксперименте `realty-price-prediction`, при неудаче — загружает локальный файл `models/catboost_model`.

| Команда    | Описание                    |
|------------|-----------------------------|
| `/start`   | Приветствие                 |
| `/predict` | Запустить ввод данных       |
| `/cancel`  | Отменить текущий ввод       |
| `/help`    | Справка                     |

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
- гиперпараметрами (в т.ч. лучшими параметрами Optuna и CLIP-флагами: `use_clip_features`, `clip_pca_components`, `clip_explained_variance`, `total_features`);
- метриками (`train_r2`, `val_r2`, `train_mae`, `val_mae`, `train_mape`, `val_mape`, `optuna_best_val_rmse`);
- артефактами: `feature_importance.csv`, `optuna_trials.csv`, модель CatBoost;
- зарегистрированной моделью: `realty-price-catboost` (без CLIP) или `realty-price-catboost-clip` (с CLIP).

> **Важно:** при запуске с HTTP tracking URI необходимо передать credentials MinIO:
> ```bash
> AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
> MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
> MLFLOW_TRACKING_URI=http://localhost:5001 \
> uv run python -m realty_scraper.train
> ```
> Без них MLflow логирует метрики локально в `mlruns/`, но не может загрузить артефакты в S3.

