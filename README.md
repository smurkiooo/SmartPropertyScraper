# realty-scraper

Проект: сбор структурированных данных, текстовых описаний и фотографий
объявлений о недвижимости с российского портала **cian.ru**, загрузка фото
в локальный MinIO/S3, валидация Pandera-схемой, сохранение в parquet под
контролем DVC и обучение модели предсказания цены на основе **CatBoostRegressor**
с подбором гиперпараметров через **Optuna TPE** и логированием в **MLflow**.

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

## Структура проекта

```
.
├── pyproject.toml              # uv-проект, зависимости
├── uv.lock
├── docker-compose.yml          # MinIO + MLflow + автосоздание бакетов
├── .env.example                # пример учёток MinIO, S3, MLflow
├── .gitignore
├── .dvc/                       # конфигурация DVC
├── src/
│   └── realty_scraper/
│       ├── __init__.py
│       ├── cian.py             # шаг 1: скрэйпер Cian API
│       ├── s3_utils.py         # boto3-клиент для MinIO
│       ├── images.py           # шаг 2: скачивание фото + заливка в S3
│       ├── schema.py           # Pandera-схема датасета
│       ├── clean.py            # шаг 3: типизация + валидация + parquet
│       └── train.py            # шаг 4: Feature Engineering + CatBoost + MLflow
├── data/
│   ├── raw/                    # сырые JSONL-ответы API (под DVC)
│   └── processed/
│       ├── listings.csv        # структурированные поля + http image_urls
│       ├── listings_s3.csv     # те же поля, но image_uris → s3://…
│       ├── listings.parquet    # типизированный + валидированный датасет
│       └── preprocessed_listings.csv  # датасет после Feature Engineering (под DVC)
├── models/
│   └── catboost_model          # обученная модель CatBoost (бинарный формат)
├── mlflow/
│   └── mlflow.db               # SQLite backend MLflow (монтируется в Docker)
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

# 3. Поднять MinIO + MLflow
cp .env.example .env
docker compose up -d
# MinIO Web UI:  http://localhost:9001  (minioadmin / minioadmin)
# S3 API:        http://localhost:9000
# MLflow UI:     http://localhost:5001
```

Контейнер `mc` автоматически создаст два бакета:
- `realty` — фото объявлений и артефакты MLflow;
- `dvc-storage` — удалённое хранилище DVC.

### Переменные окружения

Скопируйте `.env.example` в `.env` и при необходимости скорректируйте:

```dotenv
# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# S3 (src/realty_scraper/s3_utils.py)
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# MLflow (src/realty_scraper/train.py)
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

> Если `MLFLOW_TRACKING_URI` не задан, запуски логируются локально в `./mlruns/` без необходимости поднимать Docker.

---

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

> `rayon` (район) не используется: 102 уникальных значения при ~1 400 строках
> создают разреженный target encoding и приводят к переобучению.

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

Для логирования на сервер (а не локально) задайте переменные окружения из `.env` перед запуском:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

python -m realty_scraper.train --n-trials 50
```

---

## DVC

### Настройка S3-remote в MinIO

```bash
uv run dvc remote add -d s3remote s3://dvc-storage
uv run dvc remote modify s3remote endpointurl http://localhost:9000
# Креды только локально, в Git не коммитятся (.dvc/config.local)
uv run dvc remote modify --local s3remote access_key_id     minioadmin
uv run dvc remote modify --local s3remote secret_access_key minioadmin
```

### Снэпшот данных

```bash
uv run dvc add data/processed/listings.csv \
              data/processed/listings_s3.csv \
              data/processed/listings.parquet \
              data/processed/preprocessed_listings.csv

uv run dvc push              # заливка в s3://dvc-storage/...

git add data/processed/*.dvc .gitignore
git commit -m "data: add preprocessed listings snapshot"
```

Чтобы подтянуть данные на другой машине:

```bash
docker compose up -d          # поднять MinIO
uv run dvc pull               # скачать все артефакты из dvc-storage
```
