"""
Пайплайн предобработки и обучения и логироваия CatBoostRegressor на listings.parquet.
Целевая переменная: target (цена квартиры)

Запуск:
    python -m realty_scraper.train
    python -m realty_scraper.train --n-trials 100 (для кастомного числа )

"""
from __future__ import annotations

import argparse
import os
import pathlib
import tempfile

import mlflow
import mlflow.catboost
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from mlflow.models import infer_signature
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = pathlib.Path("data/processed/listings.parquet")
DATA_PATH_CLIP = pathlib.Path("data/processed/listings_clip.parquet")
MODEL_PATH = pathlib.Path("models/catboost_model")
TARGET_COL = "target"
EXPERIMENT_NAME = "realty-price-prediction"
CLIP_DIM = 512
CLIP_PCA_COMPONENTS = 64  # 512 → 64 (объясняет ~90% дисперсии)

CAT_FEATURES = ["object_type", "underground", "district"]

DROP_COLS = [
    "Unnamed: 0",
    "offer_id",
    "url",
    "deal_type",
    "city",
    "description",
    "address",
    "image_uris",
    "rayon",    # 102 уникальных значения при 1332 строках — слишком разреженно для target encoding
]

# координаты Кремля — точка отсчёта расстояния до центра Москвы, использую это для feature engineering-а
_KREMLIN_LAT = 55.752
_KREMLIN_LON = 37.617


def load_data(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(path).rename(columns={"price": TARGET_COL})


def clip_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("clip_")]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["district"] = (
            df["address"].str.split(",").str[0]
            .str.split("(").str[0]
            .str.strip()
        )
        # район: "р-н Хамовники" → "Хамовники" (преобразовываем до удаления колонки с адресом)
        df["rayon"] = df["address"].str.extract(r"р-н\s+([^,]+)")[0].str.strip()
        df["rayon"] = df["rayon"].fillna("Нет района")


        df["num_images"] = (
            df["image_uris"].fillna("").apply(lambda x: len(x.split(";")) if x else 0)
        )

        # расстояние до центра Москвы (градусные единицы) (евклидово расстояние)
        df["dist_to_center"] = np.sqrt(
            (df["latitude"] - _KREMLIN_LAT) ** 2
            + (df["longitude"] - _KREMLIN_LON) ** 2
        )

      
        df["floor_ratio"] = df["floor"] / df["floors_total"]
        df["is_top_floor"] = (df["floor"] == df["floors_total"]).astype("int8")
        df["is_ground_floor"] = (df["floor"] == 1).astype("int8")

        # flatShareSale встречается 1 раз — будем считать это шумом...
        df["object_type"] = df["object_type"].replace("flatShareSale", "flatSale")

        for col in ("rooms", "floor", "floors_total"):
            if col in df.columns:
                df[col] = df[col].astype("float64")

       
        df["rooms_is_null"] = df["rooms"].isna().astype("int8")

        df["underground"] = df["underground"].fillna("Нет метро")  # отсутствие метро будем считать информативным признаком

        return df.drop(columns=[c for c in DROP_COLS if c in df.columns])



#оптимизируем гиперпараметры с помощью алгоритма TPE в  Optuna


def _objective(trial: optuna.Trial, train_pool: Pool, val_pool: Pool) -> float:
    params = {
        "iterations": 1_000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "cat_features": CAT_FEATURES,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": False,
    }
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool)

    preds = model.predict(val_pool)
    y_val = np.array(val_pool.get_label())
    return float(np.sqrt(np.mean((y_val - preds) ** 2)))


def run_optuna(train_pool: Pool, val_pool: Pool, n_trials: int) -> tuple[dict, optuna.Study]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: _objective(trial, train_pool, val_pool),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_trial
    print(f"Optuna: лучший RMSE на валидац. выборке = {study.best_value}")
    print("лучшие гиперпараметры:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    return study.best_params, study



def build_model(params: dict) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1_000,
        loss_function="RMSE",
        eval_metric="RMSE",
        cat_features=CAT_FEATURES,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        **params,
    )


# Метрики


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def print_metrics(metrics: dict[str, float], label: str):
    print(f"\n{'='*40}")
    print(f"  {label} метрика")
    print(f"{'='*40}")
    print(f"  MAE:  {metrics['mae']:>15,.0f} RUB")
    print(f"  RMSE: {metrics['rmse']:>15,.0f} RUB")
    print(f"  R^2:   {metrics['r2']:>15.4f}")
    print(f"  MAPE: {metrics['mape']:>14.2f} %")



# главный пайплайн обучения, подбора гиперпараметров и логирования модели в mlflow


def train(data_path: pathlib.Path = DATA_PATH, model_path: pathlib.Path = MODEL_PATH, log_target: bool = True, test_size: float = 0.2, n_trials: int = 50,) -> CatBoostRegressor:

    df = load_data(data_path)
    clip_cols = clip_columns(df)
    use_clip = len(clip_cols) > 0
    print(f"Датасет: {df.shape[0]} строк × {df.shape[1]} столбцов"
          + (f"  (включая {len(clip_cols)} CLIP-признаков)" if use_clip else ""))

    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])

    if log_target:#логарифмирование целевой переменной ( = ln(1+x))
        y = np.log1p(y)

    X = FeatureEngineer().fit_transform(X)

    if use_clip:
        clip_present = [c for c in X.columns if c.startswith("clip_")]
        pca = PCA(n_components=CLIP_PCA_COMPONENTS, random_state=42)
        clip_reduced = pca.fit_transform(X[clip_present])
        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA: {len(clip_present)} → {CLIP_PCA_COMPONENTS} CLIP-признаков "
              f"(объяснённая дисперсия: {explained:.2%})")
        X = X.drop(columns=clip_present)
        pca_df = pd.DataFrame(
            clip_reduced,
            columns=[f"clip_pca_{i}" for i in range(CLIP_PCA_COMPONENTS)],
            index=X.index,
        )
        X = pd.concat([X, pca_df], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"Трейн: {len(X_train)}  |  Валидация: {len(X_val)}  |  Признаков: {X.shape[1]}")

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)

 
    print(f"Optuna TPE: {n_trials} итераций")
    best_params, study = run_optuna(train_pool, val_pool, n_trials=n_trials)

   
    print("Обучение финальной модели с лучшими гиперпараметрами…")
    model = build_model(best_params)
    model.fit(train_pool, eval_set=val_pool)

    preds_train = model.predict(train_pool)
    preds_val = model.predict(val_pool)

    if log_target:  # обязательно применить обратную функцию к целевой переменной,
        # чтобы корректно интерпретировать предсказанные значения. конкретно: exp(x)-1
        preds_train = np.expm1(preds_train)
        preds_val = np.expm1(preds_val)
        y_train_orig = np.expm1(y_train.to_numpy())
        y_val_orig = np.expm1(y_val.to_numpy())
    else:
        y_train_orig = y_train.to_numpy()
        y_val_orig = y_val.to_numpy()

    train_metrics = compute_metrics(y_train_orig, preds_train)
    val_metrics = compute_metrics(y_val_orig, preds_val)

    print_metrics(train_metrics, label="Train")
    print_metrics(val_metrics, label="Val  ")

    feat_imp = pd.Series(model.get_feature_importance(), index=X_train.columns).sort_values(ascending=False)
    print("Топ-10 признаков по их важности:")
    print(feat_imp.head(10).to_string())

   
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params({
            "log_target": log_target,
            "test_size": test_size,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "n_trials": n_trials,
            "iterations": model.get_param("iterations"),
            "use_clip_features": use_clip,
            "clip_pca_components": CLIP_PCA_COMPONENTS if use_clip else 0,
            "clip_explained_variance": round(pca.explained_variance_ratio_.sum(), 4) if use_clip else 0,
            "total_features": len(X_train.columns),
            **best_params,
        })

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metric("optuna_best_val_rmse", study.best_value)

        with tempfile.TemporaryDirectory() as tmp:
            imp_path = pathlib.Path(tmp) / "feature_importance.csv"
            feat_imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))

            trials_path = pathlib.Path(tmp) / "optuna_trials.csv"
            study.trials_dataframe().to_csv(trials_path, index=False)
            mlflow.log_artifact(str(trials_path))

        signature = infer_signature(X_val, preds_val)
        input_example = X_val.head(5)

        model_name = "realty-price-catboost-clip_PCA" if use_clip else "realty-price-catboost"

        mlflow.catboost.log_model(
            cb_model=model,
            name="catboost_model",           
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )

        run_id = mlflow.active_run().info.run_id
        artifact_uri = mlflow.get_artifact_uri("catboost_model")
        print(f"MLflow run:      {run_id}")
        print(f"Tracking URI:    {tracking_uri}")
        print(f"Артефакт модели: {artifact_uri}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение CatBoostRegressor на listings.parquet")
    p.add_argument("--data", type=pathlib.Path, default=None)
    p.add_argument("--clip", action="store_true")
    p.add_argument("--no-clip", dest="clip", action="store_false")
    p.add_argument("--model-out", type=pathlib.Path, default=MODEL_PATH)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.data is not None:
        data_path = args.data
    elif args.clip:
        data_path = DATA_PATH_CLIP
    elif DATA_PATH_CLIP.exists():
        print(f"Найден {DATA_PATH_CLIP}, используем CLIP-датасет (--no-clip чтобы отключить)")
        data_path = DATA_PATH_CLIP
    else:
        data_path = DATA_PATH

    train(data_path=data_path, model_path=args.model_out, n_trials=args.n_trials)
