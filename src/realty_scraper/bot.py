"""Telegram-бот для предсказания цены квартиры через CatBoostRegressor из MLflow.

Запуск:
    TELEGRAM_BOT_TOKEN=<token> python -m realty_scraper.bot
    TELEGRAM_BOT_TOKEN=<token> MLFLOW_TRACKING_URI=http://localhost:5001 python -m realty_scraper.bot

Бот ведёт пошаговый диалог, собирает данные о квартире и возвращает предсказанную цену.
"""
from __future__ import annotations

import logging
import os
import pathlib

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from realty_scraper.train import DROP_COLS, CAT_FEATURES, FeatureEngineer

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = pathlib.Path("models/catboost_model")

(
    ADDRESS,
    AREA,
    ROOMS,
    FLOOR,
    FLOORS_TOTAL,
    OBJECT_TYPE,
    UNDERGROUND,
    LATITUDE,
    LONGITUDE,
) = range(9)

OBJECT_TYPES = ["flatSale", "newBuildingFlatSale"]


def _load_model() -> CatBoostRegressor:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name("realty-price-prediction")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{run_id}/catboost_model"
                logger.info("Загружаем модель из MLflow run %s", run_id)
                return mlflow.catboost.load_model(model_uri)
    except Exception:
        logger.warning("Не удалось загрузить модель из MLflow, используем локальный файл")

    logger.info("Загружаем модель из %s", LOCAL_MODEL_PATH)
    m = CatBoostRegressor()
    m.load_model(str(LOCAL_MODEL_PATH))
    return m


_model: CatBoostRegressor | None = None
_fe = FeatureEngineer()


def get_model() -> CatBoostRegressor:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_price(data: dict) -> float:
    df = pd.DataFrame([data])
    df = _fe.transform(df)
    expected = get_model().feature_names_
    df = df[[c for c in expected if c in df.columns]]
    pred_log = get_model().predict(df)
    return float(np.expm1(pred_log[0]))




def _fmt_price(price: float) -> str:
    return f"{price:,.0f} ₽".replace(",", " ")


def _summary(d: dict, price: float) -> str:
    rooms_str = "студия" if pd.isna(d.get("rooms")) else str(int(d["rooms"]))
    metro = d.get("underground") or "нет метро"
    return (
        f"🏠 <b>Предсказанная цена: {_fmt_price(price)}</b>\n\n"
        f"<b>Параметры квартиры:</b>\n"
        f"  📍 Адрес: {d['address']}\n"
        f"  📐 Площадь: {d['area_total']} м²\n"
        f"  🛏 Комнат: {rooms_str}\n"
        f"  🏢 Этаж: {int(d['floor'])} из {int(d['floors_total'])}\n"
        f"  🏷 Тип: {d['object_type']}\n"
        f"  🚇 Метро: {metro}\n"
        f"  🌐 Координаты: {d['latitude']}, {d['longitude']}"
    )




async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я предсказываю цену квартиры в Москве с помощью модели CatBoost.\n\n"
        "Команды:\n"
        "  /predict — начать ввод данных\n"
        "  /cancel  — отменить текущий ввод\n"
        "  /help    — справка"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я использую модель CatBoostRegressor, обученную на объявлениях о продаже квартир.\n\n"
        "Чтобы получить прогноз, введи /predict и ответь на несколько вопросов:\n"
        "адрес, площадь, комнаты, этаж, тип объекта, метро, координаты.\n\n"
        "Цена предсказывается в рублях."
    )


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Введите адрес квартиры.")
    return ADDRESS


async def get_address(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["address"] = update.message.text.strip()
    await update.message.reply_text("Введите общую площадь квартиры (м²), например: <i>54.3</i>", parse_mode="HTML")
    return AREA


async def get_area(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["area_total"] = float(update.message.text.replace(",", "."))
    except ValueError:
        await update.message.reply_text("Неверный формат. Введите число, например: <i>54.3</i>", parse_mode="HTML")
        return AREA
    await update.message.reply_text(
        "Введите количество комнат.\n"
        "Напишите число или <i>0</i> / <i>студия</i> для студии.",
        parse_mode="HTML",
    )
    return ROOMS


async def get_rooms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip().lower()
    if text in ("0", "студия", "studio"):
        context.user_data["rooms"] = float("nan")
    else:
        try:
            context.user_data["rooms"] = float(text)
        except ValueError:
            await update.message.reply_text("Введите число комнат или <i>студия</i>.", parse_mode="HTML")
            return ROOMS
    await update.message.reply_text("Введите номер этажа квартиры, например: <i>7</i>", parse_mode="HTML")
    return FLOOR


async def get_floor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["floor"] = float(update.message.text.strip())
    except ValueError:
        await update.message.reply_text("Введите целое число.")
        return FLOOR
    await update.message.reply_text("Введите общее количество этажей в доме, например: <i>17</i>", parse_mode="HTML")
    return FLOORS_TOTAL


async def get_floors_total(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        val = float(update.message.text.strip())
    except ValueError:
        await update.message.reply_text("Введите целое число.")
        return FLOORS_TOTAL
    context.user_data["floors_total"] = val
    keyboard = [[t] for t in OBJECT_TYPES]
    await update.message.reply_text(
        "Выберите тип объекта:",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return OBJECT_TYPE


async def get_object_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    val = update.message.text.strip()
    if val not in OBJECT_TYPES:
        keyboard = [[t] for t in OBJECT_TYPES]
        await update.message.reply_text(
            "Выберите один из вариантов:",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
        )
        return OBJECT_TYPE
    context.user_data["object_type"] = val
    await update.message.reply_text(
        "Введите название ближайшей станции метро.\n"
        "Например: <i>Парк культуры</i>\n"
        "Если метро нет — напишите <i>нет</i>",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode="HTML",
    )
    return UNDERGROUND


async def get_underground(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    context.user_data["underground"] = None if text.lower() == "нет" else text
    await update.message.reply_text(
        "Введите широту (latitude) квартиры.\n"
        "Например: <i>55.7387</i>\n\n"
        "💡 Координаты можно найти в Google Maps — нажмите правой кнопкой на точку на карте.",
        parse_mode="HTML",
    )
    return LATITUDE


async def get_latitude(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["latitude"] = float(update.message.text.replace(",", "."))
    except ValueError:
        await update.message.reply_text("Неверный формат. Введите число, например: <i>55.7387</i>", parse_mode="HTML")
        return LATITUDE
    await update.message.reply_text(
        "Введите долготу (longitude) квартиры.\n"
        "Например: <i>37.5957</i>",
        parse_mode="HTML",
    )
    return LONGITUDE


async def get_longitude(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        context.user_data["longitude"] = float(update.message.text.replace(",", "."))
    except ValueError:
        await update.message.reply_text("Неверный формат. Введите число, например: <i>37.5957</i>", parse_mode="HTML")
        return LONGITUDE
    return await _run_prediction(update, context)


async def _run_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    data = context.user_data.copy()
    data.setdefault("image_uris", "")  

    await update.message.reply_text("⏳ Считаю прогноз...")

    try:
        price = predict_price(data)
        await update.message.reply_text(_summary(data, price), parse_mode="HTML")
    except Exception:
        logger.exception("Ошибка при предсказании")
        await update.message.reply_text(
            " Произошла ошибка при предсказании. Попробуйте /predict ещё раз."
        )

    await update.message.reply_text("Чтобы сделать новый прогноз — /predict")
    return ConversationHandler.END


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Ввод отменён.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END




def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Установите переменную окружения TELEGRAM_BOT_TOKEN")

    get_model()

    app = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("predict", cmd_predict)],
        states={
            ADDRESS:      [MessageHandler(filters.TEXT & ~filters.COMMAND, get_address)],
            AREA:         [MessageHandler(filters.TEXT & ~filters.COMMAND, get_area)],
            ROOMS:        [MessageHandler(filters.TEXT & ~filters.COMMAND, get_rooms)],
            FLOOR:        [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floor)],
            FLOORS_TOTAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_floors_total)],
            OBJECT_TYPE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_object_type)],
            UNDERGROUND:  [MessageHandler(filters.TEXT & ~filters.COMMAND, get_underground)],
            LATITUDE:     [MessageHandler(filters.TEXT & ~filters.COMMAND, get_latitude)],
            LONGITUDE:    [MessageHandler(filters.TEXT & ~filters.COMMAND, get_longitude)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(conv)

    logger.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
