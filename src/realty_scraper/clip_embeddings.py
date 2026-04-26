"""
Генерирует CLIP-эмбеддинги изображений квартир из S3 и добавляет их в датасет.

Алгоритм (потоковый — без накопления всех фото в RAM):
  1. Читает listings.parquet (столбец image_uris содержит ';'-разделённые S3-URI).
  2. Для каждой порции объявлений (offer_chunk):
       a. Скачивает фото параллельно (ThreadPoolExecutor).
       b. Прогоняет через CLIP ViT-B/32 батчами.
       c. Усредняет эмбеддинги по объявлению, L2-нормализует.
       d. Сразу освобождает память (PIL-объекты не накапливаются).
  3. Добавляет 512 столбцов clip_0 … clip_511 и сохраняет listings_clip.parquet.

Запуск:
    uv run python -m realty_scraper.clip_embeddings
    uv run python -m realty_scraper.clip_embeddings --offer-chunk 64 --workers 4 --batch-size 16
"""
from __future__ import annotations

import argparse
import io
import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .s3_utils import DEFAULT_BUCKET, DEFAULT_ENDPOINT, get_s3_client

logger = logging.getLogger(__name__)

DATA_PATH = pathlib.Path("data/processed/listings.parquet")
OUT_PATH = pathlib.Path("data/processed/listings_clip.parquet")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DIM = 512


def _parse_uris(cell: str | None) -> list[str]:
    if not cell or pd.isna(cell):
        return []
    return [u.strip() for u in str(cell).split(";") if u.strip()]


def _s3_key(uri: str) -> str:
    """s3://bucket/a/b/c → a/b/c"""
    parts = uri.split("/", 3)
    return parts[3] if len(parts) > 3 else uri


def _download_image(s3_client, bucket: str, uri: str) -> Image.Image | None:
    try:
        body = s3_client.get_object(Bucket=bucket, Key=_s3_key(uri))["Body"].read()
        return Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as exc:
        logger.debug("skip %s: %s", uri, exc)
        return None



def load_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    logger.info("Загружаем CLIP (%s) на %s...", CLIP_MODEL_NAME, device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def _encode_batch(images: list[Image.Image], model: CLIPModel, processor: CLIPProcessor, device: str, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        vision_outputs = model.vision_model(**inputs)
        image_feats = model.visual_projection(vision_outputs[1])  # pooler_output
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        parts.append(image_feats.cpu().float().numpy())
    return np.vstack(parts)


def _mean_embed(embeds: np.ndarray) -> np.ndarray:
    """усредняет строки и повторно L2-нормализует."""
    vec = embeds.mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec



# основной пайплайн. ключевая особенность: все изображения не накапливаются в памяти единовременно


def compute_clip_embeddings(df: pd.DataFrame, *, model: CLIPModel, processor: CLIPProcessor, device: str, s3_client, bucket: str, workers: int, batch_size: int, offer_chunk: int) -> np.ndarray:
    """
    обрабатывает объявления порциями по offer_chunk штук.
    В памяти одновременно хранится не более offer_chunk * max_photos изображений.
    Возвращает матрицу (len(df), CLIP_DIM); строки без фото — нулевой вектор.
    """
    n = len(df)
    matrix = np.zeros((n, CLIP_DIM), dtype=np.float32)
    covered = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pbar = tqdm(total=n, desc="offers", unit="offer")

        for chunk_start in range(0, n, offer_chunk):
            chunk_end = min(chunk_start + offer_chunk, n)
            chunk = df.iloc[chunk_start:chunk_end]
            future_map: dict = {}
            for local_idx, (_, row) in enumerate(chunk.iterrows()):
                for uri in _parse_uris(row.get("image_uris")):
                    fut = pool.submit(_download_image, s3_client, bucket, uri)
                    future_map[fut] = local_idx

            images_per_offer: dict[int, list[Image.Image]] = {
                i: [] for i in range(len(chunk))
            }
            for fut in as_completed(future_map):
                img = fut.result()
                if img is not None:
                    images_per_offer[future_map[fut]].append(img)

            for local_idx, imgs in images_per_offer.items():
                if not imgs:
                    continue
                embeds = _encode_batch(imgs, model, processor, device, batch_size)
                matrix[chunk_start + local_idx] = _mean_embed(embeds)
                covered += 1
                del imgs

            pbar.update(len(chunk))

        pbar.close()

    return matrix


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=DATA_PATH)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--offer-chunk", dest="offer_chunk", type=int, default=50)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("-v", "--verbose")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Устройство: %s  |  chunk=%d  workers=%d  batch=%d",
                device, args.offer_chunk, args.workers, args.batch_size)

    df = pd.read_parquet(args.data)
    logger.info("Датасет: %d строк × %d столбцов", *df.shape)

    model, processor = load_clip(device)
    s3 = get_s3_client(endpoint_url=args.endpoint)

    clip_matrix = compute_clip_embeddings(
        df,
        model=model,
        processor=processor,
        device=device,
        s3_client=s3,
        bucket=args.bucket,
        workers=args.workers,
        batch_size=args.batch_size,
        offer_chunk=args.offer_chunk,
    )

    clip_df = pd.DataFrame(
        clip_matrix,
        columns=[f"clip_{i}" for i in range(CLIP_DIM)],
        index=df.index,
    )
    df_out = pd.concat([df, clip_df], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.out, index=False)
    logger.info("cохранено  %s  (%d строк, %d столбцов)", args.out, *df_out.shape)
    print(f" Готово: {args.out}")


if __name__ == "__main__":
    main()
