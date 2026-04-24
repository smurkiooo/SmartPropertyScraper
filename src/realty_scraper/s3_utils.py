"""Тонкая обёртка над boto3 для работы с MinIO / S3-совместимым хранилищем."""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

DEFAULT_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
DEFAULT_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
DEFAULT_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
DEFAULT_REGION = os.environ.get("S3_REGION", "us-east-1")
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "realty")


def get_s3_client(
    *,
    endpoint_url: str = DEFAULT_ENDPOINT,
    access_key: str = DEFAULT_ACCESS_KEY,
    secret_key: str = DEFAULT_SECRET_KEY,
    region: str = DEFAULT_REGION,
) -> Any:
    """Создаёт boto3-клиент, настроенный на MinIO/S3."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def ensure_bucket(s3_client: Any, bucket: str = DEFAULT_BUCKET) -> None:
    """Создаёт бакет, если его ещё нет."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code not in ("404", "NoSuchBucket", "NoSuchKey"):
            raise
    s3_client.create_bucket(Bucket=bucket)


def object_exists(s3_client: Any, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise
