import base64
import io
import os
import sys

import requests
from app.config import settings
from app.qdrant import QdrantConnectionError, QdrantConnector
from celery import Celery
from loguru import logger
from pdf2image import convert_from_path
from PIL import Image
from qdrant_client import models

celery_app = Celery("tasks", broker=settings.REDIS_URL)

try:
    qdrant_service = QdrantConnector(url=settings.QDRANT_URL)
except QdrantConnectionError as e:
    logger.critical(
        f"Worker cannot start without Qdrant connection. Shutting down. Error: {e}"
    )
    sys.exit(1)


def embed(base64_image: list[str]) -> list[list[list[float]]]:
    payload = {"images_base64": base64_image}

    response = requests.post(
        f"{settings.EMBEDDING_SERVICE_URL}/embed_images", json=payload
    )
    response.raise_for_status()
    response_data = response.json()
    return response_data["embeddings"]


@celery_app.task(name="app.worker.process_pdf_task")
def process_pdf_task(self, file_path: str, collection_name: str):
    logger.info(f"Processing {file_path} for collection '{collection_name}'...")

    try:
        qdrant_service.ensure_collection_exists(collection_name)

        try:
            images: list[Image.Image] = convert_from_path(file_path, dpi=150)
        except Exception as e:
            logger.error(f"Could not convert PDF {file_path} to images: {e}")
            return

        base64_images = []
        for pil_image in images:
            buffered = io.BytesIO()
            pil_image.convert("RGB").save(buffered, format="JPEG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        if not base64_images:
            logger.warning(f"No images found in PDF {file_path}.")
            return

        try:
            embeddings = embed(base64_images)
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Embedding service failed for {file_path}. Task will fail. Error: {e}"
            )
            raise

        points_to_upload = []
        for i, (base64_image, embedding) in enumerate(zip(base64_images, embeddings)):

            point_id = f"{os.path.basename(file_path)}_page_{i}"
            payload = {
                "base64_image": base64_image,
                "file_name": os.path.basename(file_path),
                "page_number": i,
            }

            points_to_upload.append(
                models.PointStruct(id=point_id, vector=embedding, payload=payload)
            )

        qdrant_service.upsert_points(points_to_upload, collection_name)

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during processing of {file_path}: {e}"
        )
        raise self.retry(exc=e, countdown=60, max_retries=2)
