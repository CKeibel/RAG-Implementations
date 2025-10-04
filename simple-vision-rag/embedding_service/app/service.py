import base64
import io

import torch
from app.config import settings
from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from fastapi import HTTPException
from loguru import logger
from PIL import Image


class EmbeddingService:
    def __init__(self) -> None:
        self.device = settings.device
        self.batch_size = settings.batch_size
        logger.info(f"Embedding model device: {self.device}")
        logger.info(f"Image batch size: {self.batch_size}")

        self.model_id = "ModernVBERT/colmodernvbert"
        self.processor = ColModernVBertProcessor.from_pretrained(self.model_id)
        self.model = ColModernVBert.from_pretrained(
            self.model_id, torch_dtype=torch.float32, trust_remote_code=True
        ).to(self.device)

    def base64_to_pil(self, base64_str: str) -> Image.Image:
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes))

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[list[float]]]:
        """
        Return: list[list[list[float]]] (batch_size x tokens x embbeding_dim)
        """
        all_embeddings = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]

            inputs = self.processor.process_images(batch_images)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            batch_embeddings = self.model(**inputs)
            all_embeddings.append(batch_embeddings.cpu())

        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings.tolist()

    @torch.no_grad()
    def embed_text(self, query: str) -> list[list[float]]:
        inputs = self.processor.process_texts([query])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        embeddings = self.model(**inputs)
        embeddings = embeddings[0]
        return embeddings.cpu().tolist()

    def image_request(self, images_base64: str | list[str]) -> list[list[list[float]]]:
        try:
            if isinstance(images_base64, str):
                images_base64 = [images_base64]
            images = [self.base64_to_pil(img_str) for img_str in images_base64]
            embeddings = self.embed_images(images)
            return embeddings
        except Exception as e:
            self._handle_exception(e)

    def text_request(self, query: str) -> list[list[float]]:
        try:
            return self.embed_text(query)
        except Exception as e:
            self._handle_exception(e)

    def _handle_exception(self, e: Exception):
        raise HTTPException(status_code=500, detail=f"An exception occurred: {e}")
