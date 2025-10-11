import httpx
from app.config import settings
from app.services.service_mixin import ServiceMixin


class EmbeddingService(ServiceMixin):
    def __init__(self) -> None:
        self.url = settings.EMBEDDING_SERVICE_URL

    async def embed(self, text: str) -> list[list[float]]:
        try:
            async with httpx.AsyncClient() as client:
                payload = {"query": text}
                response = await client.post(self.url, json=payload)
                response.raise_for_status()
        except Exception as e:
            self._handle_error(e)
