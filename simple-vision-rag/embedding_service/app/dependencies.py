from app.lifespan import shared_state
from app.service import EmbeddingService
from fastapi import HTTPException


def get_embedding_service() -> EmbeddingService:
    service = shared_state.get("embedding_service")
    if not service:
        raise HTTPException(status_code=503, detail="Embedding Service not available.")
    return service
