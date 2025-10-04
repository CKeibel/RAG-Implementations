from contextlib import asynccontextmanager

from app.service import EmbeddingService
from fastapi import FastAPI

shared_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application's lifespan events."""
    shared_state["embedding_service"] = EmbeddingService()
    yield
    shared_state.clear()
