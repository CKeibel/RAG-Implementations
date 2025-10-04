from app.dependencies import get_embedding_service
from app.schemas import ImageRequest, ImageResponse, TextRequest, TextResponse
from app.service import EmbeddingService
from fastapi import APIRouter, Depends

embedding_router = APIRouter()


@embedding_router.post("/embed_images", response_model=ImageResponse)
def embed_images(
    request: ImageRequest, service: EmbeddingService = Depends(get_embedding_service)
) -> ImageResponse:
    embeddings = service.image_request(request.images_base64)
    return ImageResponse(embeddings=embeddings)


@embedding_router.post("/embed_query", response_model=TextResponse)
def embed_query(
    request: TextRequest, service: EmbeddingService = Depends(get_embedding_service)
) -> TextResponse:
    embeddings = service.text_request(request.query)
    return TextResponse(embeddings=embeddings)
