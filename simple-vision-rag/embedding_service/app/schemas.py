from pydantic import BaseModel


class ImageRequest(BaseModel):
    images_base64: str | list[str]


class ImageResponse(BaseModel):
    embeddings: list[list[list[float]]]


class TextRequest(BaseModel):
    query: str


class TextResponse(BaseModel):
    embeddings: list[list[float]]
