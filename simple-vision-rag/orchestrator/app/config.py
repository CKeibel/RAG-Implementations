from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_SERVICE_URL: str = "http://localhost:8001"
    VLLM_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.0


settings = Settings()
