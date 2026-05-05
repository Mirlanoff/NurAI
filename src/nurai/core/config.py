from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NurAI"
    app_version: str = "0.1.0"
    environment: str = Field(default="local")
    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    embedding_dimensions: int = Field(default=384, ge=32)
    default_top_k: int = Field(default=5, ge=1, le=20)
    vector_store_backend: str = Field(default="memory", pattern="^(memory|qdrant)$")
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "nurai_documents"
    qdrant_timeout_seconds: float = Field(default=5.0, gt=0)
    max_upload_bytes: int = Field(default=2_000_000, ge=1)

    model_config = SettingsConfigDict(env_prefix="NURAI_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
