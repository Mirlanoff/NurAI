from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NurAI"
    app_version: str = "0.1.0"
    environment: str = Field(default="local")
    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    embedding_backend: str = Field(default="hashing", pattern="^(hashing|sentence_transformers)$")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = Field(default=384, ge=32)
    default_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_backend: str = Field(default="vector", pattern="^(vector|hybrid)$")
    retrieval_candidate_multiplier: int = Field(default=3, ge=1, le=10)
    hybrid_vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    reranker_backend: str = Field(default="none", pattern="^(none|lexical|cross_encoder)$")
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    vector_store_backend: str = Field(default="memory", pattern="^(memory|qdrant)$")
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "nurai_documents"
    qdrant_timeout_seconds: float = Field(default=5.0, gt=0)
    max_upload_bytes: int = Field(default=2_000_000, ge=1)
    log_level: str = "INFO"
    api_key: str = ""
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=120, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    metrics_enabled: bool = True
    request_id_header: str = "X-Request-ID"

    model_config = SettingsConfigDict(env_prefix="NURAI_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
