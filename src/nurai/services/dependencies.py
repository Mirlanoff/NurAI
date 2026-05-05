from functools import lru_cache

from nurai.core.config import get_settings
from nurai.embeddings.hashing import HashingEmbedder
from nurai.services.rag import RagService
from nurai.vectorstores.base import VectorStore
from nurai.vectorstores.memory import InMemoryVectorStore
from nurai.vectorstores.qdrant import QdrantVectorStore


def build_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store_backend == "qdrant":
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dimensions,
            timeout_seconds=settings.qdrant_timeout_seconds,
        )
    return InMemoryVectorStore()


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        settings=settings,
        embedder=HashingEmbedder(dimensions=settings.embedding_dimensions),
        vector_store=build_vector_store(),
    )
