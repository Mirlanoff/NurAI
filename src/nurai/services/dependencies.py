from functools import lru_cache

from nurai.core.config import get_settings
from nurai.embeddings.hashing import HashingEmbedder
from nurai.services.rag import RagService
from nurai.vectorstores.memory import InMemoryVectorStore


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        settings=settings,
        embedder=HashingEmbedder(dimensions=settings.embedding_dimensions),
        vector_store=InMemoryVectorStore(),
    )
