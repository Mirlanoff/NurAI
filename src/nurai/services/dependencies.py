from functools import lru_cache

from nurai.core.config import get_settings
from nurai.embeddings.base import Embedder
from nurai.embeddings.hashing import HashingEmbedder
from nurai.embeddings.sentence_transformers import SentenceTransformerEmbedder
from nurai.rerankers.base import Reranker
from nurai.rerankers.cross_encoder import CrossEncoderReranker
from nurai.rerankers.lexical import LexicalReranker
from nurai.retrieval.bm25 import BM25Index
from nurai.services.rag import RagService
from nurai.vectorstores.base import VectorStore
from nurai.vectorstores.memory import InMemoryVectorStore
from nurai.vectorstores.qdrant import QdrantVectorStore


def build_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=settings.embedding_model_name)
    return HashingEmbedder(dimensions=settings.embedding_dimensions)


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


def build_reranker() -> Reranker | None:
    settings = get_settings()
    if settings.reranker_backend == "lexical":
        return LexicalReranker()
    if settings.reranker_backend == "cross_encoder":
        return CrossEncoderReranker(model_name=settings.reranker_model_name)
    return None


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        settings=settings,
        embedder=build_embedder(),
        vector_store=build_vector_store(),
        bm25_index=BM25Index() if settings.retrieval_backend == "hybrid" else None,
        reranker=build_reranker(),
    )
