from nurai.core.config import Settings
from nurai.embeddings.hashing import HashingEmbedder
from nurai.models.domain import Chunk, ScoredChunk
from nurai.rerankers.lexical import LexicalReranker
from nurai.retrieval.bm25 import BM25Index
from nurai.services.rag import RagService
from nurai.vectorstores.memory import InMemoryVectorStore


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=f"Document {chunk_id}",
        text=text,
        source="unit-test",
        position=0,
    )


def test_bm25_returns_lexically_relevant_chunks() -> None:
    index = BM25Index()
    index.upsert(
        [
            make_chunk("ml", "RAG systems use retrieval and generation"),
            make_chunk("ops", "Docker Compose runs services"),
        ]
    )

    results = index.search(query="retrieval generation", top_k=1)

    assert results[0].chunk.id == "ml"
    assert results[0].score > 0


def test_lexical_reranker_promotes_term_overlap() -> None:
    reranker = LexicalReranker()
    chunks = [
        make_chunk("low", "unrelated infrastructure text"),
        make_chunk("high", "retrieval generation embeddings"),
    ]
    scored_chunks = [
        ScoredChunk(chunk=chunks[0], score=0.1),
        ScoredChunk(chunk=chunks[1], score=0.1),
    ]

    results = reranker.rerank(query="retrieval embeddings", chunks=scored_chunks, top_k=1)

    assert results[0].chunk.id == "high"


def test_rag_service_hybrid_retrieval_merges_bm25_and_vector_scores() -> None:
    settings = Settings(retrieval_backend="hybrid", reranker_backend="lexical")
    vector_store = InMemoryVectorStore()
    service = RagService(
        settings=settings,
        embedder=HashingEmbedder(dimensions=settings.embedding_dimensions),
        vector_store=vector_store,
        bm25_index=BM25Index(),
        reranker=LexicalReranker(),
    )

    service.ingest_text(
        title="Hybrid",
        text="Hybrid retrieval combines vector search with BM25 lexical search.",
        source="unit-test",
        metadata={},
    )
    response = service.search(query="BM25 lexical", top_k=1)

    assert response.results[0].title == "Hybrid"
