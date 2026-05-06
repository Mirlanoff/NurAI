from nurai.agents.nodes import AgentNodes, _query_variants
from nurai.agents.state import initial_state
from nurai.core.config import Settings
from nurai.embeddings.hashing import HashingEmbedder
from nurai.retrieval.bm25 import BM25Index
from nurai.services.rag import RagService
from nurai.vectorstores.memory import InMemoryVectorStore


def _build_service() -> RagService:
    settings = Settings(
        retrieval_backend="hybrid",
        reranker_backend="lexical",
        vector_store_backend="memory",
        rate_limit_enabled=False,
    )
    service = RagService(
        settings=settings,
        embedder=HashingEmbedder(dimensions=settings.embedding_dimensions),
        vector_store=InMemoryVectorStore(),
        bm25_index=BM25Index(),
        reranker=None,
    )
    service.ingest_text(
        title="RAG handbook",
        text=(
            "Retrieval-Augmented Generation combines retrieval with generation. "
            "RAG systems need chunking, embeddings, vector search, and reranking."
        ),
        source="unit-test",
        metadata={},
    )
    return service


def test_query_variants_includes_original_and_keyword_form() -> None:
    variants = _query_variants("What does the RAG pipeline need?", max_variants=3)

    assert variants[0] == "What does the RAG pipeline need?"
    assert "rag pipeline need" in variants
    assert len(variants) <= 3
    assert all(variant.strip() for variant in variants)


def test_query_variants_handles_empty_question() -> None:
    assert _query_variants("", max_variants=3) == []


def test_rewrite_query_node_records_variants_and_trace() -> None:
    service = _build_service()
    nodes = AgentNodes(rag_service=service, max_query_rewrites=2)
    state = initial_state(question="What does RAG need?", top_k=3, min_confidence=0.0)

    update = nodes.rewrite_query(state)

    assert update["query_variants"]
    assert any(event.name == "rewrite_query" for event in update["trace"])


def test_retrieve_node_returns_unique_candidates_for_multiple_variants() -> None:
    service = _build_service()
    nodes = AgentNodes(rag_service=service, max_query_rewrites=2)
    state = initial_state(question="What does RAG need?", top_k=3, min_confidence=0.0)
    state["query_variants"] = ["What does RAG need?", "rag chunking embeddings"]

    update = nodes.retrieve(state)

    assert update["candidates"]
    chunk_ids = {scored.chunk.id for scored in update["candidates"]}
    assert len(chunk_ids) == len(update["candidates"])


def test_guardrails_refuses_when_no_chunks() -> None:
    service = _build_service()
    nodes = AgentNodes(rag_service=service, max_query_rewrites=2)
    state = initial_state(question="anything", top_k=3, min_confidence=0.5)
    state["scored_chunks"] = []
    state["confidence"] = 0.0
    state["answer"] = "should be replaced"

    update = nodes.guardrails(state)

    assert update["refusal_reason"] == "no_supporting_context"
    assert "trustworthy context" in update["answer"]


def test_guardrails_refuses_when_confidence_below_threshold() -> None:
    service = _build_service()
    nodes = AgentNodes(rag_service=service, max_query_rewrites=2)
    candidates = service.retrieve_candidates(query="What does RAG need?", top_k=3)
    state = initial_state(question="What does RAG need?", top_k=3, min_confidence=0.99)
    state["scored_chunks"] = candidates
    state["confidence"] = 0.1
    state["answer"] = "draft answer"

    update = nodes.guardrails(state)

    assert update["refusal_reason"] == "low_confidence"


def test_guardrails_passes_when_confidence_meets_threshold() -> None:
    service = _build_service()
    nodes = AgentNodes(rag_service=service, max_query_rewrites=2)
    candidates = service.retrieve_candidates(query="What does RAG need?", top_k=3)
    state = initial_state(question="What does RAG need?", top_k=3, min_confidence=0.0)
    state["scored_chunks"] = candidates
    state["confidence"] = 0.7
    state["answer"] = "draft answer"

    update = nodes.guardrails(state)

    assert update["refusal_reason"] is None
    assert update["answer"] == "draft answer"
