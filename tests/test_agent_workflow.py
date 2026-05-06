from nurai.agents.workflow import AgentWorkflow
from nurai.core.config import Settings
from nurai.embeddings.hashing import HashingEmbedder
from nurai.retrieval.bm25 import BM25Index
from nurai.services.rag import RagService
from nurai.vectorstores.memory import InMemoryVectorStore


def _build_workflow(min_confidence: float = 0.0, ingest: bool = True) -> AgentWorkflow:
    settings = Settings(
        retrieval_backend="hybrid",
        reranker_backend="lexical",
        vector_store_backend="memory",
        agent_min_confidence=min_confidence,
        rate_limit_enabled=False,
    )
    service = RagService(
        settings=settings,
        embedder=HashingEmbedder(dimensions=settings.embedding_dimensions),
        vector_store=InMemoryVectorStore(),
        bm25_index=BM25Index(),
        reranker=None,
    )
    if ingest:
        service.ingest_text(
            title="RAG handbook",
            text=(
                "Retrieval-Augmented Generation combines retrieval with generation. "
                "RAG systems need chunking, embeddings, vector search, and reranking."
            ),
            source="unit-test",
            metadata={},
        )
    return AgentWorkflow(rag_service=service)


def test_agent_workflow_runs_full_pipeline() -> None:
    workflow = _build_workflow()

    response = workflow.run(question="What does RAG need?", top_k=3)

    assert response.question == "What does RAG need?"
    assert response.answer
    assert response.confidence > 0
    assert response.refusal_reason is None
    assert response.sources
    assert response.query_variants
    trace_names = [step.name for step in response.trace]
    assert trace_names == [
        "rewrite_query",
        "retrieve",
        "rerank",
        "generate_answer",
        "guardrails",
    ]


def test_agent_workflow_refuses_when_no_documents_indexed() -> None:
    workflow = _build_workflow(ingest=False)

    response = workflow.run(question="What does RAG need?", top_k=3)

    assert response.refusal_reason == "no_supporting_context"
    assert response.sources == []
    assert "trustworthy context" in response.answer


def test_agent_workflow_healthcheck_returns_true_with_langgraph_installed() -> None:
    workflow = _build_workflow()
    assert workflow.healthcheck() is True
