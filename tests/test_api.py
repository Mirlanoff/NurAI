from fastapi.testclient import TestClient

from nurai.api.app import create_app
from nurai.core.config import get_settings
from nurai.services.dependencies import get_rag_service


def clear_caches() -> None:
    get_settings.cache_clear()
    get_rag_service.cache_clear()


def test_health_endpoint() -> None:
    clear_caches()
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_endpoint_reports_vector_store() -> None:
    clear_caches()
    app = create_app()
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["vector_store"] == "memory"
    assert payload["documents_indexed"] == 0


def test_ingest_search_and_chat_flow() -> None:
    clear_caches()
    app = create_app()
    client = TestClient(app)

    ingest_response = client.post(
        "/documents",
        json={
            "title": "RAG handbook",
            "source": "unit-test",
            "metadata": {"team": "ml"},
            "text": (
                "Retrieval-Augmented Generation combines document retrieval with answer "
                "generation. RAG systems need chunking, embeddings, vector search, reranking, "
                "and evaluation."
            ),
        },
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunks_indexed"] == 1

    search_response = client.post(
        "/search",
        json={"query": "What does RAG need?", "top_k": 1},
    )

    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["results"][0]["title"] == "RAG handbook"

    chat_response = client.post(
        "/chat",
        json={"question": "What components are needed for RAG?", "top_k": 1},
    )

    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    assert chat_payload["sources"][0]["source"] == "unit-test"
    assert "RAG systems need" in chat_payload["answer"]


def test_empty_document_returns_validation_error() -> None:
    clear_caches()
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/documents",
        json={
            "title": "Empty",
            "source": "unit-test",
            "text": "   \n\t ",
            "metadata": {},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "document has no indexable text"


def test_upload_size_limit(monkeypatch) -> None:
    monkeypatch.setenv("NURAI_MAX_UPLOAD_BYTES", "4")
    clear_caches()
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/documents/upload",
        files={"file": ("large.txt", b"too large", "text/plain")},
    )

    assert response.status_code == 413
