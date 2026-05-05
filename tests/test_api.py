from fastapi.testclient import TestClient

from nurai.api.app import create_app
from nurai.services.dependencies import get_rag_service


def test_health_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_search_and_chat_flow() -> None:
    get_rag_service.cache_clear()
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
