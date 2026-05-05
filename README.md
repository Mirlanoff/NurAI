# NurAI

NurAI is a production-style RAG assistant for corporate knowledge bases. It is designed as a portfolio project for ML/LLM Engineer roles and demonstrates document ingestion, chunking, embeddings, vector search, RAG answers, FastAPI integration, Docker, tests, linting, and type checking.

## Features

- FastAPI service with OpenAPI docs.
- Document ingestion from JSON payloads or text file uploads.
- Text cleaning, deterministic document IDs, chunking, and metadata.
- Local hashing embeddings for reproducible development without paid API keys.
- In-memory vector search for MVP and tests.
- Qdrant included in Docker Compose for the production-ready next step.
- RAG-style chat endpoint with source citations and confidence score.
- pytest, ruff, and mypy configuration.

## Architecture

```text
Client
  -> FastAPI routes
  -> RagService
  -> TextChunker
  -> HashingEmbedder
  -> InMemoryVectorStore / Qdrant-ready boundary
  -> Search and citation-based answer
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn nurai.main:app --reload
```

Open:

- API docs: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>

## Docker

```bash
docker compose up --build
```

## Example usage

Index a document:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "RAG handbook",
    "source": "manual",
    "metadata": {"team": "ml"},
    "text": "Retrieval-Augmented Generation combines document retrieval with answer generation. RAG systems need chunking, embeddings, vector search, reranking, and evaluation."
  }'
```

Search:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What does RAG need?", "top_k": 3}'
```

Chat:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What components are needed for RAG?", "top_k": 3}'
```

## Quality checks

```bash
ruff check .
mypy src
pytest
```

## Roadmap to Middle+

- Replace hashing embeddings with sentence-transformers or hosted embedding APIs.
- Add Qdrant backend implementation behind the vector store interface.
- Add hybrid search: BM25 + vector search.
- Add reranking with BGE reranker or Cohere rerank.
- Add LangGraph workflow for query rewriting, retrieval, reranking, answer generation, and guardrails.
- Add Ragas evaluation dataset and quality reports.
- Add MLflow for experiment tracking.
- Add Celery/Redis for asynchronous ingestion.
- Add auth, rate limiting, Prometheus metrics, and Grafana dashboards.
