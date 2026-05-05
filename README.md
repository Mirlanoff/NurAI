# NurAI

NurAI is a production-style RAG assistant for corporate knowledge bases. It is designed as a portfolio project for ML/LLM Engineer roles and demonstrates document ingestion, chunking, embeddings, vector search, RAG answers, FastAPI integration, Docker, CI, tests, linting, and type checking.

## Features

- FastAPI service with OpenAPI docs.
- Document ingestion from JSON payloads or text file uploads.
- Text cleaning, deterministic document IDs, chunking, and metadata.
- Local hashing embeddings for reproducible development without paid API keys.
- In-memory vector search for local development and tests.
- Qdrant vector store backend for production-like deployments.
- RAG-style chat endpoint with source citations and confidence score.
- Readiness checks, upload size limits, structured error handling, and Docker healthchecks.
- pytest, ruff, mypy, Makefile, and GitHub Actions CI.

## Architecture

```text
Client
  -> FastAPI routes
  -> RagService
  -> TextChunker
  -> HashingEmbedder
  -> VectorStore interface
  -> InMemoryVectorStore or QdrantVectorStore
  -> Search and citation-based answer
```

## Quick start

```bash
make install
make run
```

Open:

- API docs: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- Readiness: <http://127.0.0.1:8000/ready>

## Configuration

Copy `.env.example` to `.env` and tune values as needed.

| Variable | Description | Default |
| --- | --- | --- |
| `NURAI_VECTOR_STORE_BACKEND` | `memory` or `qdrant` | `qdrant` in Docker, `memory` locally |
| `NURAI_QDRANT_URL` | Qdrant HTTP URL | `http://qdrant:6333` |
| `NURAI_QDRANT_COLLECTION` | Qdrant collection name | `nurai_documents` |
| `NURAI_CHUNK_SIZE` | Chunk size in characters | `700` |
| `NURAI_CHUNK_OVERLAP` | Chunk overlap in characters | `100` |
| `NURAI_DEFAULT_TOP_K` | Default retrieval limit | `5` |
| `NURAI_MAX_UPLOAD_BYTES` | Upload endpoint size limit | `2000000` |

## Docker

```bash
make docker-up
```

For hot-reload local development with in-memory vector search:

```bash
make docker-dev
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
make quality
```

CI runs ruff, mypy, pytest, and Docker image build checks.

## Roadmap to Middle+

- Replace hashing embeddings with sentence-transformers or hosted embedding APIs.
- Add hybrid search: BM25 + vector search.
- Add reranking with BGE reranker or Cohere rerank.
- Add LangGraph workflow for query rewriting, retrieval, reranking, answer generation, and guardrails.
- Add Ragas evaluation dataset and quality reports.
- Add MLflow for experiment tracking.
- Add Celery/Redis for asynchronous ingestion.
- Add auth, rate limiting, Prometheus metrics, and Grafana dashboards.
