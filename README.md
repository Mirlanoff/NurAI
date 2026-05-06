# NurAI

NurAI is a production-style RAG assistant for corporate knowledge bases. It is designed as a portfolio project for ML/LLM Engineer roles and demonstrates document ingestion, chunking, embeddings, vector search, RAG answers, FastAPI integration, Docker, CI, tests, linting, and type checking.

## Features

- FastAPI service with OpenAPI docs.
- Document ingestion from JSON payloads or text file uploads.
- Text cleaning, deterministic document IDs, chunking, and metadata.
- Local hashing embeddings for reproducible development without paid API keys.
- In-memory vector search for local development and tests.
- Qdrant vector store backend for production-like deployments.
- Hybrid retrieval with BM25 + vector score fusion.
- Optional lexical reranker and cross-encoder reranker integration.
- Optional sentence-transformers embeddings for production-grade semantic retrieval.
- Optional Ragas dataset/evaluation helpers.
- RAG-style chat endpoint with source citations and confidence score.
- LangGraph agent workflow (`/agent/chat`) with query rewriting, retrieval, reranking, answer generation, guardrails, and full step-by-step trace.
- Readiness checks, upload size limits, structured error handling, and Docker healthchecks.
- Optional API-key auth, in-memory rate limiting, request IDs, JSON request logs, and Prometheus metrics.
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
  -> Optional BM25 hybrid fusion
  -> Optional reranker
  -> Search and citation-based answer
```

LangGraph agent workflow (`POST /agent/chat`):

```text
question
  -> rewrite_query   (heuristic variants: original + keyword form)
  -> retrieve        (vector + optional BM25 fusion across all variants)
  -> rerank          (lexical or cross-encoder reranker)
  -> generate_answer (citation-grounded extractive answer + confidence)
  -> guardrails      (refuse on no-context or low-confidence)
```

Each step records a trace event that the agent endpoint returns alongside the answer, sources, and refusal reason (if any).

## Quick start

```bash
make install
make run
```

Open:

- API docs: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- Readiness: <http://127.0.0.1:8000/ready>
- Metrics: <http://127.0.0.1:8000/metrics>

## Configuration

Copy `.env.example` to `.env` and tune values as needed.

| Variable | Description | Default |
| --- | --- | --- |
| `NURAI_VECTOR_STORE_BACKEND` | `memory` or `qdrant` | `qdrant` in Docker, `memory` locally |
| `NURAI_EMBEDDING_BACKEND` | `hashing` or `sentence_transformers` | `hashing` |
| `NURAI_RETRIEVAL_BACKEND` | `vector` or `hybrid` | `hybrid` in Docker |
| `NURAI_RERANKER_BACKEND` | `none`, `lexical`, or `cross_encoder` | `lexical` in Docker |
| `NURAI_QDRANT_URL` | Qdrant HTTP URL | `http://qdrant:6333` |
| `NURAI_QDRANT_COLLECTION` | Qdrant collection name | `nurai_documents` |
| `NURAI_CHUNK_SIZE` | Chunk size in characters | `700` |
| `NURAI_CHUNK_OVERLAP` | Chunk overlap in characters | `100` |
| `NURAI_DEFAULT_TOP_K` | Default retrieval limit | `5` |
| `NURAI_MAX_UPLOAD_BYTES` | Upload endpoint size limit | `2000000` |
| `NURAI_API_KEY` | Optional API key required via `X-API-Key` | empty |
| `NURAI_RATE_LIMIT_ENABLED` | Enable in-memory per-client rate limit | `true` |
| `NURAI_RATE_LIMIT_REQUESTS` | Requests allowed per window | `120` |
| `NURAI_RATE_LIMIT_WINDOW_SECONDS` | Rate-limit window in seconds | `60` |
| `NURAI_METRICS_ENABLED` | Enable Prometheus `/metrics` endpoint | `true` |
| `NURAI_REQUEST_ID_HEADER` | Request correlation header | `X-Request-ID` |
| `NURAI_AGENT_ENABLED` | Expose the LangGraph agent endpoint | `true` |
| `NURAI_AGENT_MIN_CONFIDENCE` | Minimum confidence for the agent to answer | `0.05` |
| `NURAI_AGENT_MAX_QUERY_REWRITES` | Maximum number of query variants the agent generates | `2` |

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

Agent chat (LangGraph workflow with query rewriting, reranking, and guardrails):

```bash
curl -X POST http://127.0.0.1:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What components are needed for RAG?", "top_k": 3}'
```

The response includes the rewritten `query_variants`, a `trace` of the executed
nodes, and a `refusal_reason` field set when guardrails block the answer.

If `NURAI_API_KEY` is configured, pass it on protected endpoints:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $NURAI_API_KEY" \
  -d '{"query": "What does RAG need?", "top_k": 3}'
```

Metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

## Quality checks

```bash
make quality
```

CI runs ruff, mypy, pytest, and Docker image build checks.

## Optional ML and evaluation extras

Install production semantic retrieval dependencies:

```bash
make install-ml
NURAI_EMBEDDING_BACKEND=sentence_transformers make run
```

Install Ragas evaluation helpers:

```bash
make install-eval
```

Install LangGraph agent dependencies (already pulled in by `make install` for
development; only needed in production deployments that opt out of `[dev]`):

```bash
make install-agent
```

## Roadmap to Middle+

- Add hosted embedding APIs and production model caching.
- Add BGE reranker or Cohere rerank.
- Add full Ragas quality reports and regression gates.
- Add MLflow for experiment tracking.
- Add Celery/Redis for asynchronous ingestion.
- Promote the LangGraph agent from heuristic rewriting to LLM-driven query rewriting and answer synthesis.
- Add Grafana dashboards and alerting rules.
