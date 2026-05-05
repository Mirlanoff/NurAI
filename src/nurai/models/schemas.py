from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    vector_store: str | None = None
    retrieval: str | None = None
    reranker: str | None = None
    documents_indexed: int | None = None


class DocumentUploadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    source: str = Field(default="manual", min_length=1, max_length=500)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str
    score: float
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: list[SourceChunk]
