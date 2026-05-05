from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse

from nurai.api.security import require_api_key
from nurai.core.config import Settings, get_settings
from nurai.core.exceptions import UploadTooLargeError
from nurai.core.metrics import metrics_registry
from nurai.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentIngestResponse,
    DocumentUploadRequest,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from nurai.services.dependencies import get_rag_service
from nurai.services.rag import RagService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name, version=settings.app_version)


@router.get("/ready", response_model=HealthResponse)
def ready(
    settings: Settings = Depends(get_settings),
    rag_service: RagService = Depends(get_rag_service),
) -> HealthResponse:
    rag_service.ensure_ready()
    return HealthResponse(
        status="ready",
        app=settings.app_name,
        version=settings.app_version,
        vector_store=settings.vector_store_backend,
        retrieval=settings.retrieval_backend,
        reranker=settings.reranker_backend,
        documents_indexed=rag_service.documents_indexed(),
    )


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(settings: Settings = Depends(get_settings)) -> PlainTextResponse:
    if not settings.metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="metrics disabled")
    return PlainTextResponse(metrics_registry.render_prometheus())


@router.post("/documents", response_model=DocumentIngestResponse)
def ingest_document(
    payload: DocumentUploadRequest,
    _: None = Depends(require_api_key),
    rag_service: RagService = Depends(get_rag_service),
) -> DocumentIngestResponse:
    return rag_service.ingest_text(
        title=payload.title,
        text=payload.text,
        source=payload.source,
        metadata=payload.metadata,
    )


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile,
    _: None = Depends(require_api_key),
    rag_service: RagService = Depends(get_rag_service),
) -> DocumentIngestResponse:
    content = await file.read()
    settings = get_settings()
    if len(content) > settings.max_upload_bytes:
        raise UploadTooLargeError("uploaded file exceeds configured size limit")
    filename = file.filename or "uploaded-document"
    text = content.decode("utf-8")
    return rag_service.ingest_text(
        title=filename,
        text=text,
        source=f"upload:{filename}",
        metadata={"content_type": file.content_type or "text/plain"},
    )


@router.post("/search", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    _: None = Depends(require_api_key),
    rag_service: RagService = Depends(get_rag_service),
) -> SearchResponse:
    return rag_service.search(query=payload.query, top_k=payload.top_k)


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    _: None = Depends(require_api_key),
    rag_service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    return rag_service.chat(question=payload.question, top_k=payload.top_k)
