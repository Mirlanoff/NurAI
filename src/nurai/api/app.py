from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nurai.api.middleware import register_middlewares
from nurai.api.routes import router
from nurai.core.config import get_settings
from nurai.core.exceptions import (
    EmptyDocumentError,
    UploadTooLargeError,
    VectorStoreUnavailableError,
)
from nurai.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-style RAG assistant for corporate knowledge bases.",
    )
    register_middlewares(app, settings)
    app.include_router(router)
    register_exception_handlers(app)
    return app


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(EmptyDocumentError)
    async def empty_document_handler(_: Request, exc: EmptyDocumentError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(UploadTooLargeError)
    async def upload_too_large_handler(_: Request, exc: UploadTooLargeError) -> JSONResponse:
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    @app.exception_handler(VectorStoreUnavailableError)
    async def vector_store_handler(_: Request, exc: VectorStoreUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
