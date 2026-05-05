import logging
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from nurai.core.config import Settings
from nurai.core.metrics import metrics_registry

logger = logging.getLogger(__name__)


class FixedWindowRateLimiter:
    def __init__(self, requests: int, window_seconds: int) -> None:
        self._requests = requests
        self._window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - self._window_seconds
        hits = self._hits[key]
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= self._requests:
            return False
        hits.append(now)
        return True


def register_middlewares(app: object, settings: Settings) -> None:
    from fastapi import FastAPI

    fastapi_app = app if isinstance(app, FastAPI) else None
    if fastapi_app is None:
        msg = "register_middlewares expects a FastAPI app"
        raise TypeError(msg)

    limiter = FixedWindowRateLimiter(
        requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )

    @fastapi_app.middleware("http")
    async def production_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(settings.request_id_header, str(uuid4()))
        start = time.perf_counter()
        if _should_rate_limit(request=request, settings=settings):
            client_key = _client_key(request)
            if not limiter.allow(client_key):
                metrics_registry.record_rate_limited()
                rate_limited_response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "rate limit exceeded"},
                )
                rate_limited_response.headers[settings.request_id_header] = request_id
                metrics_registry.record_request(
                    request.method,
                    request.url.path,
                    rate_limited_response.status_code,
                )
                return rate_limited_response

        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers[settings.request_id_header] = request_id
        metrics_registry.record_request(request.method, request.url.path, response.status_code)
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response


def _should_rate_limit(request: Request, settings: Settings) -> bool:
    if not settings.rate_limit_enabled:
        return False
    return request.url.path not in {"/health", "/ready", "/metrics", "/docs", "/openapi.json"}


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    if request.client is None:
        return "unknown"
    return request.client.host
