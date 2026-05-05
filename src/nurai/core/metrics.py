from collections import Counter
from threading import Lock


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests_total: Counter[tuple[str, str, int]] = Counter()
        self._rate_limited_total = 0

    def record_request(self, method: str, path: str, status_code: int) -> None:
        with self._lock:
            self._requests_total[(method, path, status_code)] += 1

    def record_rate_limited(self) -> None:
        with self._lock:
            self._rate_limited_total += 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP nurai_requests_total Total HTTP requests.",
            "# TYPE nurai_requests_total counter",
        ]
        with self._lock:
            for (method, path, status_code), count in sorted(self._requests_total.items()):
                lines.append(
                    "nurai_requests_total"
                    f'{{method="{method}",path="{path}",status="{status_code}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP nurai_rate_limited_total Total rate-limited HTTP requests.",
                    "# TYPE nurai_rate_limited_total counter",
                    f"nurai_rate_limited_total {self._rate_limited_total}",
                ]
            )
        return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()
