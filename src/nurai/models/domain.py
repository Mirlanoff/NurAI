from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    text: str
    source: str
    position: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
