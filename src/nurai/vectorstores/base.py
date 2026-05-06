from typing import Protocol

from nurai.models.domain import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ...

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        ...

    def count(self) -> int:
        ...

    def healthcheck(self) -> bool:
        ...
