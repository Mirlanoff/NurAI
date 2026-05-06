from typing import Protocol

from nurai.models.domain import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        ...
