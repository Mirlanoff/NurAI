from collections.abc import Sequence
from typing import Protocol

from nurai.models.domain import ScoredChunk


class CrossEncoderModel(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]]) -> Sequence[float]:
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "Install optional ML dependencies with `pip install -e '.[ml]'`."
            raise RuntimeError(msg) from exc

        self._model: CrossEncoderModel = CrossEncoder(model_name)  # type: ignore[no-untyped-call]

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        pairs = [(query, scored_chunk.chunk.text) for scored_chunk in chunks]
        scores = self._model.predict(pairs)
        reranked = [
            ScoredChunk(chunk=scored_chunk.chunk, score=float(score))
            for scored_chunk, score in zip(chunks, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
