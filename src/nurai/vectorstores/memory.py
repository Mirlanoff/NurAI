from nurai.models.domain import Chunk, ScoredChunk


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        msg = "vectors must have the same dimensions"
        raise ValueError(msg)
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            msg = "chunks and vectors must have the same length"
            raise ValueError(msg)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._items[chunk.id] = (chunk, vector)

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=cosine_similarity(vector, item_vector))
            for chunk, item_vector in self._items.values()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._items)
