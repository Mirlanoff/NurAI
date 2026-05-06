from typing import Protocol


class Embedder(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...
