from collections.abc import Sequence
from typing import Protocol


class SentenceTransformerModel(Protocol):
    def get_sentence_embedding_dimension(self) -> int | None:
        ...

    def encode(self, sentences: str, normalize_embeddings: bool) -> Sequence[float]:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "Install optional ML dependencies with `pip install -e '.[ml]'`."
            raise RuntimeError(msg) from exc

        self._model: SentenceTransformerModel = SentenceTransformer(model_name)  # type: ignore[no-untyped-call]
        dimensions = self._model.get_sentence_embedding_dimension()
        if dimensions is None:
            msg = f"model {model_name} did not report embedding dimensions"
            raise ValueError(msg)
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in embedding]
