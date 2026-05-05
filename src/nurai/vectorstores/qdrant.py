from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from nurai.models.domain import Chunk, ScoredChunk


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        timeout_seconds: float,
    ) -> None:
        self._client = QdrantClient(url=url, timeout=int(timeout_seconds))
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._ensure_collection()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            msg = "chunks and vectors must have the same length"
            raise ValueError(msg)

        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "source": chunk.source,
                    "position": chunk.position,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self._collection_name, points=points)

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        query_response = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        scored_chunks: list[ScoredChunk] = []
        for result in query_response.points:
            payload = result.payload or {}
            metadata_payload = payload.get("metadata")
            metadata = metadata_payload if isinstance(metadata_payload, dict) else {}
            chunk = Chunk(
                id=str(result.id),
                document_id=str(payload.get("document_id", "")),
                title=str(payload.get("title", "")),
                text=str(payload.get("text", "")),
                source=str(payload.get("source", "")),
                position=int(payload.get("position", 0)),
                metadata={str(key): str(value) for key, value in metadata.items()},
            )
            scored_chunks.append(ScoredChunk(chunk=chunk, score=float(result.score)))
        return scored_chunks

    def count(self) -> int:
        result = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        )
        return int(result.count)

    def healthcheck(self) -> bool:
        try:
            self._client.get_collection(collection_name=self._collection_name)
        except Exception:
            return False
        return True

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(collection_name=self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )
