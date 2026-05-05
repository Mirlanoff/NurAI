import re

from nurai.core.config import Settings
from nurai.core.exceptions import EmptyDocumentError, VectorStoreUnavailableError
from nurai.embeddings.base import Embedder
from nurai.ingestion.chunker import TextChunker
from nurai.models.domain import ScoredChunk
from nurai.models.schemas import ChatResponse, DocumentIngestResponse, SearchResponse, SourceChunk
from nurai.vectorstores.base import VectorStore


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest_text(
        self,
        title: str,
        text: str,
        source: str,
        metadata: dict[str, str],
    ) -> DocumentIngestResponse:
        document = self._chunker.build_document(
            title=title,
            text=text,
            source=source,
            metadata=metadata,
        )
        chunks = self._chunker.split(document)
        if not chunks:
            raise EmptyDocumentError("document has no indexable text")
        vectors = [self._embedder.embed(chunk.text) for chunk in chunks]
        self._vector_store.upsert(chunks=chunks, vectors=vectors)
        return DocumentIngestResponse(document_id=document.id, chunks_indexed=len(chunks))

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        scored_chunks = self._retrieve(query=query, top_k=top_k)
        return SearchResponse(
            query=query,
            results=[self._to_source_chunk(scored_chunk) for scored_chunk in scored_chunks],
        )

    def chat(self, question: str, top_k: int | None = None) -> ChatResponse:
        scored_chunks = self._retrieve(query=question, top_k=top_k)
        sources = [self._to_source_chunk(scored_chunk) for scored_chunk in scored_chunks]
        answer = self._build_answer(question=question, scored_chunks=scored_chunks)
        confidence = self._confidence(scored_chunks)
        return ChatResponse(
            question=question,
            answer=answer,
            confidence=confidence,
            sources=sources,
        )

    def _retrieve(self, query: str, top_k: int | None) -> list[ScoredChunk]:
        limit = top_k or self._settings.default_top_k
        query_vector = self._embedder.embed(query)
        return self._vector_store.search(vector=query_vector, top_k=limit)

    def ensure_ready(self) -> None:
        if not self._vector_store.healthcheck():
            raise VectorStoreUnavailableError("vector store is not ready")

    def documents_indexed(self) -> int:
        return self._vector_store.count()

    def _build_answer(self, question: str, scored_chunks: list[ScoredChunk]) -> str:
        if not scored_chunks:
            return "I do not have enough indexed context to answer this question yet."

        question_terms = self._terms(question)
        ranked_sentences: list[tuple[int, float, str]] = []
        for scored_chunk in scored_chunks:
            sentences = self._sentences(scored_chunk.chunk.text)
            for sentence in sentences:
                overlap = len(question_terms.intersection(self._terms(sentence)))
                ranked_sentences.append((overlap, scored_chunk.score, sentence))

        ranked_sentences.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
        selected = [sentence for overlap, _, sentence in ranked_sentences if overlap > 0][:3]
        if not selected:
            selected = [scored_chunks[0].chunk.text]

        return " ".join(selected)

    def _confidence(self, scored_chunks: list[ScoredChunk]) -> float:
        if not scored_chunks:
            return 0.0
        best_score = max(scored_chunk.score for scored_chunk in scored_chunks)
        return round(max(0.0, min(1.0, best_score)), 3)

    def _to_source_chunk(self, scored_chunk: ScoredChunk) -> SourceChunk:
        chunk = scored_chunk.chunk
        return SourceChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            title=chunk.title,
            source=chunk.source,
            text=chunk.text,
            score=round(scored_chunk.score, 4),
            metadata=chunk.metadata,
        )

    def _terms(self, text: str) -> set[str]:
        return set(re.findall(r"[\wа-яА-ЯёЁ]+", text.lower()))

    def _sentences(self, text: str) -> list[str]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)]
        return [sentence for sentence in sentences if sentence]
