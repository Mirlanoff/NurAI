import re
from collections.abc import Iterable

from nurai.core.config import Settings
from nurai.core.exceptions import EmptyDocumentError, VectorStoreUnavailableError
from nurai.embeddings.base import Embedder
from nurai.ingestion.chunker import TextChunker
from nurai.models.domain import ScoredChunk
from nurai.models.schemas import ChatResponse, DocumentIngestResponse, SearchResponse, SourceChunk
from nurai.rerankers.base import Reranker
from nurai.retrieval.bm25 import BM25Index
from nurai.vectorstores.base import VectorStore


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker = reranker
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
        if self._bm25_index is not None:
            self._bm25_index.upsert(chunks)
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
        candidate_limit = limit * self._settings.retrieval_candidate_multiplier
        vector_results = self._vector_store.search(vector=query_vector, top_k=candidate_limit)
        if self._settings.retrieval_backend == "hybrid" and self._bm25_index is not None:
            lexical_results = self._bm25_index.search(query=query, top_k=candidate_limit)
            results = self._merge_hybrid_results(
                vector_results=vector_results,
                lexical_results=lexical_results,
            )
        else:
            results = vector_results

        if self._reranker is not None:
            return self._reranker.rerank(query=query, chunks=results, top_k=limit)
        return results[:limit]

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

    def _merge_hybrid_results(
        self,
        vector_results: list[ScoredChunk],
        lexical_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        vector_weight = self._settings.hybrid_vector_weight
        lexical_weight = 1.0 - vector_weight
        vector_scores = self._normalize_scores(vector_results)
        lexical_scores = self._normalize_scores(lexical_results)
        chunks_by_id = {
            scored_chunk.chunk.id: scored_chunk.chunk
            for scored_chunk in [*vector_results, *lexical_results]
        }
        merged = [
            ScoredChunk(
                chunk=chunk,
                score=vector_weight * vector_scores.get(chunk_id, 0.0)
                + lexical_weight * lexical_scores.get(chunk_id, 0.0),
            )
            for chunk_id, chunk in chunks_by_id.items()
        ]
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged

    def _normalize_scores(self, scored_chunks: Iterable[ScoredChunk]) -> dict[str, float]:
        scored_list = list(scored_chunks)
        if not scored_list:
            return {}
        scores = [item.score for item in scored_list]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return {item.chunk.id: 1.0 for item in scored_list}
        return {
            item.chunk.id: (item.score - min_score) / (max_score - min_score)
            for item in scored_list
        }
