import math
import re
from collections import Counter

from nurai.models.domain import Chunk, ScoredChunk

TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: dict[str, Chunk] = {}
        self._term_counts: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._doc_frequency: Counter[str] = Counter()
        self._average_doc_length = 0.0

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self._rebuild()

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        if not query_terms or not self._chunks:
            return []

        scored: list[ScoredChunk] = []
        for chunk_id, chunk in self._chunks.items():
            score = self._score_chunk(chunk_id=chunk_id, query_terms=query_terms)
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _rebuild(self) -> None:
        self._term_counts.clear()
        self._doc_lengths.clear()
        self._doc_frequency.clear()

        for chunk_id, chunk in self._chunks.items():
            counts = Counter(tokenize(chunk.text))
            self._term_counts[chunk_id] = counts
            self._doc_lengths[chunk_id] = sum(counts.values())
            self._doc_frequency.update(counts.keys())

        total_length = sum(self._doc_lengths.values())
        self._average_doc_length = (
            total_length / len(self._doc_lengths) if self._doc_lengths else 0.0
        )

    def _score_chunk(self, chunk_id: str, query_terms: list[str]) -> float:
        score = 0.0
        doc_count = len(self._chunks)
        doc_length = self._doc_lengths[chunk_id]
        term_counts = self._term_counts[chunk_id]

        for term in query_terms:
            term_frequency = term_counts[term]
            if term_frequency == 0:
                continue
            doc_frequency = self._doc_frequency[term]
            idf = math.log(1 + (doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = term_frequency + self._k1 * (
                1 - self._b + self._b * doc_length / self._average_doc_length
            )
            score += idf * (term_frequency * (self._k1 + 1)) / denominator
        return score
