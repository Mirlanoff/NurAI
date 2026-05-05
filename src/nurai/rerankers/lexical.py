from nurai.models.domain import ScoredChunk
from nurai.retrieval.bm25 import tokenize


class LexicalReranker:
    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        query_terms = set(tokenize(query))
        reranked = [
            ScoredChunk(
                chunk=scored_chunk.chunk,
                score=scored_chunk.score
                + self._overlap_score(query_terms, scored_chunk.chunk.text),
            )
            for scored_chunk in chunks
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    def _overlap_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = set(tokenize(text))
        return len(query_terms.intersection(text_terms)) / len(query_terms)
