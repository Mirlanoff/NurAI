from __future__ import annotations

import re

from nurai.agents.state import AgentState, TraceEvent
from nurai.models.domain import ScoredChunk
from nurai.services.rag import RagService

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "have",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "please",
        "should",
        "tell",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
    }
)

_TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _append_trace(state: AgentState, event: TraceEvent) -> list[TraceEvent]:
    existing = state.get("trace") or []
    return [*existing, event]


def _query_variants(question: str, max_variants: int) -> list[str]:
    """Heuristic, network-free query rewrites used to widen retrieval."""
    base = question.strip()
    variants: list[str] = []
    if base:
        variants.append(base)
    keyword_tokens = [token for token in _tokens(base) if token not in STOPWORDS]
    if keyword_tokens:
        keyword_query = " ".join(keyword_tokens)
        if keyword_query and keyword_query.lower() != base.lower():
            variants.append(keyword_query)
    stripped_punct = re.sub(r"[^\w\sа-яА-ЯёЁ]+", " ", base)
    stripped_punct = re.sub(r"\s+", " ", stripped_punct).strip()
    if stripped_punct and stripped_punct.lower() not in {variant.lower() for variant in variants}:
        variants.append(stripped_punct)
    return variants[:max_variants]


def _merge_candidates(candidate_lists: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    """Merge candidates from multiple query variants by keeping max score per chunk."""
    best_per_chunk: dict[str, ScoredChunk] = {}
    for candidates in candidate_lists:
        for scored_chunk in candidates:
            current = best_per_chunk.get(scored_chunk.chunk.id)
            if current is None or scored_chunk.score > current.score:
                best_per_chunk[scored_chunk.chunk.id] = scored_chunk
    merged = list(best_per_chunk.values())
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged


class AgentNodes:
    """Plain node implementations used by the LangGraph workflow.

    Each node accepts and returns an `AgentState` slice so that the graph
    framework merges the partial updates back into the global state.
    """

    def __init__(self, rag_service: RagService, max_query_rewrites: int) -> None:
        self._rag_service = rag_service
        self._max_query_rewrites = max_query_rewrites

    def rewrite_query(self, state: AgentState) -> AgentState:
        question = state["question"]
        variants = _query_variants(question, max_variants=self._max_query_rewrites)
        if not variants:
            variants = [question]
        event = TraceEvent(
            name="rewrite_query",
            detail=f"generated {len(variants)} variant(s)",
        )
        return AgentState(
            query_variants=variants,
            trace=_append_trace(state, event),
        )

    def retrieve(self, state: AgentState) -> AgentState:
        variants = state.get("query_variants") or [state["question"]]
        top_k = state["top_k"]
        candidate_lists = [
            self._rag_service.retrieve_candidates(query=variant, top_k=top_k)
            for variant in variants
        ]
        merged = _merge_candidates(candidate_lists)
        event = TraceEvent(
            name="retrieve",
            detail=f"retrieved {len(merged)} unique candidate(s) across "
            f"{len(variants)} variant(s)",
        )
        return AgentState(
            candidates=merged,
            trace=_append_trace(state, event),
        )

    def rerank(self, state: AgentState) -> AgentState:
        candidates = state.get("candidates") or []
        question = state["question"]
        top_k = state["top_k"]
        reranked = self._rag_service.apply_reranker(
            query=question,
            scored_chunks=candidates,
            top_k=top_k,
        )
        event = TraceEvent(
            name="rerank",
            detail=f"selected {len(reranked)} chunk(s) from {len(candidates)} candidate(s)",
        )
        return AgentState(
            scored_chunks=reranked,
            trace=_append_trace(state, event),
        )

    def generate_answer(self, state: AgentState) -> AgentState:
        question = state["question"]
        scored_chunks = state.get("scored_chunks") or []
        answer = self._rag_service.build_answer(question=question, scored_chunks=scored_chunks)
        confidence = self._rag_service.confidence(scored_chunks)
        event = TraceEvent(
            name="generate_answer",
            detail=f"answer length={len(answer)} confidence={confidence:.3f}",
        )
        return AgentState(
            answer=answer,
            confidence=confidence,
            trace=_append_trace(state, event),
        )

    def guardrails(self, state: AgentState) -> AgentState:
        confidence = state.get("confidence", 0.0)
        min_confidence = state["min_confidence"]
        scored_chunks = state.get("scored_chunks") or []
        refusal_reason: str | None = None
        answer = state.get("answer", "")
        if not scored_chunks:
            refusal_reason = "no_supporting_context"
        elif confidence < min_confidence:
            refusal_reason = "low_confidence"
        if refusal_reason is not None:
            answer = (
                "I do not have enough trustworthy context to answer this question. "
                "Please add more documents or refine your question."
            )
        event = TraceEvent(
            name="guardrails",
            detail=(
                f"passed (confidence={confidence:.3f} >= {min_confidence:.3f})"
                if refusal_reason is None
                else f"refused: {refusal_reason}"
            ),
        )
        return AgentState(
            answer=answer,
            refusal_reason=refusal_reason,
            trace=_append_trace(state, event),
        )
