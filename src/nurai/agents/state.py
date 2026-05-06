from dataclasses import dataclass
from typing import TypedDict

from nurai.models.domain import ScoredChunk


@dataclass(frozen=True)
class TraceEvent:
    name: str
    detail: str


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    min_confidence: float
    query_variants: list[str]
    candidates: list[ScoredChunk]
    scored_chunks: list[ScoredChunk]
    answer: str
    confidence: float
    refusal_reason: str | None
    trace: list[TraceEvent]


def initial_state(
    question: str,
    top_k: int,
    min_confidence: float,
) -> AgentState:
    return AgentState(
        question=question,
        top_k=top_k,
        min_confidence=min_confidence,
        query_variants=[],
        candidates=[],
        scored_chunks=[],
        answer="",
        confidence=0.0,
        refusal_reason=None,
        trace=[],
    )
