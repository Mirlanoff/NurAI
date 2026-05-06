from __future__ import annotations

from typing import Any, cast

from nurai.agents.nodes import AgentNodes
from nurai.agents.state import AgentState, TraceEvent, initial_state
from nurai.models.schemas import AgentChatResponse, AgentTraceStep, SourceChunk
from nurai.services.rag import RagService


class AgentWorkflow:
    """High-level wrapper around the compiled LangGraph state machine."""

    def __init__(self, rag_service: RagService) -> None:
        self._rag_service = rag_service
        self._nodes = AgentNodes(
            rag_service=rag_service,
            max_query_rewrites=rag_service.settings.agent_max_query_rewrites,
        )
        self._graph: Any | None = None

    @property
    def nodes(self) -> AgentNodes:
        return self._nodes

    def _ensure_graph(self) -> Any:
        if self._graph is None:
            from nurai.agents.graph import build_graph

            self._graph = build_graph(self._nodes)
        return self._graph

    def healthcheck(self) -> bool:
        try:
            self._ensure_graph()
        except RuntimeError:
            return False
        return True

    def run(
        self,
        question: str,
        top_k: int | None = None,
        min_confidence: float | None = None,
    ) -> AgentChatResponse:
        settings = self._rag_service.settings
        resolved_top_k = top_k or settings.default_top_k
        resolved_min_confidence = (
            min_confidence if min_confidence is not None else settings.agent_min_confidence
        )
        graph = self._ensure_graph()
        state = initial_state(
            question=question,
            top_k=resolved_top_k,
            min_confidence=resolved_min_confidence,
        )
        final_state = cast(AgentState, graph.invoke(state))
        return self._build_response(question=question, state=final_state)

    def _build_response(self, question: str, state: AgentState) -> AgentChatResponse:
        scored_chunks = state.get("scored_chunks") or []
        sources: list[SourceChunk] = [
            self._rag_service.to_source_chunk(scored_chunk) for scored_chunk in scored_chunks
        ]
        trace_events: list[TraceEvent] = state.get("trace") or []
        trace = [AgentTraceStep(name=event.name, detail=event.detail) for event in trace_events]
        return AgentChatResponse(
            question=question,
            answer=state.get("answer", ""),
            confidence=state.get("confidence", 0.0),
            sources=sources,
            refusal_reason=state.get("refusal_reason"),
            query_variants=state.get("query_variants") or [],
            trace=trace,
        )
