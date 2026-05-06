from __future__ import annotations

from typing import Any

from nurai.agents.nodes import AgentNodes
from nurai.agents.state import AgentState


def build_graph(nodes: AgentNodes) -> Any:
    """Build the LangGraph state machine.

    Pipeline: rewrite_query -> retrieve -> rerank -> generate_answer -> guardrails -> END.

    Returns a compiled LangGraph runnable. Typed as ``Any`` because the
    LangGraph generic types are unstable across versions and only the
    ``invoke`` interface is used by the workflow wrapper.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
        msg = "Install optional agent dependencies with `pip install -e '.[agent]'`."
        raise RuntimeError(msg) from exc

    graph: Any = StateGraph(AgentState)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("guardrails", nodes.guardrails)
    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", "guardrails")
    graph.add_edge("guardrails", END)
    return graph.compile()
