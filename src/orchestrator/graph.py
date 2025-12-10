"""
Code-Orchestrator-Service - LangGraph State Machine

WBS 3.1.2: Create state graph - 4 nodes: generate, validate, rank, consensus.
WBS 3.1.3: Add conditional edges - Route based on validation results.

Patterns Applied:
- LangGraph StateGraph (ai-agents CODE_UNDERSTANDING_ORCHESTRATOR_DESIGN.md)
- Compiled graph for production use
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.orchestrator.nodes import (
    build_consensus,
    generate_terms,
    rank_terms,
    validate_terms,
)
from src.orchestrator.routing import should_skip_ranking
from src.orchestrator.state import OrchestratorState


def create_orchestrator_graph() -> CompiledStateGraph[Any]:
    """Create the multi-model orchestration graph.

    WBS 3.1.2: Create state graph with 4 nodes.
    WBS 3.1.3: Add conditional edges from validate.

    Returns:
        Compiled StateGraph ready for invoke()
    """
    # Create graph with OrchestratorState TypedDict
    graph = StateGraph(OrchestratorState)

    # WBS 3.1.2: Add 4 nodes
    graph.add_node("generate", generate_terms)
    graph.add_node("validate", validate_terms)
    graph.add_node("rank", rank_terms)
    graph.add_node("consensus", build_consensus)

    # Define entry point
    graph.set_entry_point("generate")

    # Define edges
    graph.add_edge("generate", "validate")

    # WBS 3.1.3: Conditional edge from validate
    # Route to 'rank' if terms exist, 'consensus' if all rejected
    graph.add_conditional_edges(
        "validate",
        should_skip_ranking,
        {
            "rank": "rank",
            "consensus": "consensus",
        },
    )

    graph.add_edge("rank", "consensus")
    graph.add_edge("consensus", END)

    return graph.compile()
