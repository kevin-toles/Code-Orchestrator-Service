"""
Code-Orchestrator-Service - Orchestrator State Definitions

WBS 3.1.1: Define OrchestratorState - Pydantic/TypedDict model with all fields.

Patterns Applied:
- TypedDict for LangGraph state (ai-agents CODE_UNDERSTANDING_ORCHESTRATOR_DESIGN.md)
- Pydantic BaseModel for configuration (CODING_PATTERNS_ANALYSIS.md)
- Dataclass for result models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from pydantic import BaseModel, Field


class OrchestratorState(TypedDict):
    """State shared across all pipeline stages.

    WBS 3.1.1: Pydantic model with all fields for LangGraph StateGraph.
    Pattern: TypedDict for LangGraph state per ai-agents design doc.

    Stages:
    1. generate: CodeT5+ extracts terms from query
    2. validate: GraphCodeBERT filters generic terms
    3. rank: CodeBERT ranks by relevance
    4. consensus: Combine results with voting
    """

    # Input fields
    query: str
    domain: str

    # Generator stage output (CodeT5+)
    generated_terms: list[str]
    related_terms: list[str]

    # Validator stage output (GraphCodeBERT)
    validated_terms: list[str]
    rejected_terms: list[dict[str, str]]  # {"term": str, "reason": str}

    # Ranker stage output (CodeBERT)
    ranked_terms: list[dict[str, float]]  # {"term": str, "score": float}

    # Consensus stage output
    final_terms: list[dict[str, float | int]]  # {"term", "score", "models_agreed"}
    excluded_terms: list[dict[str, str]]  # {"term": str, "reason": str}

    # Metadata fields
    stages_completed: list[str]
    errors: list[str]
    retry_count: int


class OrchestratorOptions(BaseModel):
    """Configuration options for orchestrator pipeline.

    WBS 3.1.4: Includes max_retries for retry logic.
    Pattern: Pydantic BaseModel for configuration validation.
    """

    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for consensus terms",
    )
    max_terms: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of terms to return",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient failures (WBS 3.1.4)",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Timeout for individual stage operations",
    )


@dataclass
class SearchTerm:
    """A single search term with consensus data.

    Used in OrchestratorResult.search_terms list.
    """

    term: str
    score: float
    models_agreed: int


@dataclass
class OrchestratorResult:
    """Result returned by Orchestrator.run() method.

    WBS 3.1: Contains stages_completed and search_terms for verification.
    """

    stages_completed: list[str] = field(default_factory=list)
    search_terms: list[SearchTerm] = field(default_factory=list)
    excluded_terms: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


def create_initial_state(query: str, domain: str) -> OrchestratorState:
    """Create initial state for orchestrator pipeline.

    Args:
        query: Search query text
        domain: Domain context (e.g., "ai-ml", "systems")

    Returns:
        Initialized OrchestratorState with empty lists
    """
    return OrchestratorState(
        query=query,
        domain=domain,
        generated_terms=[],
        related_terms=[],
        validated_terms=[],
        rejected_terms=[],
        ranked_terms=[],
        final_terms=[],
        excluded_terms=[],
        stages_completed=[],
        errors=[],
        retry_count=0,
    )
