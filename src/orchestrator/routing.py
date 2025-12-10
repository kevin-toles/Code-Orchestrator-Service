"""
Code-Orchestrator-Service - Conditional Routing Functions

WBS 3.1.3: Add conditional edges - Route based on validation results.

Patterns Applied:
- Pure functions for routing decisions
- State inspection without mutation
"""

from typing import Literal

from src.orchestrator.state import OrchestratorState


def should_skip_ranking(state: OrchestratorState) -> Literal["rank", "consensus"]:
    """Determine whether to proceed to rank or skip to consensus.

    WBS 3.1.3: Route based on validation results.

    If all terms were rejected during validation, skip ranking and go
    directly to consensus (which will report no valid terms).

    Args:
        state: Current orchestrator state after validate stage

    Returns:
        "rank" if there are validated terms to rank
        "consensus" if all terms were rejected (skip ranking)
    """
    validated_terms = state.get("validated_terms", [])

    if validated_terms:
        return "rank"
    return "consensus"


def should_retry(state: OrchestratorState, max_retries: int = 3) -> bool:
    """Determine whether to retry the pipeline on error.

    WBS 3.1.4: Max 3 retries on failure.

    Args:
        state: Current orchestrator state
        max_retries: Maximum allowed retry attempts

    Returns:
        True if retry should be attempted, False otherwise
    """
    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)

    # Only retry on transient errors
    if not errors:
        return False

    last_error = errors[-1] if errors else ""

    # Check if error is retryable (transient)
    transient_patterns = [
        "timeout",
        "network",
        "connection",
        "temporary",
        "transient",
    ]

    is_transient = any(
        pattern in last_error.lower() for pattern in transient_patterns
    )

    return is_transient and retry_count < max_retries
