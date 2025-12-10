"""
Code-Orchestrator-Service - Pipeline Node Functions

WBS 3.1.2: Individual node functions for LangGraph StateGraph.

Each node:
1. Receives OrchestratorState
2. Performs its operation
3. Returns updated state dict

Patterns Applied:
- Pure functions for testability
- State immutability (return new dict, don't mutate)
- Error capture in state.errors list
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.orchestrator.state import OrchestratorState

logger = get_logger(__name__)


def generate_terms(state: OrchestratorState) -> dict[str, Any]:
    """Generate stage: Extract terms using CodeT5+.

    WBS 3.1.2: Node 1 of 4 in the pipeline.

    Args:
        state: Current orchestrator state

    Returns:
        Dict with generated_terms, related_terms, and updated stages_completed
    """
    logger.info(
        "generate_terms_start",
        query=state["query"],
        domain=state["domain"],
    )

    try:
        # Placeholder: Real implementation will use CodeT5Agent
        # For now, extract simple terms from query
        query = state["query"]
        terms = _extract_simple_terms(query)

        generated = terms[:5] if len(terms) > 5 else terms
        related = terms[5:10] if len(terms) > 5 else []

        stages = [*state["stages_completed"], "generate"]

        logger.info(
            "generate_terms_complete",
            generated_count=len(generated),
            related_count=len(related),
        )

        return {
            "generated_terms": generated,
            "related_terms": related,
            "stages_completed": stages,
        }

    except Exception as e:
        logger.error("generate_terms_error", error=str(e))
        return {
            "errors": [*state["errors"], f"GenerateError: {e}"],
            "stages_completed": [*state["stages_completed"], "generate"],
        }


def validate_terms(state: OrchestratorState) -> dict[str, Any]:
    """Validate stage: Filter generic terms using GraphCodeBERT.

    WBS 3.1.2: Node 2 of 4 in the pipeline.

    Args:
        state: Current orchestrator state

    Returns:
        Dict with validated_terms, rejected_terms, and updated stages_completed
    """
    logger.info(
        "validate_terms_start",
        term_count=len(state["generated_terms"]),
        domain=state["domain"],
    )

    try:
        # Placeholder: Real implementation will use GraphCodeBERTAgent
        # For now, filter common generic terms
        generic_terms = {"split", "data", "the", "a", "an", "is", "are", "get", "set"}

        validated: list[str] = []
        rejected: list[dict[str, str]] = []

        all_terms = state["generated_terms"] + state["related_terms"]
        for term in all_terms:
            if term.lower() in generic_terms:
                rejected.append({"term": term, "reason": "too generic"})
            else:
                validated.append(term)

        stages = [*state["stages_completed"], "validate"]

        logger.info(
            "validate_terms_complete",
            validated_count=len(validated),
            rejected_count=len(rejected),
        )

        return {
            "validated_terms": validated,
            "rejected_terms": rejected,
            "stages_completed": stages,
        }

    except Exception as e:
        logger.error("validate_terms_error", error=str(e))
        return {
            "errors": [*state["errors"], f"ValidateError: {e}"],
            "stages_completed": [*state["stages_completed"], "validate"],
        }


def rank_terms(state: OrchestratorState) -> dict[str, Any]:
    """Rank stage: Score terms using CodeBERT embeddings.

    WBS 3.1.2: Node 3 of 4 in the pipeline.

    Args:
        state: Current orchestrator state

    Returns:
        Dict with ranked_terms and updated stages_completed
    """
    logger.info(
        "rank_terms_start",
        term_count=len(state["validated_terms"]),
    )

    try:
        # Placeholder: Real implementation will use CodeBERTAgent
        # For now, assign mock scores based on term length
        ranked: list[dict[str, Any]] = []

        for i, term in enumerate(state["validated_terms"]):
            # Simple mock score: earlier terms rank higher
            score = 1.0 - (i * 0.1)
            score = max(0.1, score)  # Minimum 0.1
            ranked.append({"term": term, "score": score})

        # Sort by score descending
        ranked.sort(key=lambda x: x["score"], reverse=True)

        stages = [*state["stages_completed"], "rank"]

        logger.info(
            "rank_terms_complete",
            ranked_count=len(ranked),
        )

        return {
            "ranked_terms": ranked,
            "stages_completed": stages,
        }

    except Exception as e:
        logger.error("rank_terms_error", error=str(e))
        return {
            "errors": [*state["errors"], f"RankError: {e}"],
            "stages_completed": [*state["stages_completed"], "rank"],
        }


def build_consensus(state: OrchestratorState) -> dict[str, Any]:
    """Consensus stage: Combine results with model agreement voting.

    WBS 3.1.2: Node 4 of 4 in the pipeline.
    WBS 3.2.1: Terms need ≥2/3 model agreement.

    Args:
        state: Current orchestrator state

    Returns:
        Dict with final_terms, excluded_terms, and updated stages_completed
    """
    logger.info(
        "build_consensus_start",
        ranked_count=len(state["ranked_terms"]),
    )

    try:
        final: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []

        # Placeholder: Real implementation will check model agreement
        # For now, all ranked terms get 3 model agreement (mock)
        for term_data in state["ranked_terms"]:
            final.append({
                "term": term_data["term"],
                "score": term_data["score"],
                "models_agreed": 3,  # Mock: all models agree
            })

        # Add rejected terms to excluded
        for rejected in state["rejected_terms"]:
            excluded.append({
                "term": rejected["term"],
                "reason": rejected["reason"],
            })

        stages = [*state["stages_completed"], "consensus"]

        logger.info(
            "build_consensus_complete",
            final_count=len(final),
            excluded_count=len(excluded),
        )

        return {
            "final_terms": final,
            "excluded_terms": excluded,
            "stages_completed": stages,
        }

    except Exception as e:
        logger.error("build_consensus_error", error=str(e))
        return {
            "errors": [*state["errors"], f"ConsensusError: {e}"],
            "stages_completed": [*state["stages_completed"], "consensus"],
        }


def _extract_simple_terms(query: str) -> list[str]:
    """Extract simple terms from query by tokenization.

    This is a placeholder. Real implementation uses CodeT5+.

    Args:
        query: Input query text

    Returns:
        List of extracted terms
    """
    # Simple word extraction
    words = query.lower().split()

    # Filter short words and common stopwords
    stopwords = {"the", "a", "an", "is", "are", "with", "for", "and", "or", "in", "on"}
    terms = [w for w in words if len(w) > 2 and w not in stopwords]

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return unique_terms
