"""
WBS 3.2: Extract API Endpoint

POST /api/v1/extract - Main term extraction endpoint
- Accepts query and domain
- Returns consensus terms with model agreement
- Tracks processing time and stages

Patterns Applied:
- FastAPI router (Anti-Pattern #9)
- Pydantic request/response models
- Dependency injection for Orchestrator
"""

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.orchestrator.consensus import ConsensusBuilder

# =============================================================================
# Request/Response Models
# =============================================================================


class ExtractOptions(BaseModel):
    """Options for extract endpoint."""

    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_terms: int = Field(default=10, ge=1, le=50)


class ExtractRequest(BaseModel):
    """Request body for extract endpoint."""

    query: str = Field(..., min_length=1, description="Query to extract terms from")
    domain: str = Field(..., min_length=1, description="Domain context for extraction")
    options: ExtractOptions | None = None


class SearchTerm(BaseModel):
    """A search term with consensus information."""

    term: str
    score: float
    models_agreed: int


class ExtractMetadata(BaseModel):
    """Metadata about the extraction process."""

    processing_time_ms: float
    stages_completed: list[str]
    total_terms_processed: int = 0


class ExtractResponse(BaseModel):
    """Response from extract endpoint."""

    search_terms: list[SearchTerm]
    metadata: ExtractMetadata


# =============================================================================
# Router
# =============================================================================

extract_router = APIRouter(prefix="/v1", tags=["extract"])


@extract_router.post("/extract", response_model=ExtractResponse)
async def extract_terms(request: ExtractRequest) -> ExtractResponse:
    """Extract search terms from query using multi-model consensus.

    Args:
        request: ExtractRequest with query, domain, and optional options

    Returns:
        ExtractResponse with consensus terms and metadata
    """
    start_time = time.perf_counter()
    stages_completed: list[str] = []

    try:
        # Stage 1: Generate terms (simulated for now - will integrate with orchestrator)
        generated_terms = _generate_terms(request.query, request.domain)
        stages_completed.append("generate")

        # Stage 2: Validate terms
        validated_terms = _validate_terms(generated_terms, request.domain)
        stages_completed.append("validate")

        # Stage 3: Rank terms (only if we have validated terms)
        if validated_terms:
            ranked_terms = _rank_terms(validated_terms, request.query)
            stages_completed.append("rank")
        else:
            ranked_terms = []

        # Stage 4: Build consensus
        term_data = _build_term_data(generated_terms, validated_terms, ranked_terms)
        builder = ConsensusBuilder()
        consensus_result = builder.build_consensus(term_data)
        stages_completed.append("consensus")

        # Apply options
        options = request.options or ExtractOptions()
        final_terms = consensus_result.final_terms[: options.max_terms]

        # Build response
        search_terms = [
            SearchTerm(
                term=t.term,
                score=t.score,
                models_agreed=t.models_agreed,
            )
            for t in final_terms
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ExtractResponse(
            search_terms=search_terms,
            metadata=ExtractMetadata(
                processing_time_ms=elapsed_ms,
                stages_completed=stages_completed,
                total_terms_processed=consensus_result.total_processed,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Helper Functions (will be replaced with orchestrator integration)
# =============================================================================


def _generate_terms(query: str, _domain: str) -> list[str]:
    """Generate candidate terms from query.

    TODO: Replace with CodeT5+ generator integration.
    Args:
        query: The search query to extract terms from
        _domain: Domain context (unused until model integration)
    """
    # Simple word extraction for now
    words = query.lower().split()
    # Filter common stop words
    stop_words = {"the", "a", "an", "is", "are", "with", "for", "to", "of", "and", "in"}
    return [w for w in words if w not in stop_words and len(w) > 2]


def _validate_terms(terms: list[str], _domain: str) -> list[str]:
    """Validate terms for domain relevance.

    TODO: Replace with GraphCodeBERT validator integration.
    Args:
        terms: List of candidate terms to validate
        _domain: Domain context (unused until model integration)
    """
    # For now, accept all terms
    return terms


def _rank_terms(terms: list[str], _query: str) -> list[dict[str, Any]]:
    """Rank terms by relevance.

    TODO: Replace with CodeBERT ranker integration.
    Args:
        terms: List of validated terms to rank
        _query: Original query for relevance scoring (unused until model integration)
    """
    # Simple ranking based on term length and position
    ranked = []
    for i, term in enumerate(terms):
        # Higher score for earlier terms and longer terms
        position_score = 1.0 - (i / max(len(terms), 1)) * 0.3
        length_score = min(len(term) / 10, 1.0)
        score = (position_score + length_score) / 2

        ranked.append({
            "term": term,
            "score": score,
        })

    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def _build_term_data(
    generated: list[str],
    validated: list[str],
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build term data for consensus builder.

    Combines results from all three stages into the format
    expected by ConsensusBuilder.build_consensus().
    """
    validated_set = set(validated)
    ranked_lookup = {t["term"]: t["score"] for t in ranked}

    result = []
    for term in generated:
        result.append({
            "term": term,
            "generator_score": 0.8,  # Simulated - all generated terms get 0.8
            "validator_approved": term in validated_set,
            "ranker_score": ranked_lookup.get(term, 0.0),
        })

    return result
