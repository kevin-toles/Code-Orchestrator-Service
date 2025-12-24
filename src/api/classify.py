"""
WBS-AC6: Classification API Endpoint

POST /api/v1/classify - Classify a single term using Hybrid Tiered Classifier
POST /api/v1/classify/batch - Classify multiple terms

AC Block: AC-6.1 through AC-6.6
- AC-6.1: Endpoint Registration
- AC-6.2: Valid Request returns ClassifyResponse
- AC-6.3: Empty Term returns 422
- AC-6.4: Optional Domain parameter
- AC-6.5: Batch Endpoint
- AC-6.6: Dependency Injection via Depends()

Patterns Applied:
- FastAPI router (CODING_PATTERNS_ANALYSIS.md)
- Pydantic request/response models with full type annotations
- Dependency injection pattern with get_classifier()
- Protocol-based typing for HybridTieredClassifier

Anti-Patterns Avoided:
- S1192: Constants for repeated string literals
- S3776: Simple endpoint logic (CC < 15)
- #12: Classifier dependency injectable for testing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.classifiers.orchestrator import (
    ClassificationResponse,
    FakeHybridTieredClassifier,
    HybridTieredClassifier,
    HybridTieredClassifierProtocol,
)

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

API_PREFIX = "/v1"
CLASSIFY_TAG = "classify"
CLASSIFY_SUMMARY = "Classify a term using Hybrid Tiered Classifier"
CLASSIFY_BATCH_SUMMARY = "Classify multiple terms in batch"

# Error messages
ERROR_TERM_EMPTY = "Term cannot be empty or whitespace"
ERROR_TERM_REQUIRED = "Term is required"

# Field descriptions
DESC_TERM = "The term to classify (concept, keyword, or noise)"
DESC_DOMAIN = "Optional domain context (e.g., 'devops', 'llm_rag')"
DESC_TERMS = "List of terms to classify"


# =============================================================================
# Request/Response Models (Pydantic)
# =============================================================================


class ClassifyRequest(BaseModel):
    """Request body for single term classification.

    AC-6.2: Request schema {"term": "microservice", "domain": "devops"}
    AC-6.3: Term validation (min_length=1)
    """

    term: str = Field(
        ...,
        min_length=1,
        description=DESC_TERM,
        examples=["microservice", "kubernetes", "RAG"],
    )
    domain: str | None = Field(
        default=None,
        description=DESC_DOMAIN,
        examples=["devops", "llm_rag", "ml_ops"],
    )

    @field_validator("term")
    @classmethod
    def validate_term_not_whitespace(cls, v: str) -> str:
        """AC-6.3: Reject whitespace-only terms."""
        if not v.strip():
            raise ValueError(ERROR_TERM_EMPTY)
        return v


class ClassifyResponse(BaseModel):
    """Response body for single term classification.

    AC-6.2: Response includes all ClassificationResponse fields.
    """

    term: str = Field(description="The original term that was classified")
    classification: str = Field(
        description="Classification result: concept, keyword, rejected, or unknown"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)"
    )
    canonical_term: str = Field(description="Normalized/canonical form of the term")
    tier_used: int = Field(
        ge=1, le=4, description="Which tier produced the result (1-4)"
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Reason for rejection if classification='rejected'",
    )

    model_config = {"from_attributes": True}


class ClassifyBatchRequest(BaseModel):
    """Request body for batch term classification.

    AC-6.5: Batch endpoint accepts list of terms.
    """

    terms: list[str] = Field(
        ...,
        description=DESC_TERMS,
        examples=[["microservice", "kubernetes", "RAG"]],
    )
    domain: str | None = Field(
        default=None,
        description=DESC_DOMAIN,
    )


class ClassifyBatchResponse(BaseModel):
    """Response body for batch term classification.

    AC-6.5: Returns list of ClassifyResponse objects.
    """

    results: list[ClassifyResponse] = Field(
        description="Classification results for each term"
    )


# =============================================================================
# Dependency Injection (AC-6.6)
# =============================================================================


def get_classifier() -> HybridTieredClassifierProtocol:
    """Dependency provider for HybridTieredClassifier.

    AC-6.6: Classifier injected via Depends() for easy testing.

    In production, this would create a real HybridTieredClassifier
    with all 4 tier components. For now, returns a FakeHybridTieredClassifier.

    Returns:
        HybridTieredClassifierProtocol instance.
    """
    # TODO: In production, wire up real components:
    # alias_lookup = AliasLookup(...)
    # trained_classifier = TrainedClassifier(...)
    # heuristic_filter = HeuristicFilter(...)
    # llm_fallback = LLMFallback(...)
    # return HybridTieredClassifier(
    #     alias_lookup=alias_lookup,
    #     trained_classifier=trained_classifier,
    #     heuristic_filter=heuristic_filter,
    #     llm_fallback=llm_fallback,
    # )

    # For now, return fake with empty responses (defaults to "unknown")
    return FakeHybridTieredClassifier(responses={})


# =============================================================================
# Router Definition (AC-6.1)
# =============================================================================


classify_router = APIRouter(prefix=API_PREFIX, tags=[CLASSIFY_TAG])


# =============================================================================
# Endpoints
# =============================================================================


@classify_router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary=CLASSIFY_SUMMARY,
    responses={
        200: {"description": "Term classified successfully"},
        422: {"description": "Validation error (empty term)"},
    },
)
async def classify_term(
    request: ClassifyRequest,
    classifier: Annotated[
        HybridTieredClassifierProtocol, Depends(get_classifier)
    ],
) -> ClassifyResponse:
    """Classify a single term using the Hybrid Tiered Classifier.

    AC-6.2: Valid request returns ClassifyResponse.
    AC-6.4: Domain parameter passed to classifier (future use).

    The classifier runs through 4 tiers:
    1. Alias Lookup (O(1), confidence=1.0)
    2. Trained Classifier (SBERT + LogisticRegression)
    3. Heuristic Filter (noise detection)
    4. LLM Fallback (ai-agents call)

    Args:
        request: ClassifyRequest with term and optional domain.
        classifier: Injected HybridTieredClassifier instance.

    Returns:
        ClassifyResponse with classification result.
    """
    # Call the classifier
    result: ClassificationResponse = await classifier.classify(request.term)

    # Convert to response model
    return ClassifyResponse(
        term=result.term,
        classification=result.classification,
        confidence=result.confidence,
        canonical_term=result.canonical_term,
        tier_used=result.tier_used,
        rejection_reason=result.rejection_reason,
    )


@classify_router.post(
    "/classify/batch",
    response_model=ClassifyBatchResponse,
    summary=CLASSIFY_BATCH_SUMMARY,
    responses={
        200: {"description": "Terms classified successfully"},
        422: {"description": "Validation error"},
    },
)
async def classify_batch(
    request: ClassifyBatchRequest,
    classifier: Annotated[
        HybridTieredClassifierProtocol, Depends(get_classifier)
    ],
) -> ClassifyBatchResponse:
    """Classify multiple terms in batch using the Hybrid Tiered Classifier.

    AC-6.5: Batch endpoint processes list of terms.

    Args:
        request: ClassifyBatchRequest with terms list and optional domain.
        classifier: Injected HybridTieredClassifier instance.

    Returns:
        ClassifyBatchResponse with list of classification results.
    """
    # Handle empty list
    if not request.terms:
        return ClassifyBatchResponse(results=[])

    # Call batch classify
    results: list[ClassificationResponse] = await classifier.classify_batch(
        request.terms
    )

    # Convert to response models
    response_results = [
        ClassifyResponse(
            term=r.term,
            classification=r.classification,
            confidence=r.confidence,
            canonical_term=r.canonical_term,
            tier_used=r.tier_used,
            rejection_reason=r.rejection_reason,
        )
        for r in results
    ]

    return ClassifyBatchResponse(results=response_results)
