"""
Code-Orchestrator-Service - GraphCodeBERT API Endpoints

Exposes GraphCodeBERT (microsoft/graphcodebert-base) capabilities:
- POST /v1/graphcodebert/validate - Validate terms against query
- POST /v1/graphcodebert/classify-domain - Classify text domain
- POST /v1/graphcodebert/expand - Expand terms with related terms
- POST /v1/graphcodebert/embeddings - Get term embeddings
- POST /v1/graphcodebert/similarity - Batch similarity scoring
- GET /v1/graphcodebert/health - Model health check

Architecture Role: VALIDATOR (Sous Chef)
Reference: KITCHEN_BRIGADE_ARCHITECTURE.md, ARCHITECTURE.md

Model Capabilities:
- Understands code structure via data flow graphs
- Catches semantic mismatches
- 768-dim embeddings for code understanding
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.logging import get_logger
from src.models.graphcodebert_validator import GraphCodeBERTValidator

logger = get_logger(__name__)

# Router with /v1/graphcodebert prefix
router = APIRouter(prefix="/v1/graphcodebert", tags=["graphcodebert"])

# Singleton validator instance
_validator: GraphCodeBERTValidator | None = None

# Constants (S1192 compliance)
_MODEL_NAME = "microsoft/graphcodebert-base"
_EMBEDDING_DIM = 768


def _get_validator() -> GraphCodeBERTValidator:
    """Get or create singleton GraphCodeBERT validator."""
    global _validator
    if _validator is None:
        _validator = GraphCodeBERTValidator()
    return _validator


# ============================================================================
# Request/Response Models
# ============================================================================


class ValidateRequest(BaseModel):
    """Request to validate terms against a query."""

    terms: list[str] = Field(..., description="Terms to validate", min_length=1)
    query: str = Field(..., description="Original query for semantic context", min_length=1)
    domain: str = Field("general", description="Target domain (ai-ml, systems, web, data)")
    min_similarity: float = Field(0.15, ge=0.0, le=1.0, description="Minimum similarity threshold")


class ValidateResponse(BaseModel):
    """Response with validated terms."""

    valid_terms: list[str] = Field(..., description="Terms that passed validation")
    rejected_terms: list[str] = Field(..., description="Terms that were rejected")
    rejection_reasons: dict[str, str] = Field(..., description="Reason for each rejection")
    similarity_scores: dict[str, float] = Field(..., description="Similarity scores for valid terms")


class ClassifyDomainRequest(BaseModel):
    """Request to classify text domain."""

    text: str = Field(..., description="Text to classify", min_length=1)


class ClassifyDomainResponse(BaseModel):
    """Response with domain classification."""

    domain: str = Field(..., description="Classified domain (ai-ml, systems, web, data, general)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")


class ExpandRequest(BaseModel):
    """Request to expand terms with related terms."""

    terms: list[str] = Field(..., description="Terms to expand", min_length=1)
    domain: str = Field("general", description="Target domain for context")
    max_expansions: int = Field(3, ge=1, le=10, description="Max related terms per input")
    candidates: list[str] | None = Field(None, description="Optional candidate terms to search")


class ExpandResponse(BaseModel):
    """Response with expanded terms."""

    original_terms: list[str] = Field(..., description="Original input terms")
    expanded_terms: list[str] = Field(..., description="All terms including expansions")
    expansion_count: int = Field(..., description="Number of new terms added")


class EmbeddingsRequest(BaseModel):
    """Request to get term embeddings."""

    terms: list[str] = Field(..., description="Terms to embed", min_length=1)


class EmbeddingsResponse(BaseModel):
    """Response with term embeddings."""

    embeddings: dict[str, list[float]] = Field(..., description="Term -> embedding vector")
    embedding_dim: int = Field(..., description="Embedding dimension")
    model: str = Field(_MODEL_NAME, description="Model used")


class SimilarityRequest(BaseModel):
    """Request for batch similarity scoring."""

    terms: list[str] = Field(..., description="Terms to score", min_length=1)
    query: str = Field(..., description="Query text for comparison", min_length=1)


class SimilarityResponse(BaseModel):
    """Response with similarity scores."""

    scores: dict[str, float] = Field(..., description="Term -> similarity score")
    query: str = Field(..., description="Query used")
    model: str = Field(_MODEL_NAME, description="Model used")


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/validate", response_model=ValidateResponse)
async def validate_terms(request: ValidateRequest) -> ValidateResponse:
    """Validate terms against a query using GraphCodeBERT embeddings.

    Filters terms based on semantic relevance to the query. Terms below
    the similarity threshold are rejected.

    Architecture Role: STATE 2 VALIDATION
    - Filters generic terms (fast, no model inference)
    - Computes similarity using 768-dim embeddings
    - Returns validation result with scores

    Args:
        request: Terms and query to validate against

    Returns:
        Validated and rejected terms with reasons and scores
    """
    logger.info("graphcodebert_validate_request", term_count=len(request.terms))

    try:
        validator = _get_validator()
        result = validator.validate_terms(
            terms=request.terms,
            original_query=request.query,
            domain=request.domain,
            min_similarity=request.min_similarity,
        )

        return ValidateResponse(
            valid_terms=result.valid_terms,
            rejected_terms=result.rejected_terms,
            rejection_reasons=result.rejection_reasons,
            similarity_scores=result.similarity_scores,
        )

    except Exception as e:
        logger.error("graphcodebert_validate_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {e}",
        )


@router.post("/classify-domain", response_model=ClassifyDomainResponse)
async def classify_domain(request: ClassifyDomainRequest) -> ClassifyDomainResponse:
    """Classify the domain of text using GraphCodeBERT.

    Uses cosine similarity between text embedding and pre-defined domain
    reference embeddings. Returns the domain with highest similarity.

    Domains: ai-ml, systems, web, data, general

    Args:
        request: Text to classify

    Returns:
        Classified domain with confidence score
    """
    logger.info("graphcodebert_classify_request", text_length=len(request.text))

    try:
        validator = _get_validator()
        domain = validator.classify_domain(request.text)

        # Get confidence by re-computing similarity (could optimize)
        text_embedding = validator._get_embedding(request.text)
        if domain != "general" and domain in validator._domain_embeddings:
            confidence = validator._cosine_similarity(
                text_embedding, validator._domain_embeddings[domain]
            )
        else:
            confidence = 0.3 if domain == "general" else 0.5

        return ClassifyDomainResponse(
            domain=domain,
            confidence=round(confidence, 4),
        )

    except Exception as e:
        logger.error("graphcodebert_classify_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {e}",
        )


@router.post("/expand", response_model=ExpandResponse)
async def expand_terms(request: ExpandRequest) -> ExpandResponse:
    """Expand terms with semantically related terms.

    Finds nearest neighbors from the candidate list using GraphCodeBERT
    embeddings. Without candidates, returns original terms unchanged.

    Args:
        request: Terms to expand with optional candidates

    Returns:
        Original and expanded terms
    """
    logger.info(
        "graphcodebert_expand_request",
        term_count=len(request.terms),
        has_candidates=request.candidates is not None,
    )

    try:
        validator = _get_validator()
        expanded = validator.expand_terms(
            terms=request.terms,
            domain=request.domain,
            max_expansions=request.max_expansions,
            expansion_candidates=request.candidates,
        )

        return ExpandResponse(
            original_terms=request.terms,
            expanded_terms=expanded,
            expansion_count=len(expanded) - len(request.terms),
        )

    except Exception as e:
        logger.error("graphcodebert_expand_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Expansion failed: {e}",
        )


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def get_embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    """Get GraphCodeBERT embeddings for terms.

    Returns 768-dimensional embeddings for each term, useful for
    downstream tasks like clustering or visualization.

    Args:
        request: Terms to embed

    Returns:
        Dictionary mapping term -> embedding vector
    """
    logger.info("graphcodebert_embeddings_request", term_count=len(request.terms))

    try:
        validator = _get_validator()
        embeddings = validator.get_term_embeddings(request.terms)

        # Convert numpy arrays to lists for JSON serialization
        embeddings_dict = {
            term: emb.tolist() for term, emb in embeddings.items()
        }

        return EmbeddingsResponse(
            embeddings=embeddings_dict,
            embedding_dim=_EMBEDDING_DIM,
            model=_MODEL_NAME,
        )

    except Exception as e:
        logger.error("graphcodebert_embeddings_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {e}",
        )


@router.post("/similarity", response_model=SimilarityResponse)
async def batch_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """Calculate similarity scores for multiple terms against a query.

    More efficient than validate_terms when you just need scores without
    filtering or rejection reasons.

    Args:
        request: Terms and query to compare

    Returns:
        Dictionary mapping term -> similarity score
    """
    logger.info("graphcodebert_similarity_request", term_count=len(request.terms))

    try:
        validator = _get_validator()
        scores = validator.batch_similarity(
            terms=request.terms,
            query=request.query,
        )

        # Round scores for cleaner output
        scores_rounded = {term: round(score, 4) for term, score in scores.items()}

        return SimilarityResponse(
            scores=scores_rounded,
            query=request.query,
            model=_MODEL_NAME,
        )

    except Exception as e:
        logger.error("graphcodebert_similarity_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity calculation failed: {e}",
        )


@router.get("/health")
async def graphcodebert_health() -> dict[str, Any]:
    """Check GraphCodeBERT model health and status."""
    try:
        validator = _get_validator()
        # Do a quick test embedding
        _ = validator._get_embedding("test")

        return {
            "status": "healthy",
            "model": _MODEL_NAME,
            "embedding_dim": _EMBEDDING_DIM,
            "loaded": True,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model": _MODEL_NAME,
            "error": str(e),
        }
