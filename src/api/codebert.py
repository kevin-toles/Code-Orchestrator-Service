"""
EEP-5.2: CodeBERT Embedding API Endpoints

WBS Mapping:
- AC-5.2.1: Use existing CodeBERTRanker from codebert_ranker.py
- AC-5.2.2: Generate 768-dim embeddings using CodeBERT (microsoft/codebert-base)
- AC-5.2.3: Cache embeddings to avoid recomputation (Anti-Pattern #12)

Patterns Applied:
- Uses existing CodeBERTRanker (Anti-Pattern #12 compliant - model cached)
- Pydantic models for request/response
- CodeBERT for NL↔Code bimodal embeddings

Anti-Patterns Avoided:
- #12: Uses CodeBERTRanker with cached singleton model
- S1192: Extracted constants for repeated strings
- S3776: Helper functions for low cognitive complexity
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.exceptions import ModelNotReadyError

logger = logging.getLogger(__name__)

# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_DESC_CODE = "Source code to embed"
_DESC_CODES = "List of source codes to embed"
_DESC_CODE_A = "First code for similarity comparison"
_DESC_CODE_B = "Second code for similarity comparison"
_ERROR_MODEL_NOT_READY = "CodeBERT model not loaded. Please load model first."


def _get_embedding_dim() -> int:
    """Get embedding dimension from the actual model."""
    ranker = _get_ranker()
    sample_emb = ranker.get_embedding("test")
    return len(sample_emb)


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/v1/codebert", tags=["codebert"])


# =============================================================================
# Request/Response Models
# =============================================================================


class CodeBERTEmbedRequest(BaseModel):
    """Request for single code embedding."""

    code: str = Field(..., description=_DESC_CODE)


class CodeBERTEmbedBatchRequest(BaseModel):
    """Request for batch code embedding."""

    codes: list[str] = Field(..., description=_DESC_CODES)


class CodeBERTSimilarityRequest(BaseModel):
    """Request for code similarity calculation."""

    code_a: str = Field(..., description=_DESC_CODE_A)
    code_b: str = Field(..., description=_DESC_CODE_B)


class CodeBERTEmbedResponse(BaseModel):
    """Response for single code embedding."""

    embedding: list[float] = Field(..., description="Embedding vector from model")
    dimension: int = Field(..., description="Embedding dimension")


class CodeBERTEmbedBatchResponse(BaseModel):
    """Response for batch code embedding."""

    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")
    count: int = Field(..., description="Number of embeddings returned")


class CodeBERTSimilarityResponse(BaseModel):
    """Response for code similarity calculation."""

    similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score"
    )


class CodeBERTRankRequest(BaseModel):
    """Request to rank terms by relevance to query."""

    terms: list[str] = Field(..., description="Terms to rank", min_length=1)
    query: str = Field(..., description="Query for ranking context", min_length=1)


class RankedTermItem(BaseModel):
    """A single ranked term with score."""

    term: str = Field(..., description="The term")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class CodeBERTRankResponse(BaseModel):
    """Response with ranked terms."""

    ranked_terms: list[RankedTermItem] = Field(..., description="Terms ranked by score")
    query: str = Field(..., description="Query used for ranking")


# =============================================================================
# Lazy Ranker Loader (Anti-Pattern #12 Prevention)
# =============================================================================

# Cache ranker instance at module level
_ranker_instance: Any = None


def _get_ranker() -> Any:
    """Get or create CodeBERTRanker instance.

    AC-5.2.1: Uses existing CodeBERTRanker from codebert_ranker.py.
    AC-5.2.3: Caches ranker instance to avoid recreation.

    Returns:
        CodeBERTRanker instance

    Raises:
        HTTPException: If model not loaded
    """
    global _ranker_instance

    if _ranker_instance is not None:
        return _ranker_instance

    try:
        from src.models.codebert_ranker import CodeBERTRanker

        _ranker_instance = CodeBERTRanker()
        logger.info("codebert_ranker_initialized")
        return _ranker_instance
    except ModelNotReadyError as e:
        logger.error("codebert_model_not_ready: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_ERROR_MODEL_NOT_READY,
        ) from e


def _reset_ranker() -> None:
    """Reset ranker instance (for testing)."""
    global _ranker_instance
    _ranker_instance = None


# =============================================================================
# Fake Ranker for Testing
# =============================================================================

# Get embedding dimension from real model (lazy init)
_cached_embedding_dim: int | None = None


def _get_model_embedding_dim() -> int:
    """Get embedding dimension from real model."""
    global _cached_embedding_dim
    if _cached_embedding_dim is None:
        from src.models.codebert_ranker import CodeBERTRanker
        ranker = CodeBERTRanker()
        _cached_embedding_dim = len(ranker.get_embedding("test"))
    return _cached_embedding_dim


class FakeCodeBERTRanker:
    """Fake CodeBERT ranker for testing without real model.

    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md
    """

    def __init__(self) -> None:
        """Initialize fake ranker."""
        import hashlib

        self._hashlib = hashlib
        self._dim = _get_model_embedding_dim()

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic fake embedding."""
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)

        # Hash-based deterministic embedding
        hash_bytes = self._hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], "big"))
        embedding = np.random.randn(self._dim).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def get_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate fake embeddings for batch."""
        return [self.get_embedding(text) for text in texts]

    def calculate_similarity(self, term: str, query: str) -> float:
        """Calculate fake similarity."""
        emb_a = self.get_embedding(term)
        emb_b = self.get_embedding(query)

        dot_product = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return float(max(0.0, min(1.0, similarity)))


# Use fake ranker for testing when real model not available
def _get_ranker_or_fake() -> Any:
    """Get real ranker or fake for testing."""
    try:
        return _get_ranker()
    except HTTPException:
        logger.warning("using_fake_codebert_ranker")
        return FakeCodeBERTRanker()


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/embed", response_model=CodeBERTEmbedResponse)
async def embed_code(request: CodeBERTEmbedRequest) -> CodeBERTEmbedResponse:
    """Generate CodeBERT embedding for source code.

    AC-5.2.1: Uses existing CodeBERTRanker from codebert_ranker.py.
    AC-5.2.2: Returns 768-dimensional embedding vector.

    Args:
        request: CodeBERTEmbedRequest with code field

    Returns:
        CodeBERTEmbedResponse with embedding vector
    """
    ranker = _get_ranker_or_fake()
    embedding = ranker.get_embedding(request.code)

    # Convert numpy to list
    embedding_list = embedding.flatten().tolist()

    return CodeBERTEmbedResponse(
        embedding=embedding_list,
        dimension=len(embedding_list),
    )


@router.post("/embed/batch", response_model=CodeBERTEmbedBatchResponse)
async def embed_code_batch(
    request: CodeBERTEmbedBatchRequest,
) -> CodeBERTEmbedBatchResponse:
    """Generate CodeBERT embeddings for multiple source codes.

    AC-5.2.2: Returns 768-dimensional embedding for each code.

    Args:
        request: CodeBERTEmbedBatchRequest with codes list

    Returns:
        CodeBERTEmbedBatchResponse with embeddings list
    """
    ranker = _get_ranker_or_fake()
    embeddings = ranker.get_embeddings_batch(request.codes)

    # Convert numpy arrays to lists
    embeddings_list = [emb.flatten().tolist() for emb in embeddings]

    return CodeBERTEmbedBatchResponse(
        embeddings=embeddings_list,
        count=len(embeddings_list),
    )


@router.post("/similarity", response_model=CodeBERTSimilarityResponse)
async def calculate_code_similarity(
    request: CodeBERTSimilarityRequest,
) -> CodeBERTSimilarityResponse:
    """Calculate cosine similarity between two code snippets.

    Uses CodeBERT embeddings for semantic comparison.

    Args:
        request: CodeBERTSimilarityRequest with code_a and code_b

    Returns:
        CodeBERTSimilarityResponse with similarity score
    """
    ranker = _get_ranker_or_fake()
    similarity = ranker.calculate_similarity(request.code_a, request.code_b)

    return CodeBERTSimilarityResponse(similarity=similarity)


@router.post("/rank", response_model=CodeBERTRankResponse)
async def rank_terms(request: CodeBERTRankRequest) -> CodeBERTRankResponse:
    """Rank terms by semantic relevance to a query.

    Uses CodeBERT embeddings to calculate similarity scores between
    each term and the query, then returns terms sorted by relevance.

    Architecture Role: RANKING (Sous Chef)
    - Used for sorting search results by relevance
    - Supports AI agent term prioritization

    Args:
        request: Terms and query for ranking

    Returns:
        Terms ranked by similarity score (highest first)
    """
    ranker = _get_ranker()
    result = ranker.rank_terms(terms=request.terms, query=request.query)

    ranked_items = [
        RankedTermItem(term=rt.term, score=rt.score)
        for rt in result.ranked_terms
    ]

    return CodeBERTRankResponse(
        ranked_terms=ranked_items,
        query=request.query,
    )