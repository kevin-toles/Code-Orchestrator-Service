"""
Keywords API Endpoint

WBS: MSE-1.2 - Keywords API Router
WBS: MSE-1.3 - Request/Response Schemas

POST /api/v1/keywords - Extract keywords from document corpus
POST /api/v1/keywords/scores - Extract keywords with TF-IDF scores (optional)

Role in Kitchen Brigade Architecture:
- Code-Orchestrator-Service (Sous Chef) hosts this endpoint
- ai-agents (Expeditor) calls this endpoint for MSEP keyword enrichment layer
- llm-gateway (Router) may proxy calls to this endpoint

Patterns Applied (per CODING_PATTERNS_ANALYSIS.md):
- FastAPI router pattern (same as src/api/extract.py)
- Pydantic request/response models with validation
- Processing time tracking in metadata

Anti-Patterns Avoided:
- S1192: Constants for duplicated strings
- #2.2: Full type annotations
- #9: FastAPI dependency injection pattern
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.tfidf_extractor import (
    KeywordExtractorConfig,
    TfidfKeywordExtractor,
)

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

API_TAG: str = "keywords"
DEFAULT_TOP_K: int = 10
MIN_TOP_K: int = 0
MAX_TOP_K: int = 100


# =============================================================================
# Request/Response Models (MSE-1.3)
# =============================================================================


class KeywordsRequest(BaseModel):
    """Request body for keywords extraction endpoint.

    Attributes:
        corpus: List of document strings to extract keywords from.
        top_k: Number of top keywords to return per document (default: 10).
    """

    corpus: list[str] = Field(
        ...,
        description="List of document strings to extract keywords from",
        min_length=0,
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="Number of top keywords to return per document",
    )


class KeywordsResponse(BaseModel):
    """Response from keywords extraction endpoint.

    Attributes:
        keywords: List of keyword lists, one per document in corpus.
        processing_time_ms: Time taken to extract keywords in milliseconds.
    """

    keywords: list[list[str]] = Field(
        ...,
        description="List of keyword lists, one per document",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )


class KeywordWithScore(BaseModel):
    """A single keyword with its TF-IDF score.

    Attributes:
        keyword: The extracted keyword or n-gram.
        score: TF-IDF score (0.0 to 1.0).
    """

    keyword: str
    score: float = Field(ge=0.0, le=1.0)


class KeywordsWithScoresResponse(BaseModel):
    """Response from keywords with scores endpoint.

    Attributes:
        keywords: List of keyword-score lists, one per document.
        processing_time_ms: Time taken to extract keywords in milliseconds.
    """

    keywords: list[list[KeywordWithScore]] = Field(
        ...,
        description="List of keyword-score pairs, one list per document",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )


# =============================================================================
# Router (MSE-1.2)
# =============================================================================

keywords_router = APIRouter(prefix="/v1", tags=[API_TAG])

# Singleton extractor instance (Anti-Pattern #12: reuse instead of creating per request)
_extractor: TfidfKeywordExtractor | None = None


def _get_extractor() -> TfidfKeywordExtractor:
    """Get or create the TF-IDF extractor singleton.

    Returns:
        TfidfKeywordExtractor instance.

    Note:
        Uses singleton pattern per Anti-Pattern #12 to avoid
        creating new TfidfVectorizer instances per request.
    """
    global _extractor
    if _extractor is None:
        _extractor = TfidfKeywordExtractor()
    return _extractor


@keywords_router.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(request: KeywordsRequest) -> KeywordsResponse:
    """Extract top-k keywords for each document in the corpus.

    This endpoint uses TF-IDF (Term Frequency-Inverse Document Frequency)
    to identify the most important keywords in each document.

    Args:
        request: KeywordsRequest with corpus and optional top_k.

    Returns:
        KeywordsResponse with keywords list and processing time.

    Example:
        POST /api/v1/keywords
        {
            "corpus": ["Machine learning is great", "Python for data science"],
            "top_k": 5
        }

        Response:
        {
            "keywords": [["machine learning", "learning", "machine"], ["python", "data science"]],
            "processing_time_ms": 12.5
        }
    """
    start_time = time.perf_counter()

    try:
        extractor = _get_extractor()
        keywords = extractor.extract_keywords(request.corpus, top_k=request.top_k)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return KeywordsResponse(
            keywords=keywords,
            processing_time_ms=processing_time_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Keyword extraction failed: {e!s}",
        ) from e


@keywords_router.post("/keywords/scores", response_model=KeywordsWithScoresResponse)
async def extract_keywords_with_scores(
    request: KeywordsRequest,
) -> KeywordsWithScoresResponse:
    """Extract top-k keywords with TF-IDF scores for each document.

    Similar to /keywords but includes the TF-IDF score for each keyword,
    useful for downstream filtering or ranking.

    Args:
        request: KeywordsRequest with corpus and optional top_k.

    Returns:
        KeywordsWithScoresResponse with keyword-score pairs and processing time.

    Example:
        POST /api/v1/keywords/scores
        {
            "corpus": ["Machine learning is great"],
            "top_k": 3
        }

        Response:
        {
            "keywords": [[
                {"keyword": "machine learning", "score": 0.707},
                {"keyword": "learning", "score": 0.500},
                {"keyword": "machine", "score": 0.500}
            ]],
            "processing_time_ms": 15.2
        }
    """
    start_time = time.perf_counter()

    try:
        extractor = _get_extractor()
        results = extractor.extract_keywords_with_scores(
            request.corpus, top_k=request.top_k
        )

        # Convert dataclass results to Pydantic models
        keywords_with_scores: list[list[KeywordWithScore]] = [
            [
                KeywordWithScore(keyword=r.keyword, score=r.score)
                for r in doc_results
            ]
            for doc_results in results
        ]

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return KeywordsWithScoresResponse(
            keywords=keywords_with_scores,
            processing_time_ms=processing_time_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Keyword extraction with scores failed: {e!s}",
        ) from e
