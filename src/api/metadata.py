"""Metadata Extraction API - WBS-1.4.

POST /api/v1/metadata/extract endpoint for unified metadata extraction.

AC Reference:
- AC-2.1: Endpoint registered at POST /api/v1/metadata/extract
- AC-2.2: Keywords with term, score, is_technical
- AC-2.3: Concepts with name, confidence, domain, tier
- AC-2.4: Noise filtering with rejected.keywords and rejected.reasons
- AC-2.5: Quality scoring between 0.0-1.0
- AC-2.6: Domain detection with detected_domain and domain_confidence
- AC-2.7: Empty text returns 400 Bad Request
- AC-2.8: Invalid options returns 422 Validation Error

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Simple endpoint logic (delegated to MetadataExtractor)
- #12: Extractor cached via get_metadata_extractor singleton
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.extractors.metadata_extractor import (
    MetadataExtractor,
    get_metadata_extractor,
)
from src.models.metadata_models import (
    MetadataExtractionOptions,
    MetadataExtractionRequest,
    MetadataExtractionResponse,
    KeywordResult,
    ConceptResult,
    ExtractionMetadata,
    RejectedKeywords,
)


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

API_PREFIX: str = "/v1/metadata"
ENDPOINT_EXTRACT: str = "/extract"
TAG_METADATA: str = "metadata"

ERROR_EMPTY_TEXT: str = "text cannot be empty or whitespace only"


# =============================================================================
# Router
# =============================================================================

metadata_router = APIRouter(
    prefix=API_PREFIX,
    tags=[TAG_METADATA],
)


# =============================================================================
# Endpoints
# =============================================================================


@metadata_router.post(
    ENDPOINT_EXTRACT,
    response_model=MetadataExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract metadata from text",
    description="""
    Extract keywords, concepts, and metadata from text.
    
    ## Features
    - **Keywords**: TF-IDF extracted keywords sorted by score
    - **Concepts**: Taxonomy-matched concepts with domain/tier
    - **Noise Filtering**: Removes OCR watermarks, broken contractions, etc.
    - **Quality Scoring**: 0.0-1.0 score based on extraction quality
    - **Domain Detection**: Infers domain from matched concepts
    
    ## Options
    - `top_k_keywords`: Number of keywords to return (default: 15)
    - `top_k_concepts`: Number of concepts to return (default: 10)
    - `min_keyword_confidence`: Minimum keyword confidence (default: 0.3)
    - `min_concept_confidence`: Minimum concept confidence (default: 0.3)
    - `filter_noise`: Enable noise filtering (default: true)
    """,
    responses={
        200: {"description": "Successful extraction"},
        400: {"description": "Empty or whitespace-only text"},
        422: {"description": "Validation error (invalid options)"},
    },
)
async def extract_metadata(
    request: MetadataExtractionRequest,
) -> MetadataExtractionResponse:
    """Extract metadata from text.

    AC-2.1: POST /api/v1/metadata/extract registered.
    AC-2.7: Empty text returns 422 (Pydantic validation).
    AC-2.8: Invalid options returns 422 (Pydantic).

    Args:
        request: MetadataExtractionRequest with text and options.

    Returns:
        MetadataExtractionResponse with keywords, concepts, metadata.
    """
    # Get cached extractor (Anti-Pattern #12)
    extractor = get_metadata_extractor()

    # Perform extraction
    result = extractor.extract(
        text=request.text,
        title=request.title,
        book_title=request.book_title,
        options=request.options,
    )

    # Build response
    return MetadataExtractionResponse(
        keywords=result.keywords,
        concepts=result.concepts,
        summary=None,  # Summary not implemented yet
        metadata=ExtractionMetadata(
            processing_time_ms=result.processing_time_ms,
            text_length=result.text_length,
            detected_domain=result.detected_domain,
            domain_confidence=result.domain_confidence,
            quality_score=result.quality_score,
            stages_completed=result.stages_completed,
        ),
        rejected=RejectedKeywords(
            keywords=result.rejected_keywords,
            reasons=result.rejection_reasons,
        ),
    )
