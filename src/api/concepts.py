"""
EEP-2.4: Concept Extraction API Endpoint

POST /api/v1/concepts - Extract domain concepts from text or keywords
AC-2.4.1: Add POST /api/v1/concepts endpoint

Request accepts:
- text: Raw text to extract concepts from
- keywords: EEP-1 filtered keywords (alternative to text)
- domain: Optional domain filter

Response returns:
- concepts: List of ExtractedConcept with confidence scores
- domain_scores: Domain confidence breakdown
- primary_domain: Dominant domain classification

Patterns Applied:
- FastAPI router (Anti-Pattern #9 prevention)
- Pydantic request/response models with full type annotations
- Dependency injection pattern

Anti-Patterns Avoided:
- S1192: Constants for repeated string literals
- S3776: Simple endpoint logic (delegated to ConceptExtractor)
- #12: Taxonomy cached in ConceptExtractor, not loaded per request
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.models.concept_extractor import (
    ConceptExtractor,
    ConceptExtractorConfig,
)

# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

DEFAULT_DOMAIN_TAXONOMY_PATH = Path(
    "/app/config/semantic-search/domain_taxonomy.json"
)
DEFAULT_TIER_TAXONOMY_PATH = Path(
    "/app/config/taxonomies/AI-ML_taxonomy_20251128.json"
)

# Error messages
ERROR_NO_INPUT = "Either 'text' or 'keywords' must be provided"
ERROR_TAXONOMY_NOT_FOUND = "Domain taxonomy file not found"


# =============================================================================
# Request/Response Models (Pydantic)
# =============================================================================


class ConceptExtractionOptions(BaseModel):
    """Options for concept extraction."""

    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for concepts",
    )
    enable_hierarchical: bool = Field(
        default=False,
        description="Enable hierarchical concept relationships",
    )
    domain_filter: str | None = Field(
        default=None,
        description="Filter concepts to specific domain",
    )


class ConceptExtractionRequest(BaseModel):
    """Request body for concept extraction endpoint.

    AC-2.4.2: Request schema {"text": "...", "domain": "llm_rag"}
    """

    text: str | None = Field(
        default=None,
        min_length=1,
        description="Text to extract concepts from",
    )
    keywords: list[str] | None = Field(
        default=None,
        description="EEP-1 filtered keywords (alternative to text)",
    )
    domain: str | None = Field(
        default=None,
        description="Domain context for extraction (AC-2.4.2)",
    )
    options: ConceptExtractionOptions | None = None


class ConceptResponse(BaseModel):
    """A single extracted concept in the response."""

    name: str
    confidence: float
    domain: str
    tier: str
    parent_concept: str | None = None


class ConceptExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""

    processing_time_ms: float
    total_concepts: int
    domains_matched: list[str]
    taxonomy_source: str


class ConceptExtractionResponse(BaseModel):
    """Response from concept extraction endpoint.

    AC-2.4.3: Response schema {"concepts": [...], "domain_score": 0.85}
    """

    concepts: list[ConceptResponse]
    domain_score: float = Field(
        description="Primary domain confidence score (AC-2.4.3)",
    )
    domain_scores: dict[str, float] = Field(
        description="All domain confidence scores (extended)",
    )
    primary_domain: str | None
    metadata: ConceptExtractionMetadata


# =============================================================================
# Module-level extractor (Anti-Pattern #12 - cached, not per-request)
# =============================================================================

_extractor: ConceptExtractor | None = None


def _get_extractor() -> ConceptExtractor:
    """Get or create cached ConceptExtractor instance.

    Anti-Pattern #12: Taxonomy cached at module level, not per request.

    Returns:
        ConceptExtractor instance

    Raises:
        HTTPException: If taxonomy file not found
    """
    global _extractor

    if _extractor is None:
        # Check for taxonomy file
        if not DEFAULT_DOMAIN_TAXONOMY_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=ERROR_TAXONOMY_NOT_FOUND,
            )

        config = ConceptExtractorConfig(
            domain_taxonomy_path=DEFAULT_DOMAIN_TAXONOMY_PATH,
            tier_taxonomy_path=DEFAULT_TIER_TAXONOMY_PATH if DEFAULT_TIER_TAXONOMY_PATH.exists() else None,
        )
        _extractor = ConceptExtractor(config)

    return _extractor


# =============================================================================
# Router
# =============================================================================

concepts_router = APIRouter(prefix="/v1", tags=["concepts"])


@concepts_router.post("/concepts", response_model=ConceptExtractionResponse)
async def extract_concepts(request: ConceptExtractionRequest) -> ConceptExtractionResponse:
    """Extract domain concepts from text or keywords.

    AC-2.4.1: POST /api/v1/concepts endpoint

    Args:
        request: ConceptExtractionRequest with text or keywords

    Returns:
        ConceptExtractionResponse with concepts and domain scores

    Raises:
        HTTPException: If neither text nor keywords provided
    """
    start_time = time.perf_counter()

    # Validate input
    if not request.text and not request.keywords:
        raise HTTPException(status_code=400, detail=ERROR_NO_INPUT)

    # Get options
    options = request.options or ConceptExtractionOptions()

    # Get extractor
    extractor = _get_extractor()

    # Update extractor config if needed
    if options.enable_hierarchical:
        extractor._config.enable_hierarchical = True
    if options.min_confidence != extractor._config.min_confidence:
        extractor._config.min_confidence = options.min_confidence

    # Extract concepts
    if request.text:
        result = extractor.extract_concepts(request.text)
    else:
        result = extractor.extract_concepts_from_keywords(request.keywords or [])

    # Apply domain filter - check request.domain first (AC-2.4.2), then options
    domain_filter = request.domain or options.domain_filter
    concepts = result.concepts
    if domain_filter:
        concepts = [c for c in concepts if c.domain == domain_filter]

    # Calculate processing time
    processing_time_ms = (time.perf_counter() - start_time) * 1000

    # Calculate primary domain score (AC-2.4.3)
    primary_domain_score = 0.0
    if result.primary_domain and result.primary_domain in result.domain_scores:
        primary_domain_score = result.domain_scores[result.primary_domain]

    # Build response
    return ConceptExtractionResponse(
        concepts=[
            ConceptResponse(
                name=c.name,
                confidence=c.confidence,
                domain=c.domain,
                tier=c.tier,
                parent_concept=c.parent_concept,
            )
            for c in concepts
        ],
        domain_score=primary_domain_score,
        domain_scores=result.domain_scores,
        primary_domain=result.primary_domain,
        metadata=ConceptExtractionMetadata(
            processing_time_ms=processing_time_ms,
            total_concepts=len(concepts),
            domains_matched=list(result.domain_scores.keys()),
            taxonomy_source=str(DEFAULT_DOMAIN_TAXONOMY_PATH),
        ),
    )


@concepts_router.get("/concepts/domains", response_model=dict[str, list[str]])
async def get_domain_concepts() -> dict[str, list[str]]:
    """Get all concepts organized by domain.

    Returns:
        Dict mapping domain names to concept lists
    """
    extractor = _get_extractor()

    domains_concepts: dict[str, list[str]] = {}
    for domain_name in extractor.domains:
        domains_concepts[domain_name] = extractor.get_domain_concepts(domain_name)

    return domains_concepts


@concepts_router.get("/concepts/domains/{domain}", response_model=list[str])
async def get_concepts_for_domain(domain: str) -> list[str]:
    """Get concepts for a specific domain.

    Args:
        domain: Domain name

    Returns:
        List of concepts for the domain

    Raises:
        HTTPException: If domain not found
    """
    extractor = _get_extractor()

    if domain not in extractor.domains:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found in taxonomy",
        )

    return extractor.get_domain_concepts(domain)
