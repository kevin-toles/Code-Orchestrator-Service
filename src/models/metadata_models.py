"""
Metadata Extraction Models

WBS: CME-1.1.2 - Create src/models/metadata_models.py with request model
WBS: CME-1.1.5 - Add response model with keywords/concepts/metadata

AC Reference:
- AC-2.2: Keyword Extraction - keywords with term, score, is_technical
- AC-2.3: Concept Extraction - concepts with name, confidence, domain, tier
- AC-2.4: Noise Filtering - rejected.keywords and rejected.reasons
- AC-2.5: Quality Scoring - quality_score between 0.0-1.0
- AC-2.6: Domain Detection - detected_domain and domain_confidence
- AC-2.7: Empty Text Error - text cannot be empty
- AC-2.8: Invalid Options - validation on options fields

TDD Phase: GREEN (implement to pass tests)

Anti-Patterns Avoided:
- S1192: Constants imported from constants.py (WBS-1.1.6)
- AC-6.5: Full type annotations throughout
- Anti-Pattern #7: Proper exception naming (uses Pydantic ValidationError)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.models.constants import (
    DEFAULT_MIN_CONCEPT_CONFIDENCE,
    DEFAULT_MIN_KEYWORD_CONFIDENCE,
    DEFAULT_SUMMARY_RATIO,
    DEFAULT_TOP_K_CONCEPTS,
    DEFAULT_TOP_K_KEYWORDS,
    ERROR_TEXT_EMPTY,
    MAX_CONFIDENCE,
    MAX_SCORE,
    MAX_TOP_K,
    MIN_CONFIDENCE,
    MIN_SCORE,
    MIN_TEXT_LENGTH,
    MIN_TOP_K,
)


# =============================================================================
# Options Model
# =============================================================================


class MetadataExtractionOptions(BaseModel):
    """Options for metadata extraction.
    
    Attributes:
        top_k_keywords: Number of top keywords to return (default: 15).
        top_k_concepts: Number of top concepts to return (default: 10).
        min_keyword_confidence: Minimum confidence for keywords (default: 0.3).
        min_concept_confidence: Minimum confidence for concepts (default: 0.3).
        enable_summary: Whether to generate summary (default: False).
        summary_ratio: Ratio of text for summary (default: 0.2).
        validate_dictionary: Validate against dictionary (default: True).
        filter_noise: Filter noise terms (default: True).
    """

    top_k_keywords: int = Field(
        default=DEFAULT_TOP_K_KEYWORDS,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="Number of top keywords to return",
    )
    top_k_concepts: int = Field(
        default=DEFAULT_TOP_K_CONCEPTS,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="Number of top concepts to return",
    )
    min_keyword_confidence: float = Field(
        default=DEFAULT_MIN_KEYWORD_CONFIDENCE,
        ge=MIN_CONFIDENCE,
        le=MAX_CONFIDENCE,
        description="Minimum confidence threshold for keywords",
    )
    min_concept_confidence: float = Field(
        default=DEFAULT_MIN_CONCEPT_CONFIDENCE,
        ge=MIN_CONFIDENCE,
        le=MAX_CONFIDENCE,
        description="Minimum confidence threshold for concepts",
    )
    enable_summary: bool = Field(
        default=False,
        description="Whether to generate summary",
    )
    summary_ratio: float = Field(
        default=DEFAULT_SUMMARY_RATIO,
        ge=0.0,
        le=1.0,
        description="Ratio of text length for summary",
    )
    validate_dictionary: bool = Field(
        default=True,
        description="Validate keywords against dictionary",
    )
    filter_noise: bool = Field(
        default=True,
        description="Filter noise terms from keywords",
    )
    use_hybrid_extraction: bool = Field(
        default=True,
        description="Use hybrid concept extraction pipeline (YAKE+TextRank+dedup)",
    )


# =============================================================================
# Request Model
# =============================================================================


class MetadataExtractionRequest(BaseModel):
    """Request body for metadata extraction endpoint.
    
    Attributes:
        text: Chapter content to extract metadata from (required, non-empty).
        title: Chapter title (optional).
        book_title: Book title for domain inference (optional).
        options: Extraction options (optional, uses defaults).
    
    AC Reference:
        - AC-2.7: text cannot be empty
    """

    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        description="Chapter content to extract metadata from",
    )
    title: str | None = Field(
        default=None,
        description="Chapter title (optional)",
    )
    book_title: str | None = Field(
        default=None,
        description="Book title for domain inference (optional)",
    )
    options: MetadataExtractionOptions = Field(
        default_factory=MetadataExtractionOptions,
        description="Extraction options",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty or whitespace only.
        
        AC Reference: AC-2.7 - text cannot be empty
        """
        if not v or not v.strip():
            raise ValueError(ERROR_TEXT_EMPTY)
        return v


# =============================================================================
# Response Models
# =============================================================================


class KeywordResult(BaseModel):
    """A single keyword extraction result.
    
    Attributes:
        term: The extracted keyword term.
        score: Confidence score (0.0 to 1.0).
        is_technical: Whether the term is technical/domain-specific.
        sources: List of extraction methods that found this term.
    
    AC Reference:
        - AC-2.2: keywords with term, score, is_technical
    """

    term: str = Field(
        ...,
        description="The extracted keyword term",
    )
    score: float = Field(
        ...,
        ge=MIN_SCORE,
        le=MAX_SCORE,
        description="Confidence score (0.0 to 1.0)",
    )
    is_technical: bool = Field(
        ...,
        description="Whether the term is technical/domain-specific",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of extraction methods that found this term",
    )


class ConceptResult(BaseModel):
    """A single concept extraction result.
    
    Attributes:
        name: The concept name.
        confidence: Confidence score (0.0 to 1.0).
        domain: The domain this concept belongs to.
        tier: The taxonomy tier (T0-T5).
    
    AC Reference:
        - AC-2.3: concepts with name, confidence, domain, tier
    """

    name: str = Field(
        ...,
        description="The concept name",
    )
    confidence: float = Field(
        ...,
        ge=MIN_CONFIDENCE,
        le=MAX_CONFIDENCE,
        description="Confidence score (0.0 to 1.0)",
    )
    domain: str = Field(
        ...,
        description="The domain this concept belongs to",
    )
    tier: str = Field(
        ...,
        description="The taxonomy tier (T0-T5)",
    )


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process.
    
    Attributes:
        processing_time_ms: Time taken in milliseconds.
        text_length: Length of input text.
        detected_domain: Primary detected domain.
        domain_confidence: Confidence in domain detection.
        quality_score: Overall quality score (0.0 to 1.0).
        stages_completed: List of processing stages completed.
    
    AC Reference:
        - AC-2.5: quality_score between 0.0-1.0
        - AC-2.6: detected_domain and domain_confidence
    """

    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    text_length: int = Field(
        ...,
        ge=0,
        description="Length of input text",
    )
    detected_domain: str | None = Field(
        default=None,
        description="Primary detected domain (None if not detected)",
    )
    domain_confidence: float | None = Field(
        default=None,
        ge=MIN_CONFIDENCE,
        le=MAX_CONFIDENCE,
        description="Confidence in domain detection (None if not detected)",
    )
    quality_score: float = Field(
        ...,
        ge=MIN_SCORE,
        le=MAX_SCORE,
        description="Overall quality score (0.0 to 1.0)",
    )
    stages_completed: list[str] = Field(
        default_factory=list,
        description="List of processing stages completed",
    )


class RejectedKeywords(BaseModel):
    """Keywords that were rejected during noise filtering.
    
    Attributes:
        keywords: List of rejected keyword terms.
        reasons: Map of term to rejection reason.
    
    AC Reference:
        - AC-2.4: rejected.keywords and rejected.reasons
    """

    keywords: list[str] = Field(
        default_factory=list,
        description="List of rejected keyword terms",
    )
    reasons: dict[str, str] = Field(
        default_factory=dict,
        description="Map of term to rejection reason",
    )


class MetadataExtractionResponse(BaseModel):
    """Response from metadata extraction endpoint.
    
    Attributes:
        keywords: List of extracted keywords with scores.
        concepts: List of extracted concepts with confidence.
        summary: Optional summary of the text.
        metadata: Extraction process metadata.
        rejected: Keywords rejected during noise filtering.
    """

    keywords: list[KeywordResult] = Field(
        default_factory=list,
        description="List of extracted keywords with scores",
    )
    concepts: list[ConceptResult] = Field(
        default_factory=list,
        description="List of extracted concepts with confidence",
    )
    summary: str | None = Field(
        default=None,
        description="Optional summary of the text",
    )
    metadata: ExtractionMetadata = Field(
        ...,
        description="Extraction process metadata",
    )
    rejected: RejectedKeywords = Field(
        default_factory=RejectedKeywords,
        description="Keywords rejected during noise filtering",
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "MetadataExtractionOptions",
    "MetadataExtractionRequest",
    "KeywordResult",
    "ConceptResult",
    "ExtractionMetadata",
    "RejectedKeywords",
    "MetadataExtractionResponse",
]
