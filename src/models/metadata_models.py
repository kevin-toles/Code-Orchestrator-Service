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
    ERROR_TEXT_EMPTY,
    MAX_CONFIDENCE,
    MAX_SCORE,
    MIN_CONFIDENCE,
    MIN_SCORE,
    MIN_TEXT_LENGTH,
)


# =============================================================================
# Options Model
# =============================================================================


class MetadataExtractionOptions(BaseModel):
    """Options for metadata extraction.
    
    Attributes:
        min_keyword_confidence: Minimum confidence for keywords (default: 0.3).
        min_concept_confidence: Minimum confidence for concepts (default: 0.3).
        enable_summary: Whether to generate summary (default: False).
        summary_ratio: Ratio of text for summary (default: 0.2).
        validate_dictionary: Validate against dictionary (default: True).
        filter_noise: Filter noise terms (default: True).
        
    NOTE: top_k_keywords and top_k_concepts REMOVED per user requirement.
          Extract ALL keywords/concepts, filter with confirmed lists, then dedupe.
    """

    # NOTE: top_k_keywords and top_k_concepts REMOVED - extract ALL, filter downstream
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
    summary_model: str | None = Field(
        default=None,
        description="Model used for summary generation (e.g., 'phi-4', 'statistical-fallback')",
    )
    summary_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens used for summary generation (0 if statistical)",
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
# Batch Models
# =============================================================================


class BatchTextItem(BaseModel):
    """A single text item in a batch request.
    
    Attributes:
        id: Unique identifier for this item (e.g., chapter_id).
        text: The text content to extract metadata from.
        title: Optional title for the text.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this item",
    )
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        description="Text content to extract metadata from",
    )
    title: str | None = Field(
        default=None,
        description="Optional title for the text",
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError(ERROR_TEXT_EMPTY)
        return v


class BatchExtractionRequest(BaseModel):
    """Request for batch metadata extraction.
    
    Attributes:
        items: List of text items to process.
        book_title: Book title for domain inference (applies to all items).
        options: Extraction options (applies to all items).
    """

    items: list[BatchTextItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text items to process (max 100)",
    )
    book_title: str | None = Field(
        default=None,
        description="Book title for domain inference",
    )
    options: MetadataExtractionOptions = Field(
        default_factory=MetadataExtractionOptions,
        description="Extraction options (applies to all items)",
    )


class BatchItemResult(BaseModel):
    """Result for a single item in batch processing.
    
    Attributes:
        id: The item ID from the request.
        success: Whether extraction succeeded.
        result: The extraction result (if successful).
        error: Error message (if failed).
    """

    id: str = Field(
        ...,
        description="The item ID from the request",
    )
    success: bool = Field(
        ...,
        description="Whether extraction succeeded",
    )
    result: MetadataExtractionResponse | None = Field(
        default=None,
        description="The extraction result (if successful)",
    )
    error: str | None = Field(
        default=None,
        description="Error message (if failed)",
    )


class BatchExtractionResponse(BaseModel):
    """Response from batch metadata extraction.
    
    Attributes:
        results: List of results for each item.
        total_items: Total number of items processed.
        successful: Number of successful extractions.
        failed: Number of failed extractions.
        total_processing_time_ms: Total processing time in milliseconds.
    """

    results: list[BatchItemResult] = Field(
        ...,
        description="List of results for each item",
    )
    total_items: int = Field(
        ...,
        description="Total number of items processed",
    )
    successful: int = Field(
        ...,
        description="Number of successful extractions",
    )
    failed: int = Field(
        ...,
        description="Number of failed extractions",
    )
    total_processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
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
    # Batch Models
    "BatchTextItem",
    "BatchExtractionRequest",
    "BatchItemResult",
    "BatchExtractionResponse",
]
