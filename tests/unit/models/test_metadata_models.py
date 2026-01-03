"""
Unit tests for Metadata Extraction Models

WBS: CME-1.1.1 - Write failing test: MetadataExtractionRequest validation
WBS: CME-1.1.4 - Write failing test: MetadataExtractionResponse schema

AC Reference:
- AC-2.7: Empty Text Error - POST with {"text": ""} returns 400
- AC-2.8: Invalid Options - POST with {"options": {"top_k_keywords": -1}} returns 422
- AC-2.2: Keyword Extraction - Response includes keywords sorted by score
- AC-2.3: Concept Extraction - Response includes concepts with domain/tier

TDD Phase: RED (tests written first, expected to fail)

Anti-Patterns Avoided:
- S1192: Constants for test strings
- AC-6.5: Full type annotations
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

VALID_TEXT: str = "Microservices architecture enables scalable distributed systems."
EMPTY_TEXT: str = ""
VALID_TITLE: str = "Chapter 1: Introduction"
VALID_BOOK_TITLE: str = "Building Microservices"

# Options constants
DEFAULT_TOP_K_KEYWORDS: int = 15
DEFAULT_TOP_K_CONCEPTS: int = 10
DEFAULT_MIN_KEYWORD_CONFIDENCE: float = 0.3
DEFAULT_MIN_CONCEPT_CONFIDENCE: float = 0.3
INVALID_TOP_K: int = -1


# =============================================================================
# WBS-1.1.1: MetadataExtractionRequest Tests (RED)
# =============================================================================


class TestMetadataExtractionRequest:
    """Tests for MetadataExtractionRequest Pydantic model."""

    def test_valid_request_with_text_only(self) -> None:
        """Request with only required text field should be valid.
        
        AC Reference: AC-2.7 (validates text is required)
        """
        from src.models.metadata_models import MetadataExtractionRequest

        request = MetadataExtractionRequest(text=VALID_TEXT)
        
        assert request.text == VALID_TEXT
        assert request.title is None
        assert request.book_title is None
        assert request.options is not None  # Should have defaults

    def test_valid_request_with_all_fields(self) -> None:
        """Request with all fields should be valid."""
        from src.models.metadata_models import (
            MetadataExtractionOptions,
            MetadataExtractionRequest,
        )

        options = MetadataExtractionOptions(
            filter_noise=True,
        )
        request = MetadataExtractionRequest(
            text=VALID_TEXT,
            title=VALID_TITLE,
            book_title=VALID_BOOK_TITLE,
            options=options,
        )
        
        assert request.text == VALID_TEXT
        assert request.title == VALID_TITLE
        assert request.book_title == VALID_BOOK_TITLE
        assert request.options.filter_noise is True

    def test_empty_text_raises_validation_error(self) -> None:
        """Empty text should raise ValidationError.
        
        AC Reference: AC-2.7 - Empty Text Error
        """
        from src.models.metadata_models import MetadataExtractionRequest

        with pytest.raises(ValidationError) as exc_info:
            MetadataExtractionRequest(text=EMPTY_TEXT)
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        # Check that error is about text being empty
        error_fields = [e["loc"][0] for e in errors]
        assert "text" in error_fields

    def test_missing_text_raises_validation_error(self) -> None:
        """Missing text field should raise ValidationError.
        
        AC Reference: AC-2.7 - text is required
        """
        from src.models.metadata_models import MetadataExtractionRequest

        with pytest.raises(ValidationError) as exc_info:
            MetadataExtractionRequest()  # type: ignore[call-arg]
        
        errors = exc_info.value.errors()
        assert len(errors) >= 1


class TestMetadataExtractionOptions:
    """Tests for MetadataExtractionOptions Pydantic model."""

    def test_default_options(self) -> None:
        """Options should have sensible defaults."""
        from src.models.metadata_models import MetadataExtractionOptions

        options = MetadataExtractionOptions()
        
        # top_k_keywords and top_k_concepts were REMOVED per requirement
        assert options.min_keyword_confidence == DEFAULT_MIN_KEYWORD_CONFIDENCE
        assert options.min_concept_confidence == DEFAULT_MIN_CONCEPT_CONFIDENCE
        assert options.enable_summary is False
        assert options.filter_noise is True

    def test_invalid_confidence_raises_validation_error(self) -> None:
        """Confidence outside 0-1 range should raise ValidationError."""
        from src.models.metadata_models import MetadataExtractionOptions

        with pytest.raises(ValidationError):
            MetadataExtractionOptions(min_keyword_confidence=1.5)
        
        with pytest.raises(ValidationError):
            MetadataExtractionOptions(min_concept_confidence=-0.1)


# =============================================================================
# WBS-1.1.4: MetadataExtractionResponse Tests (RED)
# =============================================================================


class TestKeywordResult:
    """Tests for KeywordResult model."""

    def test_keyword_result_creation(self) -> None:
        """KeywordResult should have term, score, is_technical, sources.
        
        AC Reference: AC-2.2 - keywords with term, score, is_technical
        """
        from src.models.metadata_models import KeywordResult

        keyword = KeywordResult(
            term="microservices",
            score=0.85,
            is_technical=True,
            sources=["tfidf", "domain_taxonomy"],
        )
        
        assert keyword.term == "microservices"
        assert keyword.score == 0.85
        assert keyword.is_technical is True
        assert "tfidf" in keyword.sources

    def test_keyword_score_range(self) -> None:
        """Score must be between 0.0 and 1.0."""
        from src.models.metadata_models import KeywordResult

        with pytest.raises(ValidationError):
            KeywordResult(
                term="invalid",
                score=1.5,  # Invalid: > 1.0
                is_technical=False,
                sources=[],
            )


class TestConceptResult:
    """Tests for ConceptResult model."""

    def test_concept_result_creation(self) -> None:
        """ConceptResult should have name, confidence, domain, tier.
        
        AC Reference: AC-2.3 - concepts with name, confidence, domain, tier
        """
        from src.models.metadata_models import ConceptResult

        concept = ConceptResult(
            name="distributed systems",
            confidence=0.8,
            domain="architecture",
            tier="T2",
        )
        
        assert concept.name == "distributed systems"
        assert concept.confidence == 0.8
        assert concept.domain == "architecture"
        assert concept.tier == "T2"


class TestExtractionMetadata:
    """Tests for ExtractionMetadata model."""

    def test_metadata_creation(self) -> None:
        """ExtractionMetadata should have processing_time, quality_score, etc.
        
        AC Reference: AC-2.5 - quality_score between 0.0-1.0
        AC Reference: AC-2.6 - detected_domain and domain_confidence
        """
        from src.models.metadata_models import ExtractionMetadata

        metadata = ExtractionMetadata(
            processing_time_ms=123.45,
            text_length=5000,
            detected_domain="architecture",
            domain_confidence=0.75,
            quality_score=0.82,
            stages_completed=["keywords", "concepts", "validation"],
        )
        
        assert metadata.processing_time_ms == 123.45
        assert metadata.text_length == 5000
        assert metadata.detected_domain == "architecture"
        assert metadata.domain_confidence == 0.75
        assert metadata.quality_score == 0.82
        assert "keywords" in metadata.stages_completed

    def test_quality_score_range(self) -> None:
        """quality_score must be between 0.0 and 1.0.
        
        AC Reference: AC-2.5 - quality_score between 0.0-1.0
        """
        from src.models.metadata_models import ExtractionMetadata

        with pytest.raises(ValidationError):
            ExtractionMetadata(
                processing_time_ms=100.0,
                text_length=1000,
                detected_domain="test",
                domain_confidence=0.5,
                quality_score=1.5,  # Invalid: > 1.0
                stages_completed=[],
            )


class TestRejectedKeywords:
    """Tests for RejectedKeywords model."""

    def test_rejected_keywords_creation(self) -> None:
        """RejectedKeywords should have keywords list and reasons dict.
        
        AC Reference: AC-2.4 - rejected.keywords and rejected.reasons
        """
        from src.models.metadata_models import RejectedKeywords

        rejected = RejectedKeywords(
            keywords=["oceanofpdf", "'ll", "www"],
            reasons={
                "oceanofpdf": "noise_watermark",
                "'ll": "broken_contraction",
                "www": "noise_url_fragment",
            },
        )
        
        assert "oceanofpdf" in rejected.keywords
        assert rejected.reasons["oceanofpdf"] == "noise_watermark"


class TestMetadataExtractionResponse:
    """Tests for MetadataExtractionResponse model."""

    def test_full_response_creation(self) -> None:
        """Full response should include keywords, concepts, metadata, rejected."""
        from src.models.metadata_models import (
            ConceptResult,
            ExtractionMetadata,
            KeywordResult,
            MetadataExtractionResponse,
            RejectedKeywords,
        )

        response = MetadataExtractionResponse(
            keywords=[
                KeywordResult(
                    term="microservices",
                    score=0.85,
                    is_technical=True,
                    sources=["tfidf"],
                ),
            ],
            concepts=[
                ConceptResult(
                    name="distributed systems",
                    confidence=0.8,
                    domain="architecture",
                    tier="T2",
                ),
            ],
            summary=None,
            metadata=ExtractionMetadata(
                processing_time_ms=100.0,
                text_length=1000,
                detected_domain="architecture",
                domain_confidence=0.75,
                quality_score=0.82,
                stages_completed=["keywords", "concepts"],
            ),
            rejected=RejectedKeywords(
                keywords=["www"],
                reasons={"www": "noise_url_fragment"},
            ),
        )
        
        assert len(response.keywords) == 1
        assert response.keywords[0].term == "microservices"
        assert len(response.concepts) == 1
        assert response.summary is None
        assert response.metadata.quality_score == 0.82
        assert "www" in response.rejected.keywords

    def test_response_with_summary(self) -> None:
        """Response can include optional summary."""
        from src.models.metadata_models import (
            ExtractionMetadata,
            MetadataExtractionResponse,
            RejectedKeywords,
        )

        response = MetadataExtractionResponse(
            keywords=[],
            concepts=[],
            summary="This chapter covers microservices architecture.",
            metadata=ExtractionMetadata(
                processing_time_ms=100.0,
                text_length=1000,
                detected_domain="architecture",
                domain_confidence=0.75,
                quality_score=0.82,
                stages_completed=["keywords", "concepts", "summary"],
            ),
            rejected=RejectedKeywords(keywords=[], reasons={}),
        )
        
        assert response.summary is not None
        assert "summary" in response.metadata.stages_completed
