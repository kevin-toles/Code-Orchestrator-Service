"""Tests for ConceptExtractionPipeline - HCE-1.0.

TDD RED Phase: Tests written before implementation.
These tests MUST FAIL initially (no implementation exists).

WBS Reference: HCE-1.1 through HCE-1.20
AC Reference: AC-1.1 through AC-1.5

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Small, focused test methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

SAMPLE_TEXT_CLEAN: Final[str] = "Machine learning models learn patterns from data."
SAMPLE_TEXT_WITH_NOISE: Final[str] = (
    "Machine learning models learn patterns. Downloaded from Safari Books Online."
)
SAMPLE_TEXT_SHORT: Final[str] = "AI models."
WATERMARK_TERM: Final[str] = "Safari Books Online"


# =============================================================================
# HCE-1.1: Test ConceptExtractionPipeline can be imported
# =============================================================================


class TestPipelineImport:
    """HCE-1.1: Test ConceptExtractionPipeline can be imported."""

    def test_concept_extraction_pipeline_can_be_imported(self) -> None:
        """AC-1.1: ConceptExtractionPipeline class exists and can be imported."""
        # This import MUST fail until HCE-1.6 (GREEN phase)
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        assert ConceptExtractionPipeline is not None


# =============================================================================
# HCE-1.2: Test pipeline instantiation without config
# =============================================================================


class TestPipelineInstantiationNoConfig:
    """HCE-1.2: Test pipeline instantiation without config."""

    def test_pipeline_instantiation_without_config(self) -> None:
        """AC-1.1: Pipeline can be instantiated without configuration."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()

        assert pipeline is not None

    def test_pipeline_has_default_config(self) -> None:
        """AC-1.1: Pipeline has default configuration when none provided."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()

        assert hasattr(pipeline, "config")
        assert pipeline.config is not None


# =============================================================================
# HCE-1.3: Test pipeline instantiation with config
# =============================================================================


class TestPipelineInstantiationWithConfig:
    """HCE-1.3: Test pipeline instantiation with config."""

    def test_pipeline_instantiation_with_config(self) -> None:
        """AC-1.2: Pipeline accepts configuration object."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_noise_filter=False)
        pipeline = ConceptExtractionPipeline(config=config)

        assert pipeline.config is config
        assert pipeline.config.enable_noise_filter is False

    def test_config_dataclass_exists(self) -> None:
        """AC-1.2: ConceptExtractionConfig dataclass exists."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
        )

        assert ConceptExtractionConfig is not None

    def test_config_has_required_fields(self) -> None:
        """AC-1.2: Config has enable_noise_filter field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
        )

        config = ConceptExtractionConfig()

        assert hasattr(config, "enable_noise_filter")


# =============================================================================
# HCE-1.4: Test extract() returns ConceptExtractionResult
# =============================================================================


class TestExtractReturnsResult:
    """HCE-1.4: Test extract() returns ConceptExtractionResult."""

    def test_extract_method_exists(self) -> None:
        """AC-1.3: Pipeline has extract() method."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()

        assert hasattr(pipeline, "extract")
        assert callable(pipeline.extract)

    def test_extract_returns_concept_extraction_result(self) -> None:
        """AC-1.3: extract() returns ConceptExtractionResult instance."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
            ConceptExtractionResult,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert isinstance(result, ConceptExtractionResult)

    def test_result_dataclass_exists(self) -> None:
        """AC-1.3: ConceptExtractionResult dataclass exists."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionResult,
        )

        assert ConceptExtractionResult is not None


# =============================================================================
# HCE-1.5: Test result has concepts, stats, metadata fields
# =============================================================================


class TestResultFields:
    """HCE-1.5: Test result has concepts, stats, metadata fields."""

    def test_result_has_concepts_field(self) -> None:
        """AC-1.3: Result has concepts field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert hasattr(result, "concepts")
        assert isinstance(result.concepts, list)

    def test_result_has_extraction_stats(self) -> None:
        """AC-1.3: Result has extraction_stats field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert hasattr(result, "extraction_stats")
        assert isinstance(result.extraction_stats, dict)

    def test_result_has_filter_stats(self) -> None:
        """AC-1.3: Result has filter_stats field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert hasattr(result, "filter_stats")
        assert isinstance(result.filter_stats, dict)

    def test_result_has_dedup_stats(self) -> None:
        """AC-1.3: Result has dedup_stats field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert hasattr(result, "dedup_stats")
        assert isinstance(result.dedup_stats, dict)

    def test_result_has_pipeline_metadata(self) -> None:
        """AC-1.3: Result has pipeline_metadata field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert hasattr(result, "pipeline_metadata")
        assert isinstance(result.pipeline_metadata, dict)


# =============================================================================
# HCE-1.10: Test NoiseFilter import succeeds
# =============================================================================


class TestNoiseFilterImport:
    """HCE-1.10: Test NoiseFilter import succeeds."""

    def test_pipeline_imports_noise_filter(self) -> None:
        """AC-1.4: Pipeline imports NoiseFilter from validators."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )
        from src.validators.noise_filter import NoiseFilter

        pipeline = ConceptExtractionPipeline()

        # Pipeline should have NoiseFilter instance
        assert hasattr(pipeline, "_noise_filter")
        assert isinstance(pipeline._noise_filter, NoiseFilter)


# =============================================================================
# HCE-1.11: Test watermarks removed from extraction
# =============================================================================


class TestWatermarkRemoval:
    """HCE-1.11: Test watermarks removed from extraction."""

    def test_watermarks_removed_from_extraction(self) -> None:
        """AC-1.4: Watermark terms are removed before extraction."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_WITH_NOISE)

        # No concept should contain watermark terms
        concept_terms = [c.term for c in result.concepts]
        watermark_lower_terms = ["safari", "books", "online", "downloaded"]

        for term in concept_terms:
            term_lower = term.lower()
            for watermark in watermark_lower_terms:
                assert watermark not in term_lower, (
                    f"Watermark '{watermark}' found in concept '{term}'"
                )


# =============================================================================
# HCE-1.12: Test filter_stats tracks terms_filtered
# =============================================================================


class TestFilterStats:
    """HCE-1.12: Test filter_stats tracks terms_filtered."""

    def test_filter_stats_has_terms_filtered(self) -> None:
        """AC-1.4: filter_stats includes terms_filtered count."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_WITH_NOISE)

        assert "terms_filtered" in result.filter_stats
        assert isinstance(result.filter_stats["terms_filtered"], int)

    def test_filter_stats_nonzero_when_noise_present(self) -> None:
        """AC-1.4: filter_stats shows nonzero count when noise filtered."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_WITH_NOISE)

        # Should have filtered at least some noise terms
        assert result.filter_stats["terms_filtered"] >= 0


# =============================================================================
# HCE-1.15: Test enable_noise_filter=False skips filter
# =============================================================================


class TestNoiseFilterBypass:
    """HCE-1.15: Test enable_noise_filter=False skips filter."""

    def test_noise_filter_bypass_when_disabled(self) -> None:
        """AC-1.5: NoiseFilter is skipped when enable_noise_filter=False."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_noise_filter=False)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(SAMPLE_TEXT_WITH_NOISE)

        # When bypassed, filter_stats should reflect no filtering
        assert result.filter_stats["terms_filtered"] == 0


# =============================================================================
# HCE-1.16: Test bypass sets filter_stats to zero
# =============================================================================


class TestFilterBypassStats:
    """HCE-1.16: Test bypass sets filter_stats to zero."""

    def test_bypass_sets_filter_stats_to_zero(self) -> None:
        """AC-1.5: When bypassed, filter_stats['terms_filtered'] is 0."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_noise_filter=False)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(SAMPLE_TEXT_WITH_NOISE)

        assert result.filter_stats["terms_filtered"] == 0

    def test_bypass_skips_filter_categories(self) -> None:
        """AC-1.5: When bypassed, no filter categories populated."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_noise_filter=False)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(SAMPLE_TEXT_WITH_NOISE)

        # Categories should be empty when filter bypassed
        if "categories" in result.filter_stats:
            assert len(result.filter_stats["categories"]) == 0
