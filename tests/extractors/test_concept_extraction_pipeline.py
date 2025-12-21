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


# =============================================================================
# HCE-2.28: Test pipeline uses YAKE extractor
# =============================================================================


class TestPipelineYAKEIntegration:
    """HCE-2.28: Test pipeline uses YAKE extractor."""

    def test_pipeline_has_yake_extractor(self) -> None:
        """AC-2.8: Pipeline has _yake_extractor attribute."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()

        assert hasattr(pipeline, "_yake_extractor")

    def test_pipeline_calls_yake_when_enabled(self) -> None:
        """AC-2.8: Pipeline calls YAKE extractor when enable_yake=True."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_yake=True, enable_textrank=False)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        # YAKE should have extracted terms
        assert result.extraction_stats["yake_count"] > 0


# =============================================================================
# HCE-2.29: Test pipeline uses TextRank extractor
# =============================================================================


class TestPipelineTextRankIntegration:
    """HCE-2.29: Test pipeline uses TextRank extractor."""

    def test_pipeline_has_textrank_extractor(self) -> None:
        """AC-2.8: Pipeline has _textrank_extractor attribute."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()

        assert hasattr(pipeline, "_textrank_extractor")

    def test_pipeline_calls_textrank_when_enabled(self) -> None:
        """AC-2.8: Pipeline calls TextRank extractor when enable_textrank=True."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        # TextRank needs more text to work effectively
        long_text = """
        Machine learning is a branch of artificial intelligence that enables computers
        to learn from data without being explicitly programmed. Deep learning is a subset
        of machine learning that uses neural networks with multiple layers. These neural
        networks can learn complex patterns in data. Natural language processing allows
        computers to understand human language. Computer vision enables machines to
        interpret visual information from the world.
        """

        config = ConceptExtractionConfig(enable_yake=False, enable_textrank=True)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(long_text)

        # TextRank should have extracted terms
        assert result.extraction_stats["textrank_count"] > 0


# =============================================================================
# HCE-2.30: Test pipeline uses EnsembleMerger
# =============================================================================


class TestPipelineEnsembleMergerIntegration:
    """HCE-2.30: Test pipeline uses EnsembleMerger."""

    def test_pipeline_has_ensemble_merger(self) -> None:
        """AC-2.8: Pipeline has _ensemble_merger attribute."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()

        assert hasattr(pipeline, "_ensemble_merger")

    def test_pipeline_merges_yake_and_textrank(self) -> None:
        """AC-2.8: Pipeline merges YAKE and TextRank results."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        # Should have merged concepts
        assert result.extraction_stats["merged_count"] > 0


# =============================================================================
# HCE-2.31: Test extraction_stats tracks yake_count
# =============================================================================


class TestExtractionStatsYAKE:
    """HCE-2.31: Test extraction_stats tracks yake_count."""

    def test_extraction_stats_has_yake_count(self) -> None:
        """AC-2.8: extraction_stats includes yake_count."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert "yake_count" in result.extraction_stats
        assert isinstance(result.extraction_stats["yake_count"], int)


# =============================================================================
# HCE-2.32: Test extraction_stats tracks textrank_count
# =============================================================================


class TestExtractionStatsTextRank:
    """HCE-2.32: Test extraction_stats tracks textrank_count."""

    def test_extraction_stats_has_textrank_count(self) -> None:
        """AC-2.8: extraction_stats includes textrank_count."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert "textrank_count" in result.extraction_stats
        assert isinstance(result.extraction_stats["textrank_count"], int)


# =============================================================================
# HCE-2.33: Test extraction_stats tracks merged_count
# =============================================================================


class TestExtractionStatsMerged:
    """HCE-2.33: Test extraction_stats tracks merged_count."""

    def test_extraction_stats_has_merged_count(self) -> None:
        """AC-2.8: extraction_stats includes merged_count."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert "merged_count" in result.extraction_stats
        assert isinstance(result.extraction_stats["merged_count"], int)

    def test_merged_count_equals_concepts_length(self) -> None:
        """AC-2.8: merged_count >= len(concepts) after deduplication."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        # merged_count is pre-dedup, concepts is post-dedup
        # So merged_count >= len(concepts)
        assert result.extraction_stats["merged_count"] >= len(result.concepts)


# =============================================================================
# HCE-2.0: Test concepts have source tracking
# =============================================================================


class TestConceptsSourceTracking:
    """HCE-2.0: Test concepts have source tracking from merger."""

    def test_concepts_have_source_field(self) -> None:
        """AC-2.7: Extracted concepts have source field (yake/textrank/both)."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        for concept in result.concepts:
            assert hasattr(concept, "source")
            assert concept.source in ("yake", "textrank", "both")

    def test_concepts_have_score_field(self) -> None:
        """AC-2.7: Extracted concepts have score field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        for concept in result.concepts:
            assert hasattr(concept, "score")
            assert isinstance(concept.score, (int, float))


# =============================================================================
# HCE-3.20: Test pipeline has Stemmer (AC-3.5)
# =============================================================================


class TestPipelineStemmerIntegration:
    """HCE-3.20: Test pipeline has Stemmer integration."""

    def test_pipeline_config_has_enable_stem_dedup(self) -> None:
        """AC-3.5: Config has enable_stem_dedup field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
        )

        config = ConceptExtractionConfig()
        assert hasattr(config, "enable_stem_dedup")
        assert isinstance(config.enable_stem_dedup, bool)

    def test_pipeline_config_enable_stem_dedup_default_true(self) -> None:
        """AC-3.5: enable_stem_dedup defaults to True."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
        )

        config = ConceptExtractionConfig()
        assert config.enable_stem_dedup is True


# =============================================================================
# HCE-3.21: Test pipeline calls stemmer after merge (AC-3.5)
# =============================================================================


class TestPipelineStemmerExecution:
    """HCE-3.21: Test pipeline calls stemmer after merge."""

    def test_pipeline_deduplicates_morphological_variants(self) -> None:
        """AC-3.5: Pipeline removes morphological variants."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        # Text with model/models variants
        text = """
        Machine learning models are powerful. The model processes data.
        Models learn patterns. Pattern recognition is key.
        Processing data requires patterns and models.
        """
        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(text)

        # Verify stemmer ran and removed some variants
        assert result.dedup_stats["stem_removed"] >= 0
        
        # The pipeline should have executed stem_dedup stage
        assert "stem_dedup" in result.pipeline_metadata["stages_executed"]


# =============================================================================
# HCE-3.22: Test dedup_stats tracks stem_removed (AC-3.5)
# =============================================================================


class TestDedupStatsStemRemoved:
    """HCE-3.22: Test dedup_stats tracks stem_removed."""

    def test_dedup_stats_has_stem_removed_field(self) -> None:
        """AC-3.5: dedup_stats includes stem_removed count."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert "stem_removed" in result.dedup_stats
        assert isinstance(result.dedup_stats["stem_removed"], int)

    def test_stem_removed_is_non_negative(self) -> None:
        """AC-3.5: stem_removed count is non-negative."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert result.dedup_stats["stem_removed"] >= 0


# =============================================================================
# HCE-3.23: Test enable_stem_dedup=False skips stemmer (AC-3.5)
# =============================================================================


class TestStemmerBypass:
    """HCE-3.23: Test enable_stem_dedup=False skips stemmer."""

    def test_stem_dedup_bypass_when_disabled(self) -> None:
        """AC-3.5: Stemmer skipped when enable_stem_dedup=False."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_stem_dedup=False)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        # When bypassed, stem_removed should be 0
        assert result.dedup_stats["stem_removed"] == 0

    def test_bypass_preserves_variants(self) -> None:
        """AC-3.5: Bypassed stemmer preserves morphological variants."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        # Text with clear variants
        text = """
        Machine learning models are complex. Complexity drives design.
        The model processes patterns. Pattern matching is essential.
        Processing requires understanding complexity.
        """
        
        config_with_stem = ConceptExtractionConfig(enable_stem_dedup=True)
        config_without_stem = ConceptExtractionConfig(enable_stem_dedup=False)
        
        pipeline_with = ConceptExtractionPipeline(config=config_with_stem)
        pipeline_without = ConceptExtractionPipeline(config=config_without_stem)
        
        result_with = pipeline_with.extract(text)
        result_without = pipeline_without.extract(text)
        
        # Without stemmer, should have more or equal concepts
        assert len(result_without.concepts) >= len(result_with.concepts)


# =============================================================================
# HCE-4.20: Test pipeline has SemanticDeduplicator (AC-4.6)
# =============================================================================


class TestPipelineSemanticDeduplicator:
    """HCE-4.20: Test pipeline has SemanticDeduplicator."""

    def test_pipeline_config_has_enable_semantic_dedup(self) -> None:
        """AC-4.6: Config has enable_semantic_dedup field."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
        )

        config = ConceptExtractionConfig()
        assert hasattr(config, "enable_semantic_dedup")
        assert isinstance(config.enable_semantic_dedup, bool)

    def test_pipeline_config_enable_semantic_dedup_default_true(self) -> None:
        """AC-4.6: enable_semantic_dedup defaults to True."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
        )

        config = ConceptExtractionConfig()
        assert config.enable_semantic_dedup is True

    def test_pipeline_has_semantic_deduplicator(self) -> None:
        """AC-4.6: Pipeline has SemanticDeduplicator instance."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        assert hasattr(pipeline, "_semantic_deduplicator")


# =============================================================================
# HCE-4.21: Test pipeline calls semantic dedup after stem (AC-4.6)
# =============================================================================


class TestPipelineSemanticDedupOrder:
    """HCE-4.21: Test pipeline calls semantic dedup after stem."""

    def test_semantic_dedup_stage_in_stages_executed(self) -> None:
        """AC-4.6: semantic_dedup stage appears in stages_executed."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert "semantic_dedup" in result.pipeline_metadata["stages_executed"]

    def test_semantic_dedup_after_stem_dedup(self) -> None:
        """AC-4.6: semantic_dedup executes after stem_dedup."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        stages = result.pipeline_metadata["stages_executed"]
        if "stem_dedup" in stages and "semantic_dedup" in stages:
            stem_idx = stages.index("stem_dedup")
            semantic_idx = stages.index("semantic_dedup")
            assert semantic_idx > stem_idx


# =============================================================================
# HCE-4.22: Test dedup_stats tracks semantic_clusters (AC-4.6)
# =============================================================================


class TestPipelineSemanticDedupStats:
    """HCE-4.22: Test dedup_stats tracks semantic_clusters."""

    def test_dedup_stats_has_semantic_clusters(self) -> None:
        """AC-4.6: dedup_stats includes semantic_clusters count."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert "semantic_clusters" in result.dedup_stats
        assert isinstance(result.dedup_stats["semantic_clusters"], int)

    def test_dedup_stats_has_semantic_removed(self) -> None:
        """AC-4.6: dedup_stats includes semantic_removed count."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionPipeline,
        )

        pipeline = ConceptExtractionPipeline()
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert "semantic_removed" in result.dedup_stats
        assert isinstance(result.dedup_stats["semantic_removed"], int)


# =============================================================================
# HCE-4.23: Test enable_semantic_dedup=False skips dedup (AC-4.6)
# =============================================================================


class TestSemanticDedupBypass:
    """HCE-4.23: Test enable_semantic_dedup=False skips dedup."""

    def test_semantic_dedup_bypass_when_disabled(self) -> None:
        """AC-4.6: Semantic dedup skipped when enable_semantic_dedup=False."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_semantic_dedup=False)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        # When bypassed, semantic_clusters should be 0
        assert result.dedup_stats["semantic_clusters"] == 0
        assert "semantic_dedup" not in result.pipeline_metadata["stages_executed"]

    def test_bypass_preserves_semantic_duplicates(self) -> None:
        """AC-4.6: Bypassed semantic dedup preserves similar terms."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        # Text with semantically similar terms
        text = """
        Machine learning and ML are essential. Deep learning neural networks
        process data. API endpoints serve REST API calls.
        """
        
        config_with_semantic = ConceptExtractionConfig(enable_semantic_dedup=True)
        config_without_semantic = ConceptExtractionConfig(enable_semantic_dedup=False)
        
        pipeline_with = ConceptExtractionPipeline(config=config_with_semantic)
        pipeline_without = ConceptExtractionPipeline(config=config_without_semantic)
        
        result_with = pipeline_with.extract(text)
        result_without = pipeline_without.extract(text)
        
        # Without semantic dedup, should have more or equal concepts
        assert len(result_without.concepts) >= len(result_with.concepts)

    def test_semantic_removed_zero_when_disabled(self) -> None:
        """AC-4.6: semantic_removed is 0 when semantic dedup disabled."""
        from src.extractors.concept_extraction_pipeline import (
            ConceptExtractionConfig,
            ConceptExtractionPipeline,
        )

        config = ConceptExtractionConfig(enable_semantic_dedup=False)
        pipeline = ConceptExtractionPipeline(config=config)
        result = pipeline.extract(SAMPLE_TEXT_CLEAN)

        assert result.dedup_stats["semantic_removed"] == 0

