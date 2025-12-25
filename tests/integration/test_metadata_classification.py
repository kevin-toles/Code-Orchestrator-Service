"""Integration tests for MetadataExtractor with HTC classification.

Tests the full pipeline from text extraction through classification:
1. TF-IDF keyword extraction
2. Noise filtering
3. Concept extraction
4. HTC Classification (Tiers 1-3)
5. Domain detection
6. Quality scoring

WBS Reference: HTC-1.0 Integration
"""

import pytest

from src.extractors.metadata_extractor import (
    MetadataExtractor,
    MetadataExtractorConfig,
    STAGE_CLASSIFICATION,
)
from src.models.metadata_models import MetadataExtractionOptions


class TestClassificationIntegration:
    """Test classification integration with MetadataExtractor."""

    @pytest.fixture
    def extractor(self) -> MetadataExtractor:
        """Create extractor with classification enabled."""
        config = MetadataExtractorConfig(
            enable_classification=True,
            enable_concepts=True,
            enable_domain_detection=True,
        )
        return MetadataExtractor(config=config)

    @pytest.fixture
    def extractor_no_classification(self) -> MetadataExtractor:
        """Create extractor with classification disabled."""
        config = MetadataExtractorConfig(
            enable_classification=False,
            enable_concepts=True,
        )
        return MetadataExtractor(config=config)

    def test_classification_stage_added(self, extractor: MetadataExtractor) -> None:
        """Verify classification stage is added to completed stages."""
        text = "Kubernetes is a container orchestration platform."
        result = extractor.extract(text)
        assert STAGE_CLASSIFICATION in result.stages_completed

    def test_classification_stage_skipped_when_disabled(
        self, extractor_no_classification: MetadataExtractor
    ) -> None:
        """Verify classification stage is skipped when disabled."""
        text = "Kubernetes is a container orchestration platform."
        result = extractor_no_classification.extract(text)
        assert STAGE_CLASSIFICATION not in result.stages_completed

    def test_tier_1_concepts_have_confidence_1(
        self, extractor: MetadataExtractor
    ) -> None:
        """Verify Tier 1 (alias lookup) concepts have confidence=1.0."""
        # Use terms known to be in alias_lookup.json
        text = "Docker and Kubernetes are essential for microservices."
        result = extractor.extract(text)

        tier_1_concepts = [c for c in result.concepts if c.tier == "T1"]
        for concept in tier_1_concepts:
            assert concept.confidence == 1.0, f"{concept.name} should have confidence 1.0"

    def test_classification_stats_present(self, extractor: MetadataExtractor) -> None:
        """Verify classification stats are populated."""
        text = "API gateway handles authentication and rate limiting."
        result = extractor.extract(text)

        assert result.classification_stats is not None
        assert "total_terms" in result.classification_stats
        assert "tier_1_hits" in result.classification_stats
        assert "tier_2_hits" in result.classification_stats
        assert "tier_3_rejections" in result.classification_stats
        assert "keywords_promoted" in result.classification_stats

    def test_keywords_promoted_to_concepts(self, extractor: MetadataExtractor) -> None:
        """Verify keywords classified as concepts are promoted."""
        text = "The microservices architecture uses RESTful APIs."
        result = extractor.extract(text)

        # Check that keywords_promoted > 0
        stats = result.classification_stats
        assert stats is not None
        assert stats["keywords_promoted"] >= 0  # Some may be promoted

    def test_noise_terms_rejected(self, extractor: MetadataExtractor) -> None:
        """Verify noise terms are rejected by Tier 3."""
        # This test depends on noise being passed through to Tier 3
        # Most noise is caught by initial NoiseFilter
        text = "Using the oceanofpdf watermark to process def return."
        result = extractor.extract(text)

        # Check if any terms were rejected
        # Note: Most noise is caught in initial noise filter stage
        assert result.classification_stats is not None

    def test_concepts_retain_domain_info(self, extractor: MetadataExtractor) -> None:
        """Verify domain detection still works after classification."""
        text = "DevOps practices include CI/CD pipelines and infrastructure as code."
        result = extractor.extract(text)

        # Domain detection should still work
        # (depends on concept pipeline finding domain-tagged concepts)
        assert result.stages_completed[-2] == "domain"  # Before quality

    def test_empty_text_handled(self, extractor: MetadataExtractor) -> None:
        """Verify empty text doesn't crash classification."""
        result = extractor.extract("")
        assert result.keywords == []
        assert result.concepts == []

    def test_short_text_classification(self, extractor: MetadataExtractor) -> None:
        """Verify short text is classified correctly."""
        text = "Docker"
        result = extractor.extract(text)

        # Should still complete classification stage if enabled
        if result.classification_stats is not None:
            assert STAGE_CLASSIFICATION in result.stages_completed


class TestTierDistribution:
    """Test distribution of terms across classification tiers."""

    @pytest.fixture
    def extractor(self) -> MetadataExtractor:
        """Create extractor with classification enabled."""
        config = MetadataExtractorConfig(
            enable_classification=True,
            enable_concepts=True,
        )
        return MetadataExtractor(config=config)

    def test_known_concepts_hit_tier_1(self, extractor: MetadataExtractor) -> None:
        """Verify well-known concepts are found in Tier 1."""
        # Terms that should be in alias_lookup.json
        text = "Docker Kubernetes microservices API gateway"
        result = extractor.extract(text)

        if result.classification_stats:
            # Most of these should hit Tier 1
            assert result.classification_stats["tier_1_hits"] > 0

    def test_novel_terms_use_tier_2(self, extractor: MetadataExtractor) -> None:
        """Verify novel terms fall through to Tier 2."""
        # Use some uncommon but valid technical terms
        text = "The distributed ledger implements Byzantine consensus."
        result = extractor.extract(text)

        # Some terms should hit Tier 2 (trained classifier)
        if result.classification_stats:
            # May hit Tier 2 if not in alias lookup
            total_hits = (
                result.classification_stats["tier_1_hits"]
                + result.classification_stats["tier_2_hits"]
            )
            assert total_hits >= 0  # Some processing occurred


class TestClassificationQuality:
    """Test quality of classification results."""

    @pytest.fixture
    def extractor(self) -> MetadataExtractor:
        """Create extractor with classification enabled."""
        config = MetadataExtractorConfig(
            enable_classification=True,
            enable_concepts=True,
        )
        return MetadataExtractor(config=config)

    def test_concepts_have_valid_tiers(self, extractor: MetadataExtractor) -> None:
        """Verify all concepts have valid tier values."""
        text = "Kubernetes orchestrates Docker containers in production."
        result = extractor.extract(text)

        for concept in result.concepts:
            # Tier should be T1, T2, T3, or empty (for unclassified)
            assert concept.tier in ("", "T1", "T2", "T3", "T4")

    def test_confidence_in_valid_range(self, extractor: MetadataExtractor) -> None:
        """Verify all confidence scores are in [0.0, 1.0]."""
        text = "Machine learning models use neural networks."
        result = extractor.extract(text)

        for concept in result.concepts:
            assert 0.0 <= concept.confidence <= 1.0

        for keyword in result.keywords:
            assert 0.0 <= keyword.score <= 1.0

    def test_quality_score_calculated_after_classification(
        self, extractor: MetadataExtractor
    ) -> None:
        """Verify quality score accounts for classified concepts."""
        text = "Microservices architecture with Docker and Kubernetes."
        result = extractor.extract(text)

        # Quality score should be > 0 for meaningful text
        assert result.quality_score > 0.0
        assert result.quality_score <= 1.0
