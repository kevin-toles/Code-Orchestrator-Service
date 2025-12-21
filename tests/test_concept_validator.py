"""Unit tests for ConceptValidator - HCE-4.5.

Tests the two-stage validation pipeline:
1. Pattern Filter - Author names, copyright, noise terms
2. SBERT Validation - Semantic similarity to programming concepts
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.nlp.concept_validator import (
    ConceptValidator,
    ConceptValidationConfig,
    ConceptValidationResult,
    REJECTION_AUTHOR_PATTERN,
    REJECTION_NOISE_TERM,
    REJECTION_LOW_SIMILARITY,
    STAGE_PATTERN_FILTER,
    STAGE_SBERT_VALIDATE,
    KNOWN_AUTHORS,
    NOISE_TERMS,
    SEED_PROGRAMMING_CONCEPTS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def validator() -> ConceptValidator:
    """Create default validator instance."""
    return ConceptValidator()


@pytest.fixture
def validator_pattern_only() -> ConceptValidator:
    """Create validator with only pattern filter enabled."""
    config = ConceptValidationConfig(
        enable_pattern_filter=True,
        enable_sbert_validation=False,
    )
    return ConceptValidator(config=config)


@pytest.fixture
def validator_sbert_only() -> ConceptValidator:
    """Create validator with only SBERT validation enabled."""
    config = ConceptValidationConfig(
        enable_pattern_filter=False,
        enable_sbert_validation=True,
    )
    return ConceptValidator(config=config)


@pytest.fixture
def sample_concepts() -> list[str]:
    """Sample concepts mixing valid and invalid terms."""
    return [
        "Information Hiding",      # Valid - APOSD concept
        "Complexity",              # Valid - seed concept
        "John Ousterhout",         # Invalid - author name
        "Copyright",               # Invalid - copyright
        "taking",                  # Invalid - noise term
        "Module",                  # Valid - seed concept
        "Design Pattern",          # Valid - seed concept
    ]


# =============================================================================
# Pattern Filter Tests
# =============================================================================


class TestPatternFilter:
    """Tests for Stage 1 - Pattern Filter."""

    def test_filters_copyright_keyword(self, validator_pattern_only: ConceptValidator) -> None:
        """AC-4.5.1: Filter concepts containing 'Copyright'."""
        concepts = ["Copyright Notice", "Software Design", "© 2023"]
        result = validator_pattern_only.validate(concepts)

        assert "Software Design" in result.valid_concepts
        assert "Copyright Notice" in result.rejected_concepts
        assert "© 2023" in result.rejected_concepts
        assert result.rejection_reasons["Copyright Notice"] == REJECTION_AUTHOR_PATTERN

    def test_filters_known_authors(self, validator_pattern_only: ConceptValidator) -> None:
        """AC-4.5.2: Filter concepts containing known author names."""
        concepts = [
            "John Ousterhout",          # Known author
            "Design by Fowler",          # Known author
            "Clean Code",               # Valid
            "Martin Luther King",       # Has 'martin' but context is author
        ]
        result = validator_pattern_only.validate(concepts)

        assert "Clean Code" in result.valid_concepts
        assert "John Ousterhout" in result.rejected_concepts
        assert "Design by Fowler" in result.rejected_concepts
        assert result.rejection_reasons["John Ousterhout"] == REJECTION_AUTHOR_PATTERN

    def test_filters_noise_terms(self, validator_pattern_only: ConceptValidator) -> None:
        """AC-4.5.3: Filter generic noise terms."""
        concepts = ["taking", "little", "Complexity", "things"]
        result = validator_pattern_only.validate(concepts)

        assert "Complexity" in result.valid_concepts
        assert len([c for c in concepts if c in NOISE_TERMS]) == 3
        for noise in ["taking", "little", "things"]:
            assert noise in result.rejected_concepts
            assert result.rejection_reasons[noise] == REJECTION_NOISE_TERM

    def test_filters_short_concepts(self, validator_pattern_only: ConceptValidator) -> None:
        """AC-4.5.4: Filter concepts shorter than min_length."""
        config = ConceptValidationConfig(
            enable_pattern_filter=True,
            enable_sbert_validation=False,
            min_concept_length=3,
        )
        validator = ConceptValidator(config=config)
        concepts = ["a", "API", "Design"]
        result = validator.validate(concepts)

        assert "API" in result.valid_concepts
        assert "Design" in result.valid_concepts
        assert "a" in result.rejected_concepts

    def test_additional_noise_terms(self) -> None:
        """AC-4.5.5: Support additional custom noise terms."""
        config = ConceptValidationConfig(
            enable_pattern_filter=True,
            enable_sbert_validation=False,
            additional_noise_terms=frozenset({"customnoise"}),
        )
        validator = ConceptValidator(config=config)
        concepts = ["customnoise", "Complexity"]
        result = validator.validate(concepts)

        assert "customnoise" in result.rejected_concepts
        assert "Complexity" in result.valid_concepts

    def test_additional_authors(self) -> None:
        """AC-4.5.6: Support additional custom author names."""
        config = ConceptValidationConfig(
            enable_pattern_filter=True,
            enable_sbert_validation=False,
            additional_authors=frozenset({"customauthor"}),
        )
        validator = ConceptValidator(config=config)
        concepts = ["Design by CustomAuthor", "Complexity"]
        result = validator.validate(concepts)

        assert "Design by CustomAuthor" in result.rejected_concepts
        assert "Complexity" in result.valid_concepts


# =============================================================================
# SBERT Validation Tests
# =============================================================================


class TestSBERTValidation:
    """Tests for Stage 2 - SBERT Semantic Validation."""

    def test_validates_seed_concepts(self, validator_sbert_only: ConceptValidator) -> None:
        """AC-4.5.7: Exact seed concepts get similarity=1.0."""
        concepts = ["algorithm", "complexity", "design pattern"]
        result = validator_sbert_only.validate(concepts)

        assert all(c in result.valid_concepts for c in concepts)
        # Exact matches should have very high similarity
        for c in concepts:
            assert result.similarity_scores[c] >= 0.9

    def test_validates_similar_concepts(self, validator_sbert_only: ConceptValidator) -> None:
        """AC-4.5.8: Similar programming concepts pass validation."""
        concepts = ["Information Hiding", "Deep modules", "Error handling"]
        result = validator_sbert_only.validate(concepts)

        # These are APOSD concepts, should be similar to seeds
        assert len(result.valid_concepts) >= 2
        for c in result.valid_concepts:
            assert result.similarity_scores[c] >= 0.35

    def test_rejects_dissimilar_concepts(self, validator_sbert_only: ConceptValidator) -> None:
        """AC-4.5.9: Non-programming concepts rejected."""
        concepts = ["banana", "weather", "breakfast"]
        result = validator_sbert_only.validate(concepts)

        # These should have low similarity to programming seeds
        assert len(result.rejected_concepts) >= 2
        for c in result.rejected_concepts:
            assert result.rejection_reasons[c] == REJECTION_LOW_SIMILARITY

    def test_similarity_threshold_adjustable(self) -> None:
        """AC-4.5.10: Threshold can be adjusted for stricter/looser validation."""
        # Strict threshold
        strict_config = ConceptValidationConfig(
            enable_pattern_filter=False,
            enable_sbert_validation=True,
            sbert_similarity_threshold=0.8,
        )
        strict_validator = ConceptValidator(config=strict_config)

        # Loose threshold
        loose_config = ConceptValidationConfig(
            enable_pattern_filter=False,
            enable_sbert_validation=True,
            sbert_similarity_threshold=0.2,
        )
        loose_validator = ConceptValidator(config=loose_config)

        concepts = ["general purpose", "specific implementation"]
        strict_result = strict_validator.validate(concepts)
        loose_result = loose_validator.validate(concepts)

        # Loose should accept more
        assert len(loose_result.valid_concepts) >= len(strict_result.valid_concepts)


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Tests for complete validation pipeline."""

    def test_two_stage_validation(self, validator: ConceptValidator, sample_concepts: list[str]) -> None:
        """AC-4.5.11: Both stages execute in sequence."""
        result = validator.validate(sample_concepts)

        assert STAGE_PATTERN_FILTER in result.stages_executed
        assert STAGE_SBERT_VALIDATE in result.stages_executed
        assert result.stages_executed.index(STAGE_PATTERN_FILTER) < result.stages_executed.index(STAGE_SBERT_VALIDATE)

    def test_pattern_filter_runs_before_sbert(self, validator: ConceptValidator) -> None:
        """AC-4.5.12: Author names filtered before expensive SBERT call."""
        concepts = ["John Ousterhout", "Information Hiding"]
        result = validator.validate(concepts)

        # John Ousterhout should be rejected by pattern filter (not SBERT)
        assert "John Ousterhout" in result.rejected_concepts
        assert result.rejection_reasons["John Ousterhout"] == REJECTION_AUTHOR_PATTERN
        # Information Hiding passes pattern filter, gets validated by SBERT
        assert "Information Hiding" in result.valid_concepts
        assert "Information Hiding" in result.similarity_scores

    def test_empty_input(self, validator: ConceptValidator) -> None:
        """AC-4.5.13: Empty input returns empty result."""
        result = validator.validate([])

        assert result.valid_concepts == []
        assert result.rejected_concepts == []

    def test_all_rejected_concepts_tracked(self, validator: ConceptValidator) -> None:
        """AC-4.5.14: All rejected concepts have reasons."""
        concepts = ["Copyright Notice", "John Ousterhout", "banana", "taking"]
        result = validator.validate(concepts)

        for rejected in result.rejected_concepts:
            assert rejected in result.rejection_reasons

    def test_similarity_scores_only_for_valid(self, validator: ConceptValidator) -> None:
        """AC-4.5.15: Similarity scores only populated for valid concepts."""
        concepts = ["Complexity", "John Ousterhout"]
        result = validator.validate(concepts)

        # Valid concept should have score
        if "Complexity" in result.valid_concepts:
            assert "Complexity" in result.similarity_scores
        # Rejected should not have score
        assert "John Ousterhout" not in result.similarity_scores


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for validation configuration."""

    def test_disable_pattern_filter(self) -> None:
        """AC-4.5.16: Pattern filter can be disabled."""
        config = ConceptValidationConfig(enable_pattern_filter=False)
        validator = ConceptValidator(config=config)
        concepts = ["Copyright Notice", "Complexity"]
        result = validator.validate(concepts)

        assert STAGE_PATTERN_FILTER not in result.stages_executed
        # Copyright should not be filtered since pattern filter is off
        # It will only be rejected if SBERT finds low similarity
        assert "Complexity" in result.valid_concepts

    def test_disable_sbert_validation(self) -> None:
        """AC-4.5.17: SBERT validation can be disabled."""
        config = ConceptValidationConfig(enable_sbert_validation=False)
        validator = ConceptValidator(config=config)
        concepts = ["Complexity", "banana"]
        result = validator.validate(concepts)

        assert STAGE_SBERT_VALIDATE not in result.stages_executed
        assert result.similarity_scores == {}

    def test_default_config_enables_both(self) -> None:
        """AC-4.5.18: Default config enables both stages."""
        config = ConceptValidationConfig()

        assert config.enable_pattern_filter is True
        assert config.enable_sbert_validation is True
        assert config.sbert_similarity_threshold == 0.35


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with real SBERT model."""

    def test_aposd_chapter_concepts(self, validator: ConceptValidator) -> None:
        """AC-4.5.19: Validates APOSD-specific concepts correctly."""
        aposd_concepts = [
            "Information Hiding",
            "Deep modules",
            "Shallow modules",
            "Pass-through methods",
            "Exception masking",
            "John Ousterhout",  # Author - should be rejected
        ]
        result = validator.validate(aposd_concepts)

        # Programming concepts should pass
        assert "Information Hiding" in result.valid_concepts
        assert "Deep modules" in result.valid_concepts
        # Author should be rejected
        assert "John Ousterhout" in result.rejected_concepts

    def test_known_authors_coverage(self) -> None:
        """AC-4.5.20: Verify known authors set covers major textbook authors."""
        expected_authors = {
            "ousterhout", "fowler", "martin", "gamma",
            "knuth", "bloch", "kernighan", "ritchie"
        }
        assert expected_authors.issubset(KNOWN_AUTHORS)

    def test_seed_concepts_coverage(self) -> None:
        """AC-4.5.21: Verify seed concepts cover core programming domains."""
        seed_set = set(SEED_PROGRAMMING_CONCEPTS)

        # Core concepts should be present
        assert "algorithm" in seed_set
        assert "data structure" in seed_set
        assert "design pattern" in seed_set
        assert "abstraction" in seed_set
        assert "complexity" in seed_set

        # APOSD-specific seeds
        assert "information hiding" in seed_set
        assert "deep module" in seed_set
