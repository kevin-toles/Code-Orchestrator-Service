"""Tests for YAKEExtractor - HCE-2.3 through HCE-2.9.

TDD RED Phase: Tests written before implementation.
These tests MUST FAIL initially (no implementation exists).

WBS Reference: HCE-2.3 through HCE-2.12
AC Reference: AC-2.1, AC-2.3, AC-2.5

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Small, focused test methods
- #12: YAKE instance caching tested
"""

from __future__ import annotations

from typing import Final

import pytest

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

SAMPLE_TEXT: Final[str] = """
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience. Deep learning uses neural 
networks with multiple layers. Natural language processing handles text 
and speech understanding.
"""

SAMPLE_TEXT_SHORT: Final[str] = "AI models process data."

EMPTY_TEXT: Final[str] = ""


# =============================================================================
# HCE-2.3: Test YAKEExtractor can be imported
# =============================================================================


class TestYAKEExtractorImport:
    """HCE-2.3: Test YAKEExtractor can be imported."""

    def test_yake_extractor_can_be_imported(self) -> None:
        """AC-2.3: YAKEExtractor class exists and can be imported."""
        from src.nlp.yake_extractor import YAKEExtractor

        assert YAKEExtractor is not None

    def test_yake_config_can_be_imported(self) -> None:
        """AC-2.5: YAKEConfig dataclass exists and can be imported."""
        from src.nlp.yake_extractor import YAKEConfig

        assert YAKEConfig is not None


# =============================================================================
# HCE-2.4: Test YAKEConfig has required fields
# =============================================================================


class TestYAKEConfig:
    """HCE-2.4: Test YAKEConfig has required fields."""

    def test_config_has_top_n_field(self) -> None:
        """AC-2.5: Config has top_n field with default value."""
        from src.nlp.yake_extractor import YAKEConfig

        config = YAKEConfig()
        assert hasattr(config, "top_n")
        assert isinstance(config.top_n, int)

    def test_config_has_n_gram_size_field(self) -> None:
        """AC-2.5: Config has n_gram_size field with default value."""
        from src.nlp.yake_extractor import YAKEConfig

        config = YAKEConfig()
        assert hasattr(config, "n_gram_size")
        assert isinstance(config.n_gram_size, int)

    def test_config_has_dedup_threshold_field(self) -> None:
        """AC-2.5: Config has dedup_threshold field with default value."""
        from src.nlp.yake_extractor import YAKEConfig

        config = YAKEConfig()
        assert hasattr(config, "dedup_threshold")
        assert isinstance(config.dedup_threshold, float)

    def test_config_accepts_custom_values(self) -> None:
        """AC-2.5: Config accepts custom values."""
        from src.nlp.yake_extractor import YAKEConfig

        config = YAKEConfig(top_n=5, n_gram_size=2, dedup_threshold=0.8)
        assert config.top_n == 5
        assert config.n_gram_size == 2
        assert config.dedup_threshold == 0.8


# =============================================================================
# HCE-2.5: Test extract() returns List[Tuple[str, float]]
# =============================================================================


class TestYAKEExtractorExtract:
    """HCE-2.5: Test extract() returns List[Tuple[str, float]]."""

    def test_extract_returns_list(self) -> None:
        """AC-2.3: extract() returns a list."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()
        result = extractor.extract(SAMPLE_TEXT)

        assert isinstance(result, list)

    def test_extract_returns_tuples(self) -> None:
        """AC-2.3: extract() returns list of tuples."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) > 0
        assert all(isinstance(item, tuple) for item in result)

    def test_extract_tuples_have_term_and_score(self) -> None:
        """AC-2.3: Each tuple has (term: str, score: float)."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) > 0
        for term, score in result:
            assert isinstance(term, str)
            assert isinstance(score, float)

    def test_extract_scores_are_positive(self) -> None:
        """AC-2.3: YAKE scores are positive (lower is better)."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) > 0
        for _, score in result:
            assert score >= 0.0


# =============================================================================
# HCE-2.6: Test extract() respects top_n config
# =============================================================================


class TestYAKEExtractorTopN:
    """HCE-2.6: Test extract() respects top_n config."""

    def test_extract_respects_top_n_5(self) -> None:
        """AC-2.5: Extract returns max 5 terms when top_n=5."""
        from src.nlp.yake_extractor import YAKEConfig, YAKEExtractor

        config = YAKEConfig(top_n=5)
        extractor = YAKEExtractor(config=config)
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) <= 5

    def test_extract_respects_top_n_10(self) -> None:
        """AC-2.5: Extract returns max 10 terms when top_n=10."""
        from src.nlp.yake_extractor import YAKEConfig, YAKEExtractor

        config = YAKEConfig(top_n=10)
        extractor = YAKEExtractor(config=config)
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) <= 10

    def test_extract_default_top_n_is_20(self) -> None:
        """AC-2.5: Default top_n is 20."""
        from src.nlp.yake_extractor import YAKEConfig

        config = YAKEConfig()
        assert config.top_n == 20


# =============================================================================
# HCE-2.7: Test extract() respects n_gram_size config
# =============================================================================


class TestYAKEExtractorNGramSize:
    """HCE-2.7: Test extract() respects n_gram_size config."""

    def test_extract_with_n_gram_1_returns_single_words(self) -> None:
        """AC-2.5: n_gram_size=1 returns single words only."""
        from src.nlp.yake_extractor import YAKEConfig, YAKEExtractor

        config = YAKEConfig(n_gram_size=1)
        extractor = YAKEExtractor(config=config)
        result = extractor.extract(SAMPLE_TEXT)

        # All terms should be single words (no spaces)
        for term, _ in result:
            word_count = len(term.split())
            assert word_count == 1, f"Expected single word, got '{term}'"

    def test_extract_with_n_gram_3_allows_phrases(self) -> None:
        """AC-2.5: n_gram_size=3 allows up to 3-word phrases."""
        from src.nlp.yake_extractor import YAKEConfig, YAKEExtractor

        config = YAKEConfig(n_gram_size=3)
        extractor = YAKEExtractor(config=config)
        result = extractor.extract(SAMPLE_TEXT)

        # All terms should have at most 3 words
        for term, _ in result:
            word_count = len(term.split())
            assert word_count <= 3, f"Expected max 3 words, got '{term}'"


# =============================================================================
# HCE-2.8: Test extract() handles empty text
# =============================================================================


class TestYAKEExtractorEmptyText:
    """HCE-2.8: Test extract() handles empty text."""

    def test_extract_empty_text_returns_empty_list(self) -> None:
        """AC-2.3: extract() returns empty list for empty text."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()
        result = extractor.extract(EMPTY_TEXT)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_extract_whitespace_only_returns_empty_list(self) -> None:
        """AC-2.3: extract() returns empty list for whitespace-only text."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()
        result = extractor.extract("   \n\t  ")

        assert isinstance(result, list)
        assert len(result) == 0


# =============================================================================
# HCE-2.9: Test YAKE instance is cached
# =============================================================================


class TestYAKEExtractorCaching:
    """HCE-2.9: Test YAKE instance is cached."""

    def test_yake_instance_is_cached(self) -> None:
        """AC-2.3/#12: YAKE extractor reuses same internal instance."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()

        # First call initializes
        _ = extractor.extract(SAMPLE_TEXT)
        yake_instance_1 = extractor._yake_extractor

        # Second call should reuse
        _ = extractor.extract(SAMPLE_TEXT_SHORT)
        yake_instance_2 = extractor._yake_extractor

        assert yake_instance_1 is yake_instance_2

    def test_extractor_has_private_yake_attribute(self) -> None:
        """AC-2.3/#12: Extractor has _yake_extractor for caching."""
        from src.nlp.yake_extractor import YAKEExtractor

        extractor = YAKEExtractor()
        _ = extractor.extract(SAMPLE_TEXT)

        assert hasattr(extractor, "_yake_extractor")
        assert extractor._yake_extractor is not None
