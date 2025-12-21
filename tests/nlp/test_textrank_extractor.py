"""Tests for TextRankExtractor - HCE-2.13 through HCE-2.17.

TDD RED Phase: Tests written before implementation.
These tests MUST FAIL initially (no implementation exists).

WBS Reference: HCE-2.13 through HCE-2.20
AC Reference: AC-2.2, AC-2.4, AC-2.6

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Small, focused test methods
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
and speech understanding. These technologies are transforming many industries.
"""

SAMPLE_TEXT_SHORT: Final[str] = "AI models process data efficiently."

EMPTY_TEXT: Final[str] = ""


# =============================================================================
# HCE-2.13: Test TextRankExtractor can be imported
# =============================================================================


class TestTextRankExtractorImport:
    """HCE-2.13: Test TextRankExtractor can be imported."""

    def test_textrank_extractor_can_be_imported(self) -> None:
        """AC-2.4: TextRankExtractor class exists and can be imported."""
        from src.nlp.textrank_extractor import TextRankExtractor

        assert TextRankExtractor is not None

    def test_textrank_config_can_be_imported(self) -> None:
        """AC-2.6: TextRankConfig dataclass exists and can be imported."""
        from src.nlp.textrank_extractor import TextRankConfig

        assert TextRankConfig is not None


# =============================================================================
# HCE-2.14: Test TextRankConfig has required fields
# =============================================================================


class TestTextRankConfig:
    """HCE-2.14: Test TextRankConfig has required fields."""

    def test_config_has_words_field(self) -> None:
        """AC-2.6: Config has words field with default value."""
        from src.nlp.textrank_extractor import TextRankConfig

        config = TextRankConfig()
        assert hasattr(config, "words")
        assert isinstance(config.words, int)

    def test_config_has_split_field(self) -> None:
        """AC-2.6: Config has split field with default value."""
        from src.nlp.textrank_extractor import TextRankConfig

        config = TextRankConfig()
        assert hasattr(config, "split")
        assert isinstance(config.split, bool)

    def test_config_default_words_is_20(self) -> None:
        """AC-2.6: Default words is 20."""
        from src.nlp.textrank_extractor import TextRankConfig

        config = TextRankConfig()
        assert config.words == 20

    def test_config_accepts_custom_values(self) -> None:
        """AC-2.6: Config accepts custom values."""
        from src.nlp.textrank_extractor import TextRankConfig

        config = TextRankConfig(words=10, split=False)
        assert config.words == 10
        assert config.split is False


# =============================================================================
# HCE-2.15: Test extract() returns List[str]
# =============================================================================


class TestTextRankExtractorExtract:
    """HCE-2.15: Test extract() returns List[str]."""

    def test_extract_returns_list(self) -> None:
        """AC-2.4: extract() returns a list."""
        from src.nlp.textrank_extractor import TextRankExtractor

        extractor = TextRankExtractor()
        result = extractor.extract(SAMPLE_TEXT)

        assert isinstance(result, list)

    def test_extract_returns_strings(self) -> None:
        """AC-2.4: extract() returns list of strings."""
        from src.nlp.textrank_extractor import TextRankExtractor

        extractor = TextRankExtractor()
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) > 0
        assert all(isinstance(item, str) for item in result)

    def test_extract_returns_nonempty_strings(self) -> None:
        """AC-2.4: Extracted terms are non-empty strings."""
        from src.nlp.textrank_extractor import TextRankExtractor

        extractor = TextRankExtractor()
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) > 0
        assert all(len(item.strip()) > 0 for item in result)


# =============================================================================
# HCE-2.16: Test extract() respects words config
# =============================================================================


class TestTextRankExtractorWordsConfig:
    """HCE-2.16: Test extract() respects words config."""

    def test_extract_respects_words_5(self) -> None:
        """AC-2.6: Extract returns max 5 terms when words=5."""
        from src.nlp.textrank_extractor import TextRankConfig, TextRankExtractor

        config = TextRankConfig(words=5)
        extractor = TextRankExtractor(config=config)
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) <= 5

    def test_extract_respects_words_10(self) -> None:
        """AC-2.6: Extract returns max 10 terms when words=10."""
        from src.nlp.textrank_extractor import TextRankConfig, TextRankExtractor

        config = TextRankConfig(words=10)
        extractor = TextRankExtractor(config=config)
        result = extractor.extract(SAMPLE_TEXT)

        assert len(result) <= 10


# =============================================================================
# HCE-2.17: Test extract() handles empty text
# =============================================================================


class TestTextRankExtractorEmptyText:
    """HCE-2.17: Test extract() handles empty text."""

    def test_extract_empty_text_returns_empty_list(self) -> None:
        """AC-2.4: extract() returns empty list for empty text."""
        from src.nlp.textrank_extractor import TextRankExtractor

        extractor = TextRankExtractor()
        result = extractor.extract(EMPTY_TEXT)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_extract_whitespace_only_returns_empty_list(self) -> None:
        """AC-2.4: extract() returns empty list for whitespace-only text."""
        from src.nlp.textrank_extractor import TextRankExtractor

        extractor = TextRankExtractor()
        result = extractor.extract("   \n\t  ")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_extract_short_text_handles_gracefully(self) -> None:
        """AC-2.4: extract() handles short text without error."""
        from src.nlp.textrank_extractor import TextRankExtractor

        extractor = TextRankExtractor()
        result = extractor.extract(SAMPLE_TEXT_SHORT)

        assert isinstance(result, list)
        # May return empty or partial results for very short text
