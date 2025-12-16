"""
Unit Tests for TF-IDF Keyword Extractor

WBS: MSE-1.5 - Unit Tests for TF-IDF Keyword Extractor
TDD Phase: RED (tests written BEFORE implementation)

Tests for:
- TfidfKeywordExtractor class
- KeywordExtractorConfig dataclass
- KeywordExtractionResult dataclass
- extract_keywords() method
- extract_keywords_batch() method

Acceptance Criteria (from MSEP WBS):
- AC-1.1.1: TfidfExtractor class exists with extract_keywords() method
- AC-1.1.2: Uses TfidfVectorizer(max_features=5000, ngram_range=(1,2))
- AC-1.1.3: Returns top K keywords per document
- AC-1.1.4: Filters English stop words
- AC-1.1.5: Type annotations pass Mypy strict

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Constants reused from semantic_similarity_engine.py
- #2.2: Full type annotations
- #12: Single vectorizer instance per extractor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from src.models.tfidf_extractor import (
        KeywordExtractorConfig,
        KeywordExtractionResult,
        TfidfKeywordExtractor,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_corpus() -> list[str]:
    """Sample corpus for keyword extraction tests."""
    return [
        "Machine learning and deep learning are subfields of artificial intelligence. "
        "Neural networks power many modern AI systems.",
        "Python programming language is widely used for data science and machine learning. "
        "Libraries like scikit-learn and TensorFlow are popular.",
        "Natural language processing enables computers to understand human language. "
        "Text classification and sentiment analysis are common NLP tasks.",
    ]


@pytest.fixture
def single_document() -> list[str]:
    """Single document corpus for edge case testing."""
    return ["Python is a programming language used for machine learning and data science."]


@pytest.fixture
def empty_corpus() -> list[str]:
    """Empty corpus for edge case testing."""
    return []


# =============================================================================
# AC-1.1.1: TfidfExtractor class exists with extract_keywords() method
# =============================================================================


class TestTfidfKeywordExtractorExists:
    """Test that TfidfKeywordExtractor class exists and is importable."""

    def test_class_is_importable(self) -> None:
        """AC-1.1.1: TfidfKeywordExtractor class should be importable."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        assert TfidfKeywordExtractor is not None

    def test_class_is_instantiable(self) -> None:
        """AC-1.1.1: TfidfKeywordExtractor should be instantiable."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        assert extractor is not None

    def test_extract_keywords_method_exists(self) -> None:
        """AC-1.1.1: extract_keywords() method should exist."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        assert hasattr(extractor, "extract_keywords")
        assert callable(extractor.extract_keywords)


# =============================================================================
# AC-1.1.2: Uses TfidfVectorizer(max_features=5000, ngram_range=(1,2))
# =============================================================================


class TestTfidfVectorizerConfiguration:
    """Test TF-IDF vectorizer configuration per MSEP WBS requirements."""

    def test_default_max_features_is_5000(self) -> None:
        """AC-1.1.2: Default max_features should be 5000."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        assert extractor.config.max_features == 5000

    def test_default_ngram_range_is_1_2(self) -> None:
        """AC-1.1.2: Default ngram_range should be (1, 2)."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        assert extractor.config.ngram_range == (1, 2)

    def test_stop_words_is_english(self) -> None:
        """AC-1.1.2: stop_words should be 'english'."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        assert extractor.config.stop_words == "english"

    def test_custom_config_is_respected(self) -> None:
        """Custom config should override defaults."""
        from src.models.tfidf_extractor import (
            KeywordExtractorConfig,
            TfidfKeywordExtractor,
        )

        config = KeywordExtractorConfig(max_features=1000, ngram_range=(1, 3))
        extractor = TfidfKeywordExtractor(config=config)
        
        assert extractor.config.max_features == 1000
        assert extractor.config.ngram_range == (1, 3)


# =============================================================================
# AC-1.1.3: Returns top K keywords per document
# =============================================================================


class TestExtractKeywordsBasicFunctionality:
    """Test extract_keywords() returns correct structure and top-k behavior."""

    def test_returns_list_of_lists(self, sample_corpus: list[str]) -> None:
        """AC-1.1.3: Should return list of keyword lists (one per document)."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(sample_corpus)

        assert isinstance(result, list)
        assert len(result) == len(sample_corpus)
        for keywords in result:
            assert isinstance(keywords, list)

    def test_respects_top_k_parameter(self, sample_corpus: list[str]) -> None:
        """AC-1.1.3: Should return at most top_k keywords per document."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        
        for top_k in [3, 5, 10]:
            result = extractor.extract_keywords(sample_corpus, top_k=top_k)
            for keywords in result:
                assert len(keywords) <= top_k

    def test_default_top_k_is_10(self, sample_corpus: list[str]) -> None:
        """AC-1.1.3: Default top_k should be 10."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(sample_corpus)
        
        for keywords in result:
            assert len(keywords) <= 10

    def test_keywords_are_strings(self, sample_corpus: list[str]) -> None:
        """AC-1.1.3: Each keyword should be a string."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(sample_corpus)

        for keywords in result:
            for keyword in keywords:
                assert isinstance(keyword, str)
                assert len(keyword) > 0

    def test_keywords_sorted_by_tfidf_score(self, single_document: list[str]) -> None:
        """AC-1.1.3: Keywords should be sorted by TF-IDF score (descending)."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        # Get result with scores
        result_with_scores = extractor.extract_keywords_with_scores(single_document, top_k=5)
        
        # Verify scores are in descending order
        for doc_result in result_with_scores:
            scores = [item.score for item in doc_result]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# AC-1.1.4: Filters English stop words
# =============================================================================


class TestStopWordFiltering:
    """Test that English stop words are filtered from results."""

    def test_common_stop_words_not_in_results(self, sample_corpus: list[str]) -> None:
        """AC-1.1.4: Common stop words should not appear in results."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(sample_corpus)

        # Common English stop words that should be filtered
        stop_words = {"the", "and", "is", "are", "a", "an", "of", "to", "in", "for", "on", "with"}

        for keywords in result:
            for keyword in keywords:
                # Single word keywords should not be stop words
                if " " not in keyword:
                    assert keyword.lower() not in stop_words, f"Stop word '{keyword}' found in results"

    def test_meaningful_terms_preserved(self, sample_corpus: list[str]) -> None:
        """AC-1.1.4: Meaningful terms should be preserved."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(sample_corpus, top_k=20)
        
        # Flatten all keywords
        all_keywords = [kw.lower() for doc_kws in result for kw in doc_kws]
        
        # At least some of these should appear (domain-specific terms)
        expected_terms = {"learning", "machine", "python", "language", "data"}
        found_terms = expected_terms.intersection(set(all_keywords))
        
        assert len(found_terms) >= 2, f"Expected to find domain terms, got: {all_keywords}"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_corpus_returns_empty_list(self, empty_corpus: list[str]) -> None:
        """Empty corpus should return empty list."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(empty_corpus)
        
        assert result == []

    def test_single_document_corpus(self, single_document: list[str]) -> None:
        """Single document corpus should work correctly."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(single_document)
        
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_empty_document_in_corpus(self) -> None:
        """Corpus with empty document should handle gracefully."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        corpus = ["Python is great for machine learning.", "", "Data science uses Python."]
        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(corpus)
        
        assert len(result) == 3
        # Empty document should return empty keywords
        assert result[1] == []

    def test_top_k_larger_than_vocab(self, single_document: list[str]) -> None:
        """top_k larger than vocabulary should return all keywords."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(single_document, top_k=1000)
        
        # Should return whatever keywords exist, not throw error
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_top_k_zero_returns_empty(self, sample_corpus: list[str]) -> None:
        """top_k=0 should return empty lists."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(sample_corpus, top_k=0)
        
        for keywords in result:
            assert keywords == []


# =============================================================================
# KeywordExtractorConfig Dataclass Tests
# =============================================================================


class TestKeywordExtractorConfig:
    """Test KeywordExtractorConfig dataclass."""

    def test_config_is_dataclass(self) -> None:
        """Config should be a dataclass."""
        from dataclasses import is_dataclass

        from src.models.tfidf_extractor import KeywordExtractorConfig

        assert is_dataclass(KeywordExtractorConfig)

    def test_config_default_values(self) -> None:
        """Config should have correct default values."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        
        assert config.max_features == 5000
        assert config.ngram_range == (1, 2)
        assert config.stop_words == "english"
        assert config.min_df == 1
        assert config.max_df == 0.95
        assert config.default_top_k == 10


# =============================================================================
# KeywordExtractionResult Dataclass Tests
# =============================================================================


class TestKeywordExtractionResult:
    """Test KeywordExtractionResult dataclass."""

    def test_result_is_dataclass(self) -> None:
        """Result should be a dataclass."""
        from dataclasses import is_dataclass

        from src.models.tfidf_extractor import KeywordExtractionResult

        assert is_dataclass(KeywordExtractionResult)

    def test_result_has_required_fields(self) -> None:
        """Result should have keyword, score fields."""
        from src.models.tfidf_extractor import KeywordExtractionResult

        result = KeywordExtractionResult(keyword="machine learning", score=0.85)
        
        assert result.keyword == "machine learning"
        assert result.score == 0.85


# =============================================================================
# extract_keywords_with_scores() Tests
# =============================================================================


class TestExtractKeywordsWithScores:
    """Test extract_keywords_with_scores() method."""

    def test_method_exists(self) -> None:
        """extract_keywords_with_scores() method should exist."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        assert hasattr(extractor, "extract_keywords_with_scores")
        assert callable(extractor.extract_keywords_with_scores)

    def test_returns_results_with_scores(self, sample_corpus: list[str]) -> None:
        """Should return KeywordExtractionResult objects with scores."""
        from src.models.tfidf_extractor import (
            KeywordExtractionResult,
            TfidfKeywordExtractor,
        )

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords_with_scores(sample_corpus, top_k=5)

        assert len(result) == len(sample_corpus)
        for doc_results in result:
            assert isinstance(doc_results, list)
            for item in doc_results:
                assert isinstance(item, KeywordExtractionResult)
                assert isinstance(item.keyword, str)
                assert isinstance(item.score, float)
                assert 0.0 <= item.score <= 1.0

    def test_scores_are_non_negative(self, sample_corpus: list[str]) -> None:
        """All scores should be non-negative."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords_with_scores(sample_corpus)

        for doc_results in result:
            for item in doc_results:
                assert item.score >= 0.0


# =============================================================================
# Type Annotation Tests (AC-1.1.5)
# =============================================================================


class TestTypeAnnotations:
    """Test type annotations are correct for Mypy strict mode."""

    def test_extract_keywords_return_type(self, sample_corpus: list[str]) -> None:
        """AC-1.1.5: extract_keywords() should have correct return type."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()
        result = extractor.extract_keywords(sample_corpus)

        # Type should be list[list[str]]
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], list)
            if result[0]:
                assert isinstance(result[0][0], str)

    def test_config_property_type(self) -> None:
        """AC-1.1.5: config property should return KeywordExtractorConfig."""
        from src.models.tfidf_extractor import (
            KeywordExtractorConfig,
            TfidfKeywordExtractor,
        )

        extractor = TfidfKeywordExtractor()
        assert isinstance(extractor.config, KeywordExtractorConfig)
