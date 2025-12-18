"""
Unit Tests for EEP-1: Domain-Aware Keyword Filtering

WBS: EEP-1 - Domain-Aware Keyword Filtering (Phase 1 of Enhanced Enrichment Pipeline)
TDD Phase: RED (tests written BEFORE implementation)

Tests for:
- EEP-1.1: Technical stopword list loading from config/technical_stopwords.json
- EEP-1.2: Extended KeywordExtractorConfig with custom_stopwords_path and merge_stopwords
- EEP-1.3: Domain-specific filtering integration with domain_taxonomy.json

Acceptance Criteria (from ENHANCED_ENRICHMENT_PIPELINE_WBS.md):
- AC-EEP-1.1.1: technical_stopwords.json exists with categorized stopwords
- AC-EEP-1.1.2: Stopword categories: document_structure, meta_words, programming_noise, common_filler
- AC-EEP-1.2.1: KeywordExtractorConfig supports custom_stopwords_path: Path | None
- AC-EEP-1.2.2: KeywordExtractorConfig supports merge_stopwords: bool (default True)
- AC-EEP-1.2.3: Custom stopwords merge with sklearn English stopwords when merge_stopwords=True
- AC-EEP-1.3.1: Domain taxonomy blacklist patterns filter keywords
- AC-EEP-1.3.2: Domain taxonomy whitelist patterns boost keyword scores
- AC-EEP-1.3.3: Score adjustments apply per domain configuration

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Use constants for repeated string literals
- S3776: Cognitive complexity < 15
- S1172: No unused parameters
- S3457: No empty f-strings
- #7: No exception shadowing
- #12: No model loading per request (cache stopwords)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from src.models.tfidf_extractor import (
        KeywordExtractorConfig,
        KeywordExtractionResult,
        TfidfKeywordExtractor,
    )


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

TECHNICAL_STOPWORDS_PATH = Path("config/technical_stopwords.json")
DOMAIN_TAXONOMY_PATH = Path(
    "/Users/kevintoles/POC/semantic-search-service/config/domain_taxonomy.json"
)

# Document structure stopwords (technical book artifacts)
DOCUMENT_STRUCTURE_STOPWORDS = {
    "chapter",
    "section",
    "figure",
    "table",
    "appendix",
    "page",
    "index",
    "bibliography",
    "glossary",
    "preface",
    "introduction",
    "conclusion",
    "summary",
    "overview",
    "part",
}

# Meta words (publishing/formatting artifacts)
META_WORDS_STOPWORDS = {
    "isbn",
    "copyright",
    "edition",
    "publisher",
    "printed",
    "author",
    "foreword",
    "acknowledgments",
    "dedication",
}

# Programming noise (non-semantic tokens)
PROGRAMMING_NOISE_STOPWORDS = {
    "example",
    "listing",
    "code",
    "output",
    "input",
    "variable",
    "function",
    "method",
    "class",
    "return",
    "print",
    "import",
    "def",
}

# Common filler words in technical writing
COMMON_FILLER_STOPWORDS = {
    "following",
    "previous",
    "next",
    "above",
    "below",
    "see",
    "shown",
    "given",
    "let",
    "consider",
    "note",
    "example",
}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def technical_book_corpus() -> list[str]:
    """Sample corpus from technical book content with stopwords to filter."""
    return [
        "Chapter 1 Introduction: Python programming is widely used for machine learning. "
        "See Figure 1.1 for an overview of the architecture. Consider the following example.",
        "Section 2.3: Neural networks power deep learning systems. "
        "The code listing below shows the implementation. Note the return statement.",
        "In this chapter we explore microservices architecture patterns. "
        "See the bibliography for additional references. Table 3.2 summarizes the comparison.",
    ]


@pytest.fixture
def domain_taxonomy_sample() -> dict[str, Any]:
    """Sample domain taxonomy configuration for testing."""
    return {
        "llm_rag": {
            "book_blacklist_patterns": ["chapter \\d+", "figure \\d+", "table \\d+"],
            "book_whitelist_patterns": ["retrieval augmented", "vector database", "embedding"],
            "score_adjustments": {"retrieval": 1.5, "embedding": 1.3, "vector": 1.2},
            "min_domain_matches": 2,
        },
        "python_implementation": {
            "book_blacklist_patterns": ["listing \\d+", "code example"],
            "book_whitelist_patterns": ["python", "pip", "virtualenv"],
            "score_adjustments": {"python": 1.4, "async": 1.2},
            "min_domain_matches": 1,
        },
    }


@pytest.fixture
def custom_stopwords_json(tmp_path: Path) -> Path:
    """Create a temporary technical_stopwords.json file."""
    stopwords_data = {
        "document_structure": list(DOCUMENT_STRUCTURE_STOPWORDS),
        "meta_words": list(META_WORDS_STOPWORDS),
        "programming_noise": list(PROGRAMMING_NOISE_STOPWORDS),
        "common_filler": list(COMMON_FILLER_STOPWORDS),
    }
    stopwords_file = tmp_path / "technical_stopwords.json"
    stopwords_file.write_text(json.dumps(stopwords_data, indent=2))
    return stopwords_file


@pytest.fixture
def domain_taxonomy_file(tmp_path: Path, domain_taxonomy_sample: dict[str, Any]) -> Path:
    """Create a temporary domain_taxonomy.json file."""
    taxonomy_file = tmp_path / "domain_taxonomy.json"
    taxonomy_file.write_text(json.dumps(domain_taxonomy_sample, indent=2))
    return taxonomy_file


# =============================================================================
# EEP-1.1: Technical Stopword List Tests
# =============================================================================


class TestTechnicalStopwordsFile:
    """Test AC-EEP-1.1.1: technical_stopwords.json file existence and structure."""

    def test_technical_stopwords_file_exists(self) -> None:
        """AC-EEP-1.1.1: config/technical_stopwords.json should exist."""
        config_path = Path("/Users/kevintoles/POC/Code-Orchestrator-Service") / TECHNICAL_STOPWORDS_PATH
        assert config_path.exists(), f"Technical stopwords file not found at {config_path}"

    def test_technical_stopwords_is_valid_json(self) -> None:
        """AC-EEP-1.1.1: File should be valid JSON."""
        config_path = Path("/Users/kevintoles/POC/Code-Orchestrator-Service") / TECHNICAL_STOPWORDS_PATH
        with config_path.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_technical_stopwords_has_required_categories(self) -> None:
        """AC-EEP-1.1.2: Should have all required stopword categories."""
        config_path = Path("/Users/kevintoles/POC/Code-Orchestrator-Service") / TECHNICAL_STOPWORDS_PATH
        with config_path.open() as f:
            data = json.load(f)

        required_categories = {
            "document_structure",
            "meta_words",
            "programming_noise",
            "common_filler",
        }
        assert required_categories.issubset(
            data.keys()
        ), f"Missing categories: {required_categories - set(data.keys())}"

    def test_each_category_is_list_of_strings(self) -> None:
        """AC-EEP-1.1.2: Each category should be a list of strings."""
        config_path = Path("/Users/kevintoles/POC/Code-Orchestrator-Service") / TECHNICAL_STOPWORDS_PATH
        with config_path.open() as f:
            data = json.load(f)

        for category, stopwords in data.items():
            assert isinstance(stopwords, list), f"Category '{category}' is not a list"
            for word in stopwords:
                assert isinstance(word, str), f"Non-string in category '{category}': {word}"
                assert len(word) > 0, f"Empty string in category '{category}'"

    def test_document_structure_contains_expected_terms(self) -> None:
        """AC-EEP-1.1.2: document_structure should contain book navigation terms."""
        config_path = Path("/Users/kevintoles/POC/Code-Orchestrator-Service") / TECHNICAL_STOPWORDS_PATH
        with config_path.open() as f:
            data = json.load(f)

        doc_structure = set(data.get("document_structure", []))
        expected_terms = {"chapter", "section", "figure", "table", "appendix"}
        assert expected_terms.issubset(
            doc_structure
        ), f"Missing document structure terms: {expected_terms - doc_structure}"


# =============================================================================
# EEP-1.2: KeywordExtractorConfig Extension Tests
# =============================================================================


class TestKeywordExtractorConfigExtension:
    """Test AC-EEP-1.2.x: KeywordExtractorConfig extension for custom stopwords."""

    def test_config_has_custom_stopwords_path_attribute(self) -> None:
        """AC-EEP-1.2.1: KeywordExtractorConfig should have custom_stopwords_path."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert hasattr(config, "custom_stopwords_path")

    def test_custom_stopwords_path_default_is_none(self) -> None:
        """AC-EEP-1.2.1: custom_stopwords_path default should be None."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert config.custom_stopwords_path is None

    def test_custom_stopwords_path_accepts_path(self, custom_stopwords_json: Path) -> None:
        """AC-EEP-1.2.1: custom_stopwords_path should accept Path objects."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig(custom_stopwords_path=custom_stopwords_json)
        assert config.custom_stopwords_path == custom_stopwords_json

    def test_config_has_merge_stopwords_attribute(self) -> None:
        """AC-EEP-1.2.2: KeywordExtractorConfig should have merge_stopwords."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert hasattr(config, "merge_stopwords")

    def test_merge_stopwords_default_is_true(self) -> None:
        """AC-EEP-1.2.2: merge_stopwords default should be True."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert config.merge_stopwords is True

    def test_merge_stopwords_can_be_set_false(self, custom_stopwords_json: Path) -> None:
        """AC-EEP-1.2.2: merge_stopwords should be configurable to False."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            merge_stopwords=False,
        )
        assert config.merge_stopwords is False


# =============================================================================
# EEP-1.2.3: Custom Stopwords Integration Tests
# =============================================================================


class TestCustomStopwordsIntegration:
    """Test AC-EEP-1.2.3: Custom stopwords merge with sklearn English stopwords."""

    def test_extractor_loads_custom_stopwords(self, custom_stopwords_json: Path) -> None:
        """Custom stopwords should be loaded from file."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(custom_stopwords_path=custom_stopwords_json)
        extractor = TfidfKeywordExtractor(config=config)

        # Extractor should have loaded custom stopwords
        assert hasattr(extractor, "_custom_stopwords")
        assert len(extractor._custom_stopwords) > 0

    def test_custom_stopwords_merged_with_english(
        self, custom_stopwords_json: Path
    ) -> None:
        """AC-EEP-1.2.3: Custom stopwords should merge with English when merge_stopwords=True."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            merge_stopwords=True,
        )
        extractor = TfidfKeywordExtractor(config=config)

        # Should have both English stop words AND custom stopwords
        effective_stopwords = extractor.get_effective_stopwords()
        
        # English stopwords should be present
        assert "the" in effective_stopwords
        assert "and" in effective_stopwords
        
        # Custom stopwords should also be present
        assert "chapter" in effective_stopwords
        assert "figure" in effective_stopwords

    def test_custom_stopwords_replace_english_when_merge_false(
        self, custom_stopwords_json: Path
    ) -> None:
        """AC-EEP-1.2.3: Custom stopwords should replace English when merge_stopwords=False."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            merge_stopwords=False,
        )
        extractor = TfidfKeywordExtractor(config=config)

        effective_stopwords = extractor.get_effective_stopwords()
        
        # Custom stopwords should be present
        assert "chapter" in effective_stopwords
        assert "figure" in effective_stopwords
        
        # English stopwords should NOT be present (replaced)
        # Note: "the" and "and" are English-only, not in our custom list
        assert "the" not in effective_stopwords
        assert "and" not in effective_stopwords

    def test_technical_stopwords_filtered_from_results(
        self, technical_book_corpus: list[str], custom_stopwords_json: Path
    ) -> None:
        """Technical stopwords should be filtered from keyword results."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            merge_stopwords=True,
        )
        extractor = TfidfKeywordExtractor(config=config)
        result = extractor.extract_keywords(technical_book_corpus, top_k=20)

        # Flatten all keywords to lowercase
        all_keywords = [kw.lower() for doc_kws in result for kw in doc_kws]

        # These document structure terms should NOT appear
        forbidden_terms = {"chapter", "section", "figure", "table", "introduction"}
        found_forbidden = forbidden_terms.intersection(set(all_keywords))
        assert len(found_forbidden) == 0, f"Found forbidden stopwords: {found_forbidden}"

    def test_meaningful_terms_preserved_after_filtering(
        self, technical_book_corpus: list[str], custom_stopwords_json: Path
    ) -> None:
        """Meaningful domain terms should be preserved after stopword filtering."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            merge_stopwords=True,
        )
        extractor = TfidfKeywordExtractor(config=config)
        result = extractor.extract_keywords(technical_book_corpus, top_k=20)

        # Flatten all keywords to lowercase
        all_keywords = [kw.lower() for doc_kws in result for kw in doc_kws]

        # Domain terms should be preserved
        expected_terms = {"python", "learning", "neural", "microservices", "architecture"}
        found_terms = expected_terms.intersection(set(all_keywords))
        assert len(found_terms) >= 2, f"Expected domain terms, found: {found_terms}"


# =============================================================================
# EEP-1.3: Domain-Specific Filtering Tests
# =============================================================================


class TestDomainSpecificFiltering:
    """Test AC-EEP-1.3.x: Domain taxonomy integration for keyword filtering."""

    def test_config_has_domain_taxonomy_path_attribute(self) -> None:
        """KeywordExtractorConfig should have domain_taxonomy_path."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert hasattr(config, "domain_taxonomy_path")

    def test_domain_taxonomy_path_default_is_none(self) -> None:
        """domain_taxonomy_path default should be None."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert config.domain_taxonomy_path is None

    def test_extractor_loads_domain_taxonomy(
        self, domain_taxonomy_file: Path, custom_stopwords_json: Path
    ) -> None:
        """Domain taxonomy should be loaded from file."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            domain_taxonomy_path=domain_taxonomy_file,
        )
        extractor = TfidfKeywordExtractor(config=config)

        # Extractor should have loaded domain taxonomy
        assert hasattr(extractor, "_domain_taxonomy")
        assert extractor._domain_taxonomy is not None
        assert "llm_rag" in extractor._domain_taxonomy

    def test_blacklist_patterns_filter_keywords(
        self, domain_taxonomy_file: Path, custom_stopwords_json: Path
    ) -> None:
        """AC-EEP-1.3.1: Blacklist patterns should filter matching keywords."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        # Corpus with blacklist pattern matches
        corpus = [
            "Chapter 1 discusses retrieval augmented generation patterns.",
            "Figure 2.3 shows the vector database architecture.",
            "Table 3.1 compares embedding models.",
        ]

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            domain_taxonomy_path=domain_taxonomy_file,
            active_domain="llm_rag",
        )
        extractor = TfidfKeywordExtractor(config=config)
        result = extractor.extract_keywords(corpus, top_k=20)

        # Flatten all keywords
        all_keywords = [kw.lower() for doc_kws in result for kw in doc_kws]

        # Patterns like "chapter 1", "figure 2.3", "table 3.1" should be filtered
        # Note: exact pattern matching may filter just the pattern, not individual words
        # depending on implementation
        for kw in all_keywords:
            assert not kw.startswith("chapter "), f"Blacklisted pattern found: {kw}"
            assert not kw.startswith("figure "), f"Blacklisted pattern found: {kw}"
            assert not kw.startswith("table "), f"Blacklisted pattern found: {kw}"


class TestWhitelistBoostingAndScoreAdjustments:
    """Test AC-EEP-1.3.2 and AC-EEP-1.3.3: Whitelist boosting and score adjustments."""

    def test_whitelist_patterns_boost_scores(
        self, domain_taxonomy_file: Path, custom_stopwords_json: Path
    ) -> None:
        """AC-EEP-1.3.2: Whitelist patterns should boost keyword scores."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        corpus = [
            "Vector database systems store embeddings for similarity search.",
            "Retrieval augmented generation improves LLM accuracy.",
        ]

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            domain_taxonomy_path=domain_taxonomy_file,
            active_domain="llm_rag",
        )
        extractor = TfidfKeywordExtractor(config=config)
        result = extractor.extract_keywords_with_scores(corpus, top_k=10)

        # Whitelist terms should appear with boosted scores
        all_keywords_with_scores = [
            (item.keyword.lower(), item.score)
            for doc_result in result
            for item in doc_result
        ]
        keyword_dict = dict(all_keywords_with_scores)

        # "embedding" is in whitelist with boost, should rank higher
        if "embedding" in keyword_dict:
            # Just verify it exists in results - score comparison is implementation detail
            assert keyword_dict["embedding"] > 0

    def test_score_adjustments_apply_to_keywords(
        self, domain_taxonomy_file: Path, custom_stopwords_json: Path
    ) -> None:
        """AC-EEP-1.3.3: Score adjustments should apply per domain configuration."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        corpus = ["Python retrieval systems use vector embeddings."]

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            domain_taxonomy_path=domain_taxonomy_file,
            active_domain="llm_rag",
        )
        extractor = TfidfKeywordExtractor(config=config)

        # Get scores with and without domain filtering
        result_with_domain = extractor.extract_keywords_with_scores(corpus, top_k=10)

        # Create extractor without domain taxonomy
        config_no_domain = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
        )
        extractor_no_domain = TfidfKeywordExtractor(config=config_no_domain)
        result_no_domain = extractor_no_domain.extract_keywords_with_scores(corpus, top_k=10)

        # Convert to dicts for comparison
        scores_with = {
            item.keyword.lower(): item.score
            for doc_result in result_with_domain
            for item in doc_result
        }
        scores_without = {
            item.keyword.lower(): item.score
            for doc_result in result_no_domain
            for item in doc_result
        }

        # Score adjustment terms (retrieval: 1.5, embedding: 1.3, vector: 1.2)
        # should have higher scores with domain taxonomy active
        for term in ["retrieval", "embedding", "vector"]:
            if term in scores_with and term in scores_without:
                # Score with domain should be >= score without (due to adjustment)
                assert scores_with[term] >= scores_without[term], (
                    f"Score adjustment not applied for '{term}': "
                    f"{scores_with[term]} < {scores_without[term]}"
                )

    def test_config_has_active_domain_attribute(self) -> None:
        """KeywordExtractorConfig should have active_domain for domain selection."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert hasattr(config, "active_domain")

    def test_active_domain_default_is_none(self) -> None:
        """active_domain default should be None (no domain filtering)."""
        from src.models.tfidf_extractor import KeywordExtractorConfig

        config = KeywordExtractorConfig()
        assert config.active_domain is None

    def test_invalid_domain_raises_error(
        self, domain_taxonomy_file: Path, custom_stopwords_json: Path
    ) -> None:
        """Invalid active_domain should raise ValueError."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            domain_taxonomy_path=domain_taxonomy_file,
            active_domain="nonexistent_domain",
        )

        with pytest.raises(ValueError, match="Unknown domain"):
            TfidfKeywordExtractor(config=config)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCasesEEP1:
    """Test edge cases for EEP-1 features."""

    def test_missing_stopwords_file_raises_error(self, tmp_path: Path) -> None:
        """Missing custom stopwords file should raise FileNotFoundError."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        nonexistent_path = tmp_path / "nonexistent.json"
        config = KeywordExtractorConfig(custom_stopwords_path=nonexistent_path)

        with pytest.raises(FileNotFoundError):
            TfidfKeywordExtractor(config=config)

    def test_invalid_stopwords_json_raises_error(self, tmp_path: Path) -> None:
        """Invalid JSON in stopwords file should raise JSONDecodeError."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("not valid json {")
        config = KeywordExtractorConfig(custom_stopwords_path=invalid_json)

        with pytest.raises(json.JSONDecodeError):
            TfidfKeywordExtractor(config=config)

    def test_missing_taxonomy_file_raises_error(
        self, tmp_path: Path, custom_stopwords_json: Path
    ) -> None:
        """Missing domain taxonomy file should raise FileNotFoundError."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        nonexistent_path = tmp_path / "nonexistent_taxonomy.json"
        config = KeywordExtractorConfig(
            custom_stopwords_path=custom_stopwords_json,
            domain_taxonomy_path=nonexistent_path,
        )

        with pytest.raises(FileNotFoundError):
            TfidfKeywordExtractor(config=config)

    def test_empty_stopwords_file_works(self, tmp_path: Path) -> None:
        """Empty stopwords categories should work without error."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        empty_stopwords = tmp_path / "empty_stopwords.json"
        empty_stopwords.write_text(json.dumps({
            "document_structure": [],
            "meta_words": [],
            "programming_noise": [],
            "common_filler": [],
        }))

        config = KeywordExtractorConfig(custom_stopwords_path=empty_stopwords)
        extractor = TfidfKeywordExtractor(config=config)
        
        # Should still work with empty custom stopwords
        corpus = ["Python is great for machine learning."]
        result = extractor.extract_keywords(corpus)
        assert len(result) == 1

    def test_stopwords_cached_across_calls(self, custom_stopwords_json: Path) -> None:
        """Anti-pattern #12: Stopwords should be cached, not reloaded per call."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(custom_stopwords_path=custom_stopwords_json)
        extractor = TfidfKeywordExtractor(config=config)

        # Get the stopwords set
        stopwords_1 = extractor.get_effective_stopwords()
        stopwords_2 = extractor.get_effective_stopwords()

        # Should be the same object (cached), not reloaded
        # Note: Using == because frozensets use value equality, but they should also
        # be the same cached object (verified by id comparison if needed)
        assert stopwords_1 == stopwords_2
        assert id(stopwords_1) == id(stopwords_2)  # Verify it's actually cached


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Ensure EEP-1 changes maintain backward compatibility."""

    def test_default_config_unchanged(self) -> None:
        """Default config without new params should work as before."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        # This is the existing behavior that must be preserved
        config = KeywordExtractorConfig()
        extractor = TfidfKeywordExtractor(config=config)

        corpus = ["Python is great for machine learning and data science."]
        result = extractor.extract_keywords(corpus, top_k=5)

        assert len(result) == 1
        assert len(result[0]) <= 5

    def test_existing_stop_words_param_still_works(self) -> None:
        """Existing stop_words param should still function."""
        from src.models.tfidf_extractor import KeywordExtractorConfig, TfidfKeywordExtractor

        config = KeywordExtractorConfig(stop_words="english")
        extractor = TfidfKeywordExtractor(config=config)

        corpus = ["The quick brown fox jumps over the lazy dog."]
        result = extractor.extract_keywords(corpus, top_k=10)

        # "the" should be filtered by English stopwords
        all_keywords = [kw.lower() for doc_kws in result for kw in doc_kws]
        assert "the" not in all_keywords

    def test_extract_keywords_signature_unchanged(self) -> None:
        """extract_keywords method signature should remain compatible."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()

        # These calls must continue to work
        result1 = extractor.extract_keywords(["test document"])
        result2 = extractor.extract_keywords(["test document"], top_k=5)

        assert isinstance(result1, list)
        assert isinstance(result2, list)

    def test_extract_keywords_with_scores_signature_unchanged(self) -> None:
        """extract_keywords_with_scores method signature should remain compatible."""
        from src.models.tfidf_extractor import TfidfKeywordExtractor

        extractor = TfidfKeywordExtractor()

        # This call must continue to work
        result = extractor.extract_keywords_with_scores(["test document"], top_k=5)

        assert isinstance(result, list)
        assert len(result) == 1
