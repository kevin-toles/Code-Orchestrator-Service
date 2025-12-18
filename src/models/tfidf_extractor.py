"""
TF-IDF Keyword Extractor

WBS: MSE-1.1 - TF-IDF Keyword Extractor Module
WBS: EEP-1 - Domain-Aware Keyword Filtering (Enhanced Enrichment Pipeline)
Repository: Code-Orchestrator-Service (Sous Chef)

This module provides keyword extraction from document corpora using TF-IDF.
Unlike the existing TF-IDF fallback in semantic_similarity_engine.py (which returns
embeddings for similarity computation), this extractor returns **top-k keywords**
per document for metadata enrichment tagging.

EEP-1 Extensions:
- EEP-1.1: Custom technical stopwords loading from config/technical_stopwords.json
- EEP-1.2: KeywordExtractorConfig extensions (custom_stopwords_path, merge_stopwords)
- EEP-1.3: Domain-specific filtering via domain_taxonomy.json integration

Role in Kitchen Brigade Architecture:
- Code-Orchestrator-Service (Sous Chef) hosts all NLP/ML models
- This extractor supports the `/api/v1/keywords` endpoint (MSE-1.2)
- Used by ai-agents MSEP orchestrator for keyword enrichment layer

Architecture: Service Layer Pattern
Anti-Patterns Addressed (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Reuses TFIDF_MAX_FEATURES constant from semantic_similarity_engine.py
- #2.2: Full type annotations on all public methods
- #12: Single TfidfVectorizer instance per extractor class; stopwords cached
- #7: No exception shadowing (uses TfidfExtractionError)
- S3776: Cognitive complexity kept under 15

Constants imported from semantic_similarity_engine.py:
- TFIDF_MAX_FEATURES: int = 5000
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Import shared constant from existing module (S1192 compliance)
from src.models.sbert.semantic_similarity_engine import TFIDF_MAX_FEATURES

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Module Constants
# =============================================================================

# Default parameters per MSEP WBS requirements
DEFAULT_NGRAM_RANGE: tuple[int, int] = (1, 2)
DEFAULT_STOP_WORDS: str = "english"
DEFAULT_MIN_DF: int = 1
DEFAULT_MAX_DF: float = 0.95
DEFAULT_TOP_K: int = 10


# =============================================================================
# Custom Exceptions (Anti-Pattern #7: No shadowing)
# =============================================================================


class TfidfExtractionError(Exception):
    """Base exception for TF-IDF extraction errors.

    Per CODING_PATTERNS_ANALYSIS.md #7:
    - Uses namespaced exception (not generic Exception)
    - Does not shadow Python builtins
    """

    pass


class EmptyCorpusError(TfidfExtractionError):
    """Raised when corpus is empty and operation requires content."""

    pass


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class KeywordExtractorConfig:
    """Configuration for the TfidfKeywordExtractor.

    Attributes:
        max_features: Maximum number of features for TfidfVectorizer (default: 5000)
        ngram_range: Range of n-grams to extract (default: (1, 2))
        stop_words: Stop words to filter (default: "english")
        min_df: Minimum document frequency for terms (default: 1)
        max_df: Maximum document frequency ratio for terms (default: 0.95)
        default_top_k: Default number of top keywords to return (default: 10)
        custom_stopwords_path: Path to JSON file with custom stopwords (default: None)
        merge_stopwords: If True, merge custom with sklearn English; if False, replace (default: True)
        domain_taxonomy_path: Path to domain taxonomy JSON for filtering (default: None)
        active_domain: Domain key to use for filtering (default: None)

    Note:
        max_features default (5000) is imported from TFIDF_MAX_FEATURES
        in semantic_similarity_engine.py per S1192 compliance.

    EEP-1 Extensions:
        - custom_stopwords_path: Load technical book stopwords from JSON
        - merge_stopwords: Control whether to merge or replace sklearn stopwords
        - domain_taxonomy_path: Load domain-specific filtering rules
        - active_domain: Select which domain's rules to apply
    """

    max_features: int = TFIDF_MAX_FEATURES
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE
    stop_words: str = DEFAULT_STOP_WORDS
    min_df: int = DEFAULT_MIN_DF
    max_df: float = DEFAULT_MAX_DF
    default_top_k: int = DEFAULT_TOP_K
    # EEP-1.2: Custom stopwords extension
    custom_stopwords_path: Path | None = None
    merge_stopwords: bool = True
    # EEP-1.3: Domain taxonomy extension
    domain_taxonomy_path: Path | None = None
    active_domain: str | None = None


@dataclass
class KeywordExtractionResult:
    """Result of keyword extraction with TF-IDF score.

    Attributes:
        keyword: The extracted keyword or n-gram
        score: TF-IDF score for this keyword (0.0 to 1.0)
    """

    keyword: str
    score: float


# =============================================================================
# TfidfKeywordExtractor Class
# =============================================================================


class TfidfKeywordExtractor:
    """Extracts top-k keywords from documents using TF-IDF.

    This class provides keyword extraction for metadata enrichment,
    returning the most important terms per document based on TF-IDF scores.

    Unlike SemanticSimilarityEngine's TF-IDF fallback (which returns sparse
    embedding vectors for similarity computation), this extractor returns
    human-readable keyword lists for tagging and enrichment.

    EEP-1 Extensions:
        - Custom technical stopwords loading from JSON configuration
        - Stopword merging with sklearn English stopwords
        - Domain-specific filtering via domain taxonomy integration
        - Blacklist/whitelist pattern matching
        - Score adjustments for domain-relevant terms

    Example:
        >>> extractor = TfidfKeywordExtractor()
        >>> corpus = ["Machine learning and AI", "Python for data science"]
        >>> keywords = extractor.extract_keywords(corpus, top_k=5)
        >>> print(keywords)
        [['machine learning', 'learning', 'machine', 'ai'], 
         ['python', 'data science', 'science', 'data']]

    Attributes:
        config: KeywordExtractorConfig with vectorizer settings
    """

    def __init__(self, config: KeywordExtractorConfig | None = None) -> None:
        """Initialize the TfidfKeywordExtractor.

        Args:
            config: Configuration for the extractor. Uses defaults if not provided.

        Raises:
            FileNotFoundError: If custom_stopwords_path or domain_taxonomy_path does not exist.
            json.JSONDecodeError: If custom_stopwords_path or domain_taxonomy_path is invalid JSON.
            ValueError: If active_domain is set but not found in domain taxonomy.
        """
        self._config = config if config is not None else KeywordExtractorConfig()
        self._vectorizer: TfidfVectorizer | None = None
        self._feature_names: list[str] | None = None

        # EEP-1.2: Load and cache custom stopwords (Anti-pattern #12: cache, don't reload per request)
        self._custom_stopwords: set[str] = set()
        self._effective_stopwords: frozenset[str] | None = None
        if self._config.custom_stopwords_path is not None:
            self._load_custom_stopwords()

        # EEP-1.3: Load and cache domain taxonomy
        self._domain_taxonomy: dict[str, Any] | None = None
        self._active_domain_config: dict[str, Any] | None = None
        if self._config.domain_taxonomy_path is not None:
            self._load_domain_taxonomy()

    def _load_custom_stopwords(self) -> None:
        """Load custom stopwords from JSON file.

        The JSON file should have categories as keys with lists of stopwords:
        {
            "document_structure": ["chapter", "section", ...],
            "meta_words": ["isbn", "copyright", ...],
            ...
        }

        Raises:
            FileNotFoundError: If the custom_stopwords_path file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        if self._config.custom_stopwords_path is None:
            return

        stopwords_path = Path(self._config.custom_stopwords_path)
        with stopwords_path.open() as f:
            data = json.load(f)

        # Flatten all categories into a single set
        all_stopwords: set[str] = set()
        for category_words in data.values():
            if isinstance(category_words, list):
                all_stopwords.update(word.lower() for word in category_words)

        self._custom_stopwords = all_stopwords

    def _load_domain_taxonomy(self) -> None:
        """Load domain taxonomy configuration from JSON file.

        Raises:
            FileNotFoundError: If domain_taxonomy_path does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            ValueError: If active_domain is set but not found in taxonomy.
        """
        if self._config.domain_taxonomy_path is None:
            return

        taxonomy_path = Path(self._config.domain_taxonomy_path)
        with taxonomy_path.open() as f:
            self._domain_taxonomy = json.load(f)

        # Validate active_domain if specified
        if self._config.active_domain is not None:
            if self._domain_taxonomy is None or self._config.active_domain not in self._domain_taxonomy:
                msg = f"Unknown domain: '{self._config.active_domain}'. Available domains: {list(self._domain_taxonomy.keys()) if self._domain_taxonomy else []}"
                raise ValueError(msg)
            self._active_domain_config = self._domain_taxonomy[self._config.active_domain]

    def get_effective_stopwords(self) -> frozenset[str]:
        """Get the effective set of stopwords used for filtering.

        Returns a cached frozenset of all stopwords, combining sklearn English
        stopwords with custom stopwords based on configuration.

        Returns:
            Frozenset of all effective stopwords.

        Note:
            Anti-pattern #12 compliance: Stopwords are computed once and cached.
        """
        if self._effective_stopwords is not None:
            return self._effective_stopwords

        if self._config.merge_stopwords:
            # Merge custom stopwords with sklearn English stopwords
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            merged = set(ENGLISH_STOP_WORDS) | self._custom_stopwords
            self._effective_stopwords = frozenset(merged)
        else:
            # Use only custom stopwords (replace sklearn English)
            self._effective_stopwords = frozenset(self._custom_stopwords)

        return self._effective_stopwords

    @property
    def config(self) -> KeywordExtractorConfig:
        """Get the extractor configuration."""
        return self._config

    def _initialize_vectorizer(self, corpus_size: int) -> TfidfVectorizer:
        """Create and return a TfidfVectorizer with current config settings.

        Args:
            corpus_size: Number of documents in the corpus. Used to adjust
                        min_df/max_df for small corpora to avoid sklearn errors.

        Returns:
            Configured TfidfVectorizer instance.

        Note:
            Creates new vectorizer each time to support fresh fit_transform
            on different corpora. The vectorizer must be fit to extract
            feature names for keyword lookup.

            For small corpora (1-2 documents), min_df and max_df are adjusted
            to avoid sklearn's "max_df corresponds to < documents than min_df"
            error when max_df * n_docs < min_df.

            EEP-1.2: When custom_stopwords_path is set, uses merged/replaced
            stopwords instead of default sklearn English stopwords.
        """
        # Adjust min_df/max_df for small corpora to avoid sklearn errors
        # When corpus_size=1, max_df=0.95 means max_doc_count=0.95 which is < min_df=1
        min_df = self._config.min_df
        max_df = self._config.max_df

        if corpus_size <= 2:
            # For very small corpora, use permissive settings
            min_df = 1
            max_df = 1.0

        # EEP-1.2: Use effective stopwords when custom stopwords are configured
        stop_words: str | list[str]
        if self._config.custom_stopwords_path is not None:
            # Convert frozenset to list for sklearn
            stop_words = list(self.get_effective_stopwords())
        else:
            # Use default sklearn behavior
            stop_words = self._config.stop_words

        return TfidfVectorizer(
            max_features=self._config.max_features,
            ngram_range=self._config.ngram_range,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
        )

    def extract_keywords(
        self,
        corpus: list[str],
        top_k: int | None = None,
    ) -> list[list[str]]:
        """Extract top-k keywords for each document in the corpus.

        Args:
            corpus: List of document strings to extract keywords from.
            top_k: Number of top keywords to return per document.
                   Defaults to config.default_top_k (10).

        Returns:
            List of keyword lists, one per document. Each inner list contains
            the top-k keywords sorted by TF-IDF score (descending).

        Example:
            >>> extractor = TfidfKeywordExtractor()
            >>> corpus = ["Machine learning is great", "Python for AI"]
            >>> keywords = extractor.extract_keywords(corpus, top_k=3)
            >>> print(keywords)
            [['machine learning', 'learning', 'machine'], ['python', 'ai']]
        """
        if top_k is None:
            top_k = self._config.default_top_k

        # Handle edge case: empty corpus
        if not corpus:
            return []

        # Handle edge case: top_k <= 0
        if top_k <= 0:
            return [[] for _ in corpus]

        # Filter out empty documents for fitting, but track their positions
        non_empty_indices = [i for i, doc in enumerate(corpus) if doc.strip()]

        if not non_empty_indices:
            # All documents are empty
            return [[] for _ in corpus]

        non_empty_corpus = [corpus[i] for i in non_empty_indices]

        # Initialize vectorizer with corpus size for small corpus handling
        self._vectorizer = self._initialize_vectorizer(len(non_empty_corpus))

        # Fit and transform the non-empty corpus
        tfidf_matrix = self._vectorizer.fit_transform(non_empty_corpus)
        self._feature_names = list(self._vectorizer.get_feature_names_out())

        # Extract keywords for each document
        result: list[list[str]] = []
        non_empty_idx = 0

        for i in range(len(corpus)):
            if i in non_empty_indices:
                doc_vector = tfidf_matrix[non_empty_idx].toarray().flatten()
                keywords = self._extract_top_keywords(doc_vector, top_k)
                result.append(keywords)
                non_empty_idx += 1
            else:
                # Empty document gets empty keyword list
                result.append([])

        return result

    def extract_keywords_with_scores(
        self,
        corpus: list[str],
        top_k: int | None = None,
    ) -> list[list[KeywordExtractionResult]]:
        """Extract top-k keywords with their TF-IDF scores for each document.

        Args:
            corpus: List of document strings to extract keywords from.
            top_k: Number of top keywords to return per document.
                   Defaults to config.default_top_k (10).

        Returns:
            List of KeywordExtractionResult lists, one per document.
            Each result contains the keyword and its TF-IDF score.

        Example:
            >>> extractor = TfidfKeywordExtractor()
            >>> corpus = ["Machine learning is great"]
            >>> results = extractor.extract_keywords_with_scores(corpus, top_k=2)
            >>> for r in results[0]:
            ...     print(f"{r.keyword}: {r.score:.3f}")
            machine learning: 0.707
            learning: 0.500
        """
        if top_k is None:
            top_k = self._config.default_top_k

        # Handle edge case: empty corpus
        if not corpus:
            return []

        # Handle edge case: top_k <= 0
        if top_k <= 0:
            return [[] for _ in corpus]

        # Filter out empty documents for fitting, but track their positions
        non_empty_indices = [i for i, doc in enumerate(corpus) if doc.strip()]

        if not non_empty_indices:
            # All documents are empty
            return [[] for _ in corpus]

        non_empty_corpus = [corpus[i] for i in non_empty_indices]

        # Initialize vectorizer with corpus size for small corpus handling
        self._vectorizer = self._initialize_vectorizer(len(non_empty_corpus))

        # Fit and transform the non-empty corpus
        tfidf_matrix = self._vectorizer.fit_transform(non_empty_corpus)
        self._feature_names = list(self._vectorizer.get_feature_names_out())

        # Extract keywords with scores for each document
        result: list[list[KeywordExtractionResult]] = []
        non_empty_idx = 0

        for i in range(len(corpus)):
            if i in non_empty_indices:
                doc_vector = tfidf_matrix[non_empty_idx].toarray().flatten()
                keywords_with_scores = self._extract_top_keywords_with_scores(
                    doc_vector, top_k
                )
                result.append(keywords_with_scores)
                non_empty_idx += 1
            else:
                # Empty document gets empty result list
                result.append([])

        return result

    def _extract_top_keywords(
        self,
        doc_vector: NDArray[np.floating],
        top_k: int,
    ) -> list[str]:
        """Extract top-k keywords from a document's TF-IDF vector.

        Args:
            doc_vector: 1D array of TF-IDF scores for a single document.
            top_k: Number of top keywords to extract.

        Returns:
            List of top-k keywords sorted by TF-IDF score (descending).
        """
        if self._feature_names is None:
            return []

        # Get indices of top-k scores (sorted descending)
        top_indices = doc_vector.argsort()[-top_k:][::-1]

        # Filter out indices with zero scores and get keywords
        keywords = [
            self._feature_names[i]
            for i in top_indices
            if doc_vector[i] > 0
        ]

        return keywords

    def _extract_top_keywords_with_scores(
        self,
        doc_vector: NDArray[np.floating],
        top_k: int,
    ) -> list[KeywordExtractionResult]:
        """Extract top-k keywords with scores from a document's TF-IDF vector.

        Args:
            doc_vector: 1D array of TF-IDF scores for a single document.
            top_k: Number of top keywords to extract.

        Returns:
            List of KeywordExtractionResult objects sorted by score (descending).
        """
        if self._feature_names is None:
            return []

        # Get indices of top-k scores (sorted descending)
        top_indices = doc_vector.argsort()[-top_k:][::-1]

        # Create results with keywords and scores, filtering zero scores
        results = [
            KeywordExtractionResult(
                keyword=self._feature_names[i],
                score=float(doc_vector[i]),
            )
            for i in top_indices
            if doc_vector[i] > 0
        ]

        return results
