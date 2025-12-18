"""
Semantic Similarity Engine using Sentence Transformers.

This module provides semantic similarity computation for chapter content,
enabling intelligent cross-referencing and related content discovery.
It gracefully falls back to TF-IDF when Sentence Transformers is unavailable.

Migrated from: llm-document-enhancer/workflows/metadata_enrichment/scripts/
Per: SBERT_EXTRACTION_MIGRATION_WBS.md Phase M1

Role in Kitchen Brigade Architecture:
- Code-Orchestrator-Service (Sous Chef) hosts all understanding models
- SBERT translates NL requirements from LLM Gateway
- Computes similar_chapters for cross-book referencing
- Provides embeddings for semantic search operations

Architecture: Service Layer Pattern (Architecture Patterns Ch. 4)
Anti-Patterns Addressed:
- S1192: Model name extracted to DEFAULT_MODEL_NAME constant
- #7: No exception shadowing (custom exceptions namespaced)
- #12: Embedding cache for performance
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Module constants per S1192 (no duplicated literals)
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS: int = 384  # MiniLM-L6-v2 output dimensions
DEFAULT_SIMILARITY_THRESHOLD: float = 0.0
DEFAULT_TOP_K: int = 5
TFIDF_MAX_FEATURES: int = 5000

# Try to import sentence-transformers, fall back to TF-IDF if unavailable
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[misc, assignment]

# TF-IDF fallback imports - intentionally after sentence-transformers try/except
# to ensure graceful degradation pattern. E402 suppressed per PEP 8 exception.
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402


@dataclass
class SimilarityConfig:
    """Configuration for the SemanticSimilarityEngine.

    Attributes:
        model_name: Sentence Transformer model name (default: all-MiniLM-L6-v2)
        similarity_threshold: Minimum similarity score to consider chapters related
        top_k: Maximum number of similar chapters to return
        use_cache: Whether to cache embeddings for reuse
        fallback_to_tfidf: Whether to use TF-IDF when Sentence Transformers unavailable
    """

    model_name: str = DEFAULT_MODEL_NAME
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    top_k: int = DEFAULT_TOP_K
    use_cache: bool = True
    fallback_to_tfidf: bool = True


@dataclass
class SimilarityResult:
    """Result of a similarity computation between chapters.

    Attributes:
        book: Source book filename
        chapter: Chapter number
        title: Chapter title
        score: Cosine similarity score (0.0 to 1.0)
        method: Method used for similarity (sentence_transformers or tfidf)
    """

    book: str
    chapter: int
    title: str
    score: float
    method: str = "sentence_transformers"


class SemanticSimilarityEngine:
    """Engine for computing semantic similarity between chapter contents.

    Uses Sentence Transformers for high-quality semantic embeddings,
    with graceful fallback to TF-IDF when the library is unavailable.

    Example:
        >>> engine = SemanticSimilarityEngine()
        >>> corpus = ["Chapter 1 about Python", "Chapter 2 about decorators"]
        >>> embeddings = engine.compute_embeddings(corpus)
        >>> index = [{"book": "b.json", "chapter": 1, "title": "Ch1"}, ...]
        >>> similar = engine.find_similar(0, embeddings, index, top_k=2)
    """

    def __init__(
        self,
        config: SimilarityConfig | None = None,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Initialize the SemanticSimilarityEngine.

        Args:
            config: Configuration options. Uses defaults if not provided.
            model_name: Name of the Sentence Transformer model (deprecated, use config).
        """
        # Support both patterns: config object OR model_name string
        if config is not None:
            self.config = config
            self.model_name = config.model_name
        else:
            self.config = SimilarityConfig(model_name=model_name)
            self.model_name = model_name

        self._model: SentenceTransformer | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None  # Alias for backward compat
        self._embedding_cache: dict[str, NDArray[np.float64]] = {}
        self._using_fallback: bool = False

        # Initialize the appropriate model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the embedding model (Sentence Transformers or TF-IDF fallback)."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                self._using_fallback = False
            except Exception:
                # Model loading failed, use fallback
                if self.config.fallback_to_tfidf:
                    self._setup_tfidf_fallback()
                else:
                    raise
        elif self.config.fallback_to_tfidf:
            self._setup_tfidf_fallback()
        else:
            raise ImportError(
                "sentence-transformers is not installed and fallback is disabled. "
                "Install with: pip install sentence-transformers"
            )

    def _setup_tfidf_fallback(self) -> None:
        """Set up TF-IDF vectorizer as fallback."""
        self._vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self._tfidf_vectorizer = self._vectorizer  # Alias for test compatibility
        self._using_fallback = True

    @property
    def is_using_fallback(self) -> bool:
        """Check if the engine is using TF-IDF fallback instead of Sentence Transformers."""
        return self._using_fallback

    def embed(self, text: str) -> NDArray[np.float64]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            1D numpy array embedding.
        """
        embeddings = self.compute_embeddings([text])
        return embeddings[0]

    def batch_embed(self, texts: list[str]) -> NDArray[np.float64]:
        """Embed multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            2D numpy array of embeddings.
        """
        return self.compute_embeddings(texts)

    def compute_embeddings(
        self, corpus: list[str]
    ) -> NDArray[np.float64]:
        """Compute embeddings for a list of chapter texts.

        Args:
            corpus: List of chapter text strings.

        Returns:
            2D numpy array of shape (n_chapters, embedding_dim) with embeddings.
            Returns empty array if corpus is empty.
        """
        if not corpus:
            return np.array([], dtype=np.float64)

        # Check cache if enabled
        if self.config.use_cache:
            cache_key = self._compute_cache_key(corpus)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

        # Compute embeddings
        if self._using_fallback:
            embeddings = self._compute_tfidf_embeddings(corpus)
        else:
            embeddings = self._compute_transformer_embeddings(corpus)

        # Cache if enabled
        if self.config.use_cache:
            cache_key = self._compute_cache_key(corpus)
            self._embedding_cache[cache_key] = embeddings

        return embeddings

    def _compute_cache_key(self, texts: list[str]) -> str:
        """Compute a cache key for a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            Hash-based cache key.
        """
        import hashlib

        combined = "|||".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()

    def _compute_transformer_embeddings(
        self, texts: list[str]
    ) -> NDArray[np.float64]:
        """Compute embeddings using Sentence Transformers.

        Args:
            texts: List of text strings.

        Returns:
            Embedding matrix as numpy array.
        """
        if self._model is None:
            raise RuntimeError("Sentence Transformer model not initialized")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float64)

    def _compute_tfidf_embeddings(self, texts: list[str]) -> NDArray[np.float64]:
        """Compute embeddings using TF-IDF vectorization.

        Args:
            texts: List of text strings.

        Returns:
            TF-IDF matrix as numpy array.
        """
        if self._vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer not initialized")

        # Handle empty texts
        if all(not t.strip() for t in texts):
            # Return zero embeddings for empty texts
            return np.zeros((len(texts), 1), dtype=np.float64)

        # Fit and transform
        tfidf_matrix = self._vectorizer.fit_transform(texts)
        return np.array(tfidf_matrix.toarray(), dtype=np.float64)

    def compute_similarity_matrix(
        self, data: list[str] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute pairwise cosine similarity matrix.

        Args:
            data: Either a list of text strings OR pre-computed embeddings array.
                  If strings, embeddings will be computed first.

        Returns:
            2D numpy array of shape (n, n) with similarity scores.
            Diagonal elements are 1.0 (self-similarity).
            All values are clipped to [-1.0, 1.0] range.

        Raises:
            ValueError: If data is empty.
        """
        # Handle text input - compute embeddings first
        if isinstance(data, list):
            if not data:
                raise ValueError("Cannot compute similarity matrix for empty corpus")
            embeddings = self.compute_embeddings(data)
        else:
            embeddings = data

        if embeddings.size == 0:
            raise ValueError("Cannot compute similarity matrix for empty embeddings")

        # Handle 1D case (single embedding)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Clip to valid range to handle floating-point precision issues
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

        return np.array(similarity_matrix, dtype=np.float64)

    def find_similar(
        self,
        query: str | int | None = None,
        candidates: list[str] | None = None,
        *,
        query_idx: int | None = None,
        embeddings: NDArray[np.float64] | None = None,
        index: list[dict[str, Any]] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[SimilarityResult]:
        """Find the most similar texts/chapters to a query.

        Supports two API patterns:
        1. Text-based (simpler): find_similar(query="text", candidates=["a", "b"])
        2. Index-based (original): find_similar(query_idx=0, embeddings=arr, index=[...])

        Args:
            query: Query text string OR query index (for backwards compat).
            candidates: List of candidate text strings to compare against.
            query_idx: Index of the source chapter in embeddings (original API).
            embeddings: Pre-computed embeddings array (original API).
            index: List of chapter info dicts (original API).
            top_k: Number of similar items to return.
            threshold: Minimum similarity score.

        Returns:
            List of SimilarityResult objects, sorted by similarity (descending).
        """
        # Use config defaults if not specified
        k = top_k if top_k is not None else self.config.top_k
        min_threshold = (
            threshold if threshold is not None else self.config.similarity_threshold
        )
        method = "tfidf" if self._using_fallback else "sentence_transformers"

        # Text-based API (simpler)
        if isinstance(query, str) and candidates is not None:
            return self._find_similar_texts(query, candidates, k, min_threshold, method)

        # Original index-based API
        if query_idx is not None and embeddings is not None and index is not None:
            return self._find_similar_indexed(query_idx, embeddings, index, k, min_threshold, method)

        # Handle integer query as query_idx for backwards compatibility
        if isinstance(query, int) and embeddings is not None and index is not None:
            return self._find_similar_indexed(query, embeddings, index, k, min_threshold, method)

        raise ValueError(
            "Invalid arguments. Use either find_similar(query='text', candidates=[...]) "
            "or find_similar(query_idx=0, embeddings=arr, index=[...])"
        )

    def _find_similar_texts(
        self,
        query: str,
        candidates: list[str],
        top_k: int,
        threshold: float,
        method: str,
    ) -> list[SimilarityResult]:
        """Find similar candidates to a query text (text-based API)."""
        if not candidates:
            return []

        # Compute embeddings for query and candidates together
        all_texts = [query] + candidates
        all_embeddings = self.compute_embeddings(all_texts)

        # Compute similarity of query (idx 0) against all candidates
        query_embedding = all_embeddings[0:1]  # Keep 2D
        candidate_embeddings = all_embeddings[1:]

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Build results
        results: list[SimilarityResult] = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                results.append(
                    SimilarityResult(
                        book="",
                        chapter=idx + 1,
                        title=candidates[idx][:50],  # Use first 50 chars as title
                        score=float(score),
                        method=method,
                    )
                )

        # Sort by similarity score (descending) and limit to top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _find_similar_indexed(
        self,
        query_idx: int,
        embeddings: NDArray[np.float64],
        index: list[dict[str, Any]],
        top_k: int,
        threshold: float,
        method: str,
    ) -> list[SimilarityResult]:
        """Find similar chapters by index (original API)."""
        if len(embeddings) == 0 or len(index) == 0:
            return []

        if query_idx < 0 or query_idx >= len(embeddings):
            raise ValueError(
                f"query_idx {query_idx} out of bounds for {len(embeddings)} chapters"
            )

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(embeddings)

        # Get similarities for the target chapter
        similarities = similarity_matrix[query_idx]

        # Find top-k similar chapters (excluding self)
        results: list[SimilarityResult] = []
        for idx, score in enumerate(similarities):
            if idx == query_idx:
                continue  # Skip self

            if score >= threshold:
                chapter_info = index[idx]
                results.append(
                    SimilarityResult(
                        book=chapter_info.get("book", ""),
                        chapter=chapter_info.get("chapter", idx + 1),
                        title=chapter_info.get("title", f"Chapter {idx + 1}"),
                        score=float(score),
                        method=method,
                    )
                )

        # Sort by similarity score (descending) and limit to top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
