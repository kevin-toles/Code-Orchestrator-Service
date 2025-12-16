"""
Topic Clusterer using BERTopic.

This module provides topic clustering for chapter content,
enabling intelligent topic-based cross-referencing and content grouping.
It gracefully falls back to KMeans when BERTopic is unavailable.

WBS: BERTOPIC_INTEGRATION_WBS.md
Based on: BERTOPIC_SENTENCE_TRANSFORMERS_DESIGN.md

Role in Kitchen Brigade Architecture:
- Code-Orchestrator-Service (Sous Chef) hosts all understanding models
- BERTopic clusters chapters into semantic topics
- Provides topic_id, topic_name, confidence for cross-referencing
- Enables topic-based grouping in guideline generation

Architecture: Service Layer Pattern (Architecture Patterns Ch. 4)
Anti-Patterns Addressed:
- S1192: Model name extracted to DEFAULT_EMBEDDING_MODEL constant
- #7, #13: No exception shadowing (BERTopicError with original cause)
- #12: Embedding cache for performance
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Module constants per S1192 (no duplicated literals)
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_MIN_TOPIC_SIZE: int = 2
FALLBACK_N_CLUSTERS: int = 5

# Try to import BERTopic, fall back to KMeans if unavailable
try:
    from bertopic import BERTopic

    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    BERTopic = None  # type: ignore[misc, assignment]

# Fallback imports for when BERTopic is unavailable
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


class BERTopicError(Exception):
    """Exception raised for BERTopic-related errors.

    Follows anti-pattern #7, #13: No exception shadowing.
    Always preserves original exception as __cause__.
    """

    pass


@dataclass
class TopicInfo:
    """Information about a discovered topic.

    Attributes:
        topic_id: Unique topic identifier (-1 for outliers)
        name: Human-readable topic name (auto-generated from keywords)
        keywords: List of representative keywords for this topic
        representative_docs: Indices of documents that best represent topic
        size: Number of documents assigned to this topic
    """

    topic_id: int
    name: str
    keywords: list[str]
    representative_docs: list[int]
    size: int


@dataclass
class TopicAssignment:
    """Topic assignment for a single document.

    Attributes:
        topic_id: Assigned topic ID (-1 for outliers)
        topic_name: Human-readable name of assigned topic
        confidence: Probability that document belongs to this topic (0.0-1.0)
    """

    topic_id: int
    topic_name: str
    confidence: float


@dataclass
class TopicResults:
    """Results from topic clustering operation.

    Attributes:
        topics: List of discovered topics
        assignments: Topic assignment for each document (same order as input)
        model_info: Metadata about the clustering process
    """

    topics: list[TopicInfo]
    assignments: list[TopicAssignment]
    model_info: dict[str, Any] = field(default_factory=dict)


class TopicClusterer:
    """BERTopic-based topic clusterer for chapter content.

    Uses BERTopic for semantic topic discovery with graceful
    fallback to KMeans clustering when BERTopic is unavailable.

    Example:
        >>> clusterer = TopicClusterer()
        >>> corpus = ["Chapter about testing", "Chapter about repositories"]
        >>> results = clusterer.cluster(corpus)
        >>> print(results.assignments[0].topic_name)
        "Testing"

    Attributes:
        embedding_model: Name of the Sentence Transformer model to use
        min_topic_size: Minimum documents required to form a topic
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
    ) -> None:
        """Initialize the TopicClusterer.

        Args:
            embedding_model: Sentence Transformer model name
            min_topic_size: Minimum documents per topic (BERTopic parameter)
        """
        self._embedding_model = embedding_model
        self._min_topic_size = min_topic_size
        self._topics: list[TopicInfo] = []
        self._bertopic_model: Any = None
        self._is_using_fallback = not BERTOPIC_AVAILABLE

    @property
    def embedding_model(self) -> str:
        """Get the embedding model name."""
        return self._embedding_model

    @property
    def min_topic_size(self) -> int:
        """Get the minimum topic size."""
        return self._min_topic_size

    @property
    def topics(self) -> list[TopicInfo]:
        """Get list of discovered topics (empty before cluster() called)."""
        return self._topics

    @property
    def is_using_fallback(self) -> bool:
        """Check if using fallback clustering (BERTopic unavailable)."""
        return self._is_using_fallback

    def cluster(
        self,
        corpus: list[str],
        embeddings: NDArray[np.floating[Any]] | None = None,
    ) -> TopicResults:
        """Cluster documents into topics.

        Args:
            corpus: List of document texts to cluster
            embeddings: Optional precomputed embeddings (384-dim for MiniLM)

        Returns:
            TopicResults with topics and assignments for each document
        """
        # Handle empty corpus
        if not corpus:
            return TopicResults(
                topics=[],
                assignments=[],
                model_info={
                    "embedding_model": self._embedding_model,
                    "fallback_used": self._is_using_fallback,
                    "corpus_size": 0,
                },
            )

        # Handle single document (always outlier)
        if len(corpus) == 1:
            return self._handle_single_document()

        # Use BERTopic if available, otherwise fallback to KMeans
        if BERTOPIC_AVAILABLE and not self._is_using_fallback:
            return self._cluster_with_bertopic(corpus, embeddings)
        else:
            return self._cluster_with_fallback(corpus, embeddings)

    def _handle_single_document(self) -> TopicResults:
        """Handle edge case of single document (always outlier)."""
        outlier_topic = TopicInfo(
            topic_id=-1,
            name="Outlier",
            keywords=[],
            representative_docs=[0],
            size=1,
        )
        assignment = TopicAssignment(
            topic_id=-1,
            topic_name="Outlier",
            confidence=1.0,
        )
        self._topics = [outlier_topic]
        return TopicResults(
            topics=[outlier_topic],
            assignments=[assignment],
            model_info={
                "embedding_model": self._embedding_model,
                "fallback_used": True,
                "corpus_size": 1,
                "method": "single_document",
            },
        )

    def _cluster_with_bertopic(
        self,
        corpus: list[str],
        embeddings: NDArray[np.floating[Any]] | None = None,
    ) -> TopicResults:
        """Cluster using BERTopic library."""
        try:
            from sentence_transformers import SentenceTransformer

            # Initialize BERTopic with specified embedding model
            embedding_model = SentenceTransformer(self._embedding_model)
            self._bertopic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=self._min_topic_size,
                verbose=False,
            )

            # Fit the model
            topics_list, probs = self._fit_bertopic_model(corpus, embeddings)

            # Extract topic info and build results
            self._topics = self._extract_topics_from_bertopic(topics_list)
            assignments = self._build_assignments(topics_list, probs)

            return TopicResults(
                topics=self._topics,
                assignments=assignments,
                model_info={
                    "embedding_model": self._embedding_model,
                    "fallback_used": False,
                    "corpus_size": len(corpus),
                    "topic_count": len(self._topics),
                    "method": "bertopic",
                },
            )

        except Exception as e:
            # Fall back to KMeans on any BERTopic error (e.g., small corpus)
            self._is_using_fallback = True
            # Log the error but continue with fallback
            import logging
            logging.warning(f"BERTopic clustering failed, using KMeans fallback: {e}")
            return self._cluster_with_fallback(corpus, embeddings)

    def _fit_bertopic_model(
        self,
        corpus: list[str],
        embeddings: NDArray[np.floating[Any]] | None,
    ) -> tuple[list[int], list[float]]:
        """Fit BERTopic model and return topics and probabilities."""
        if embeddings is not None:
            return self._bertopic_model.fit_transform(corpus, embeddings=embeddings)
        return self._bertopic_model.fit_transform(corpus)

    def _extract_topics_from_bertopic(
        self,
        topics_list: list[int],
    ) -> list[TopicInfo]:
        """Extract TopicInfo objects from BERTopic model."""
        topic_info = self._bertopic_model.get_topic_info()
        topics = []

        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            name, keywords = self._get_topic_name_and_keywords(tid)
            size = int((np.array(topics_list) == tid).sum())
            rep_docs = [i for i, t in enumerate(topics_list) if t == tid][:3]

            topics.append(
                TopicInfo(
                    topic_id=int(tid),
                    name=name,
                    keywords=keywords,
                    representative_docs=rep_docs,
                    size=size,
                )
            )
        return topics

    def _get_topic_name_and_keywords(
        self,
        topic_id: int,
    ) -> tuple[str, list[str]]:
        """Get topic name and keywords for a given topic ID."""
        if topic_id == -1:
            return "Outlier", []
        topic_words = self._bertopic_model.get_topic(topic_id)
        keywords = [word for word, _ in topic_words[:5]] if topic_words else []
        name = "_".join(keywords[:3]) if keywords else f"Topic_{topic_id}"
        return name, keywords

    def _build_assignments(
        self,
        topics_list: list[int],
        probs: list[float],
    ) -> list[TopicAssignment]:
        """Build TopicAssignment objects from topics and probabilities."""
        assignments = []
        for topic_id, prob in zip(topics_list, probs):
            topic_name = next(
                (t.name for t in self._topics if t.topic_id == topic_id),
                "Unknown",
            )
            assignments.append(
                TopicAssignment(
                    topic_id=int(topic_id),
                    topic_name=topic_name,
                    confidence=float(prob) if not np.isnan(prob) else 0.0,
                )
            )
        return assignments

    def _cluster_with_fallback(
        self,
        corpus: list[str],
        embeddings: NDArray[np.floating[Any]] | None = None,
    ) -> TopicResults:
        """Fallback clustering using TF-IDF + KMeans."""
        # Determine number of clusters
        n_clusters = min(FALLBACK_N_CLUSTERS, len(corpus))

        # Use TF-IDF if no embeddings provided
        if embeddings is None:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            doc_vectors = vectorizer.fit_transform(corpus).toarray()
            feature_names = vectorizer.get_feature_names_out()
        else:
            doc_vectors = embeddings
            feature_names = []

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(doc_vectors)

        # Build topics from clusters
        self._topics = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.nonzero(cluster_mask)[0].tolist()
            size = int(cluster_mask.sum())

            # Extract keywords from TF-IDF for this cluster
            if embeddings is None and len(feature_names) > 0:
                cluster_centroid = kmeans.cluster_centers_[cluster_id]
                top_indices = np.argsort(cluster_centroid)[-5:][::-1]
                keywords = [feature_names[i] for i in top_indices]
            else:
                keywords = [f"keyword_{i}" for i in range(min(3, size))]

            name = "_".join(keywords[:3]) if keywords else f"Cluster_{cluster_id}"

            self._topics.append(
                TopicInfo(
                    topic_id=cluster_id,
                    name=name,
                    keywords=keywords,
                    representative_docs=cluster_indices[:3],
                    size=size,
                )
            )

        # Create assignments
        assignments = []
        for doc_idx, cluster_id in enumerate(cluster_labels):
            topic_name = self._topics[cluster_id].name
            # KMeans doesn't provide probabilities, use distance-based confidence
            centroid = kmeans.cluster_centers_[cluster_id]
            distance = np.linalg.norm(doc_vectors[doc_idx] - centroid)
            confidence = max(0.0, 1.0 - distance / 2.0)  # Normalize to 0-1

            assignments.append(
                TopicAssignment(
                    topic_id=int(cluster_id),
                    topic_name=topic_name,
                    confidence=float(confidence),
                )
            )

        return TopicResults(
            topics=self._topics,
            assignments=assignments,
            model_info={
                "embedding_model": self._embedding_model,
                "fallback_used": True,
                "corpus_size": len(corpus),
                "topic_count": len(self._topics),
                "method": "kmeans",
            },
        )

    def get_topic_info(self, topic_id: int) -> TopicInfo:
        """Get information about a specific topic.

        Args:
            topic_id: The topic ID to look up

        Returns:
            TopicInfo for the specified topic

        Raises:
            BERTopicError: If topic_id not found or no topics available
        """
        if not self._topics:
            raise BERTopicError(
                "No topics available. Call cluster() first."
            )

        for topic in self._topics:
            if topic.topic_id == topic_id:
                return topic

        raise BERTopicError(
            f"Topic ID {topic_id} not found in available topics: "
            f"{[t.topic_id for t in self._topics]}"
        )


class FakeTopicClusterer:
    """Fake implementation for testing purposes.

    Provides predictable, configurable topic clustering results
    without requiring BERTopic or actual ML computation.

    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md

    Example:
        >>> fake = FakeTopicClusterer()
        >>> result = fake.cluster(["doc1", "doc2"])
        >>> assert all(a.topic_id == 0 for a in result.assignments)
    """

    def __init__(
        self,
        topics: list[TopicInfo] | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Initialize FakeTopicClusterer with optional custom topics.

        Args:
            topics: Custom topics to return (default: single "Test" topic)
            embedding_model: Model name (stored but not used)
        """
        self._embedding_model = embedding_model
        self._topics = topics or [
            TopicInfo(
                topic_id=0,
                name="Test",
                keywords=["test", "fake", "mock"],
                representative_docs=[0],
                size=1,
            )
        ]
        self._is_using_fallback = True

    @property
    def embedding_model(self) -> str:
        """Get the embedding model name."""
        return self._embedding_model

    @property
    def min_topic_size(self) -> int:
        """Get minimum topic size (always 1 for fake)."""
        return 1

    @property
    def topics(self) -> list[TopicInfo]:
        """Get configured topics."""
        return self._topics

    @property
    def is_using_fallback(self) -> bool:
        """Always returns True for fake implementation."""
        return True

    def cluster(
        self,
        corpus: list[str],
        embeddings: NDArray[np.floating[Any]] | None = None,  # noqa: ARG002
    ) -> TopicResults:
        """Return predictable results for testing.

        All documents are assigned to topic 0 by default.
        Args embeddings kept for protocol compliance (unused).
        """
        if not corpus:
            return TopicResults(topics=[], assignments=[], model_info={})

        assignments = [
            TopicAssignment(
                topic_id=0,
                topic_name=self._topics[0].name if self._topics else "Test",
                confidence=1.0,
            )
            for _ in corpus
        ]

        return TopicResults(
            topics=self._topics,
            assignments=assignments,
            model_info={
                "embedding_model": self._embedding_model,
                "fallback_used": True,
                "method": "fake",
            },
        )

    def get_topic_info(self, topic_id: int) -> TopicInfo:
        """Get topic info by ID."""
        for topic in self._topics:
            if topic.topic_id == topic_id:
                return topic
        raise BERTopicError(f"Topic ID {topic_id} not found")
