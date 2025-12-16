"""
Unit tests for TopicClusterer - BERTopic wrapper.

WBS: BERTOPIC_INTEGRATION_WBS.md Phase B1.2
TDD Phase: RED (tests written before implementation)
Module: src/models/bertopic_clusterer.py

These tests verify the TopicClusterer class that wraps BERTopic
for topic clustering of chapter content.

Anti-Patterns Avoided:
- S1192: Model name extracted to constant
- #7, #13: No exception shadowing (custom BERTopicError)
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Constants per S1192 (avoid duplicated literals)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEST_CORPUS = [
    "Repository pattern for data persistence and domain modeling",
    "Unit testing with pytest fixtures and mocking strategies",
    "Repository implementations with SQLAlchemy ORM",
    "Test-driven development practices and test coverage",
    "Domain-driven design and aggregate patterns",
]


class TestTopicClustererInit:
    """Tests for TopicClusterer initialization."""

    def test_init_default_config(self) -> None:
        """TopicClusterer initializes with default configuration."""
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer()
        
        assert clusterer is not None
        assert clusterer.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert clusterer.min_topic_size >= 1

    def test_init_custom_embedding_model(self) -> None:
        """TopicClusterer accepts custom embedding model name."""
        from src.models.bertopic_clusterer import TopicClusterer

        custom_model = "paraphrase-MiniLM-L6-v2"
        clusterer = TopicClusterer(embedding_model=custom_model)
        
        assert clusterer.embedding_model == custom_model

    def test_init_custom_min_topic_size(self) -> None:
        """TopicClusterer accepts custom min_topic_size parameter."""
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer(min_topic_size=5)
        
        assert clusterer.min_topic_size == 5


class TestTopicClustererCluster:
    """Tests for TopicClusterer.cluster() method."""

    def test_cluster_empty_corpus_returns_empty_results(self) -> None:
        """Clustering empty corpus returns empty TopicResults."""
        from src.models.bertopic_clusterer import TopicClusterer, TopicResults

        clusterer = TopicClusterer()
        result = clusterer.cluster([])
        
        assert isinstance(result, TopicResults)
        assert len(result.topics) == 0
        assert len(result.assignments) == 0

    def test_cluster_single_document_returns_outlier_topic(self) -> None:
        """Single document is assigned to outlier topic (-1)."""
        from src.models.bertopic_clusterer import TopicClusterer, TopicResults

        clusterer = TopicClusterer()
        result = clusterer.cluster(["Single document about testing"])
        
        assert isinstance(result, TopicResults)
        assert len(result.assignments) == 1
        # BERTopic assigns -1 to outliers when insufficient data
        assert result.assignments[0].topic_id == -1

    def test_cluster_returns_topic_results_dataclass(self) -> None:
        """cluster() returns TopicResults dataclass with correct structure."""
        from src.models.bertopic_clusterer import TopicClusterer, TopicResults

        clusterer = TopicClusterer()
        result = clusterer.cluster(TEST_CORPUS)
        
        assert isinstance(result, TopicResults)
        assert hasattr(result, "topics")
        assert hasattr(result, "assignments")
        assert hasattr(result, "model_info")

    def test_cluster_assignments_match_corpus_length(self) -> None:
        """One TopicAssignment returned per input document."""
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer()
        result = clusterer.cluster(TEST_CORPUS)
        
        assert len(result.assignments) == len(TEST_CORPUS)

    def test_cluster_with_precomputed_embeddings(self) -> None:
        """cluster() accepts precomputed embeddings to skip re-encoding."""
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer()
        
        # Create dummy embeddings (384-dim for MiniLM-L6-v2)
        embeddings = np.random.rand(len(TEST_CORPUS), 384).astype(np.float64)
        
        result = clusterer.cluster(TEST_CORPUS, embeddings=embeddings)
        
        assert len(result.assignments) == len(TEST_CORPUS)
        # Should use provided embeddings, not recompute

    def test_cluster_discovers_related_topics(self) -> None:
        """Similar documents are clustered into same topic."""
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer(min_topic_size=2)
        result = clusterer.cluster(TEST_CORPUS)
        
        # Documents 0, 2, 4 are about repository/domain patterns
        # Documents 1, 3 are about testing
        # Should have at least 1 discovered topic (not all outliers)
        non_outlier_topics = [t for t in result.topics if t.topic_id != -1]
        assert len(non_outlier_topics) >= 1 or result.model_info.get("fallback_used", False)


class TestTopicClustererGetTopicInfo:
    """Tests for TopicClusterer.get_topic_info() method."""

    def test_get_topic_info_valid_id_returns_topic_info(self) -> None:
        """get_topic_info() returns TopicInfo for valid topic_id."""
        from src.models.bertopic_clusterer import TopicClusterer, TopicInfo

        clusterer = TopicClusterer()
        result = clusterer.cluster(TEST_CORPUS)
        
        if result.topics:
            topic_id = result.topics[0].topic_id
            topic_info = clusterer.get_topic_info(topic_id)
            
            assert isinstance(topic_info, TopicInfo)
            assert topic_info.topic_id == topic_id
            assert isinstance(topic_info.name, str)
            assert isinstance(topic_info.keywords, list)

    def test_get_topic_info_invalid_id_raises_error(self) -> None:
        """get_topic_info() raises BERTopicError for invalid topic_id."""
        from src.models.bertopic_clusterer import BERTopicError, TopicClusterer

        clusterer = TopicClusterer()
        clusterer.cluster(TEST_CORPUS)  # Need to cluster first
        
        with pytest.raises(BERTopicError, match="Topic ID .* not found"):
            clusterer.get_topic_info(9999)  # Invalid ID

    def test_get_topic_info_before_clustering_raises_error(self) -> None:
        """get_topic_info() raises error if called before cluster()."""
        from src.models.bertopic_clusterer import BERTopicError, TopicClusterer

        clusterer = TopicClusterer()
        
        with pytest.raises(BERTopicError, match="No topics available"):
            clusterer.get_topic_info(0)


class TestTopicClustererProperties:
    """Tests for TopicClusterer properties."""

    def test_topics_property_returns_list(self) -> None:
        """topics property returns list of all TopicInfo objects."""
        from src.models.bertopic_clusterer import TopicClusterer, TopicInfo

        clusterer = TopicClusterer()
        clusterer.cluster(TEST_CORPUS)
        
        topics = clusterer.topics
        
        assert isinstance(topics, list)
        for topic in topics:
            assert isinstance(topic, TopicInfo)

    def test_topics_property_empty_before_clustering(self) -> None:
        """topics property returns empty list before cluster() called."""
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer()
        
        assert clusterer.topics == []

    def test_is_using_fallback_property(self) -> None:
        """is_using_fallback indicates if BERTopic unavailable."""
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer()
        
        assert isinstance(clusterer.is_using_fallback, bool)


class TestTopicClustererFallback:
    """Tests for graceful degradation when BERTopic unavailable."""

    def test_fallback_when_bertopic_unavailable(self) -> None:
        """TopicClusterer falls back gracefully when BERTopic not installed."""
        from src.models.bertopic_clusterer import TopicClusterer

        # Mock BERTopic import failure
        with patch.dict("sys.modules", {"bertopic": None}):
            clusterer = TopicClusterer()
            result = clusterer.cluster(TEST_CORPUS)
            
            # Should still return valid results using fallback
            assert len(result.assignments) == len(TEST_CORPUS)
            assert result.model_info.get("fallback_used", False) is True

    def test_fallback_uses_kmeans_clustering(self) -> None:
        """Fallback mode uses scikit-learn KMeans for clustering."""
        from src.models.bertopic_clusterer import TopicClusterer

        with patch.dict("sys.modules", {"bertopic": None}):
            clusterer = TopicClusterer()
            result = clusterer.cluster(TEST_CORPUS)
            
            assert result.model_info.get("method") in ["kmeans", "fallback"]


class TestTopicDataClasses:
    """Tests for topic-related dataclasses."""

    def test_topic_info_dataclass_fields(self) -> None:
        """TopicInfo has required fields."""
        from src.models.bertopic_clusterer import TopicInfo

        topic = TopicInfo(
            topic_id=0,
            name="Repository Pattern",
            keywords=["repository", "persistence", "domain"],
            representative_docs=[0, 2, 4],
            size=3,
        )
        
        assert topic.topic_id == 0
        assert topic.name == "Repository Pattern"
        assert len(topic.keywords) == 3
        assert topic.size == 3

    def test_topic_assignment_dataclass_fields(self) -> None:
        """TopicAssignment has required fields."""
        from src.models.bertopic_clusterer import TopicAssignment

        assignment = TopicAssignment(
            topic_id=0,
            topic_name="Repository Pattern",
            confidence=0.87,
        )
        
        assert assignment.topic_id == 0
        assert assignment.topic_name == "Repository Pattern"
        assert assignment.confidence == 0.87

    def test_topic_results_dataclass_fields(self) -> None:
        """TopicResults has required fields."""
        from src.models.bertopic_clusterer import (
            TopicAssignment,
            TopicInfo,
            TopicResults,
        )

        topics = [TopicInfo(0, "Test", ["test"], [0], 1)]
        assignments = [TopicAssignment(0, "Test", 0.9)]
        
        results = TopicResults(
            topics=topics,
            assignments=assignments,
            model_info={"embedding_model": DEFAULT_EMBEDDING_MODEL},
        )
        
        assert len(results.topics) == 1
        assert len(results.assignments) == 1
        assert results.model_info["embedding_model"] == DEFAULT_EMBEDDING_MODEL


class TestTopicClustererProtocol:
    """Tests for TopicClustererProtocol conformance."""

    def test_topic_clusterer_implements_protocol(self) -> None:
        """TopicClusterer implements TopicClustererProtocol."""
        from src.models.protocols import TopicClustererProtocol
        from src.models.bertopic_clusterer import TopicClusterer

        clusterer = TopicClusterer()
        
        # Duck typing check - these should not raise
        assert hasattr(clusterer, "cluster")
        assert hasattr(clusterer, "get_topic_info")
        assert hasattr(clusterer, "topics")
        assert callable(clusterer.cluster)
        assert callable(clusterer.get_topic_info)

    def test_fake_topic_clusterer_implements_protocol(self) -> None:
        """FakeTopicClusterer implements TopicClustererProtocol for testing."""
        from src.models.bertopic_clusterer import FakeTopicClusterer

        fake = FakeTopicClusterer()
        
        assert hasattr(fake, "cluster")
        assert hasattr(fake, "get_topic_info")
        assert hasattr(fake, "topics")
        assert callable(fake.cluster)


class TestFakeTopicClusterer:
    """Tests for FakeTopicClusterer test double."""

    def test_fake_clusterer_returns_predictable_results(self) -> None:
        """FakeTopicClusterer returns predictable results for testing."""
        from src.models.bertopic_clusterer import FakeTopicClusterer

        fake = FakeTopicClusterer()
        result = fake.cluster(TEST_CORPUS)
        
        assert len(result.assignments) == len(TEST_CORPUS)
        # All documents assigned to topic 0 by default
        assert all(a.topic_id == 0 for a in result.assignments)

    def test_fake_clusterer_configurable_topics(self) -> None:
        """FakeTopicClusterer can be configured with custom topics."""
        from src.models.bertopic_clusterer import (
            FakeTopicClusterer,
            TopicAssignment,
            TopicInfo,
        )

        custom_topics = [
            TopicInfo(0, "Testing", ["test"], [1, 3], 2),
            TopicInfo(1, "Repository", ["repo"], [0, 2, 4], 3),
        ]
        
        fake = FakeTopicClusterer(topics=custom_topics)
        
        assert len(fake.topics) == 2
        assert fake.topics[0].name == "Testing"
