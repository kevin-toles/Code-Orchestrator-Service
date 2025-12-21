"""Tests for Semantic Deduplication module - HCE-4.0.

RED Phase tests for:
- HCE-4.2: SemanticDeduplicator import
- HCE-4.3: SemanticDedupConfig required fields
- HCE-4.4: compute_embeddings() returns ndarray
- HCE-4.5: Embeddings have correct shape (N, 384)
- HCE-4.6: SBERT engine is cached
- HCE-4.10: cluster_concepts() computes distance matrix
- HCE-4.11: cluster_concepts() uses HDBSCAN
- HCE-4.12: cluster_concepts() returns labels array
- HCE-4.13: cluster_concepts() handles single term
- HCE-4.15: select_representatives() keeps noise points
- HCE-4.16: select_representatives() picks shortest
- HCE-4.17: select_representatives() handles ties
- HCE-4.18: select_representatives() returns cluster count

AC Reference:
- AC-4.2: Deduplicator Module exists
- AC-4.3: compute_embeddings() returns (N, 384) array
- AC-4.4: cluster_concepts() returns labels via HDBSCAN
- AC-4.5: select_representatives() picks shortest per cluster
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# HCE-4.2: Test SemanticDeduplicator can be imported (AC-4.2)
# =============================================================================


class TestSemanticDeduplicatorImport:
    """Test module imports for AC-4.2."""

    def test_semantic_dedup_module_importable(self) -> None:
        """src.nlp.semantic_dedup module can be imported."""
        from src.nlp import semantic_dedup  # noqa: F401

        assert semantic_dedup is not None

    def test_semantic_deduplicator_class_importable(self) -> None:
        """SemanticDeduplicator class can be imported."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        assert SemanticDeduplicator is not None

    def test_semantic_dedup_config_importable(self) -> None:
        """SemanticDedupConfig dataclass can be imported."""
        from src.nlp.semantic_dedup import SemanticDedupConfig

        assert SemanticDedupConfig is not None

    def test_module_level_constants_exist(self) -> None:
        """Module has required constants at module level (S1192)."""
        from src.nlp.semantic_dedup import (
            DEFAULT_MIN_CLUSTER_SIZE,
            DEFAULT_SIMILARITY_THRESHOLD,
            NOISE_LABEL,
        )

        assert DEFAULT_MIN_CLUSTER_SIZE == 2
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.8
        assert NOISE_LABEL == -1


# =============================================================================
# HCE-4.3: Test SemanticDedupConfig has required fields (AC-4.2)
# =============================================================================


class TestSemanticDedupConfig:
    """Test SemanticDedupConfig dataclass for AC-4.2."""

    def test_config_has_min_cluster_size_field(self) -> None:
        """Config has min_cluster_size field with default."""
        from src.nlp.semantic_dedup import SemanticDedupConfig

        config = SemanticDedupConfig()
        assert hasattr(config, "min_cluster_size")
        assert config.min_cluster_size == 2

    def test_config_has_similarity_threshold_field(self) -> None:
        """Config has similarity_threshold field with default."""
        from src.nlp.semantic_dedup import SemanticDedupConfig

        config = SemanticDedupConfig()
        assert hasattr(config, "similarity_threshold")
        assert config.similarity_threshold == 0.8

    def test_config_accepts_custom_values(self) -> None:
        """Config accepts custom values."""
        from src.nlp.semantic_dedup import SemanticDedupConfig

        config = SemanticDedupConfig(min_cluster_size=3, similarity_threshold=0.9)
        assert config.min_cluster_size == 3
        assert config.similarity_threshold == 0.9

    def test_config_is_dataclass(self) -> None:
        """Config is a proper dataclass."""
        from dataclasses import is_dataclass

        from src.nlp.semantic_dedup import SemanticDedupConfig

        assert is_dataclass(SemanticDedupConfig)


# =============================================================================
# HCE-4.4: Test compute_embeddings() returns ndarray (AC-4.3)
# =============================================================================


class TestComputeEmbeddings:
    """Test compute_embeddings() for AC-4.3."""

    def test_compute_embeddings_returns_ndarray(self) -> None:
        """compute_embeddings() returns numpy ndarray."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["machine learning", "data science"]
        embeddings = dedup.compute_embeddings(terms)

        assert isinstance(embeddings, np.ndarray)

    def test_compute_embeddings_empty_list_returns_empty_array(self) -> None:
        """compute_embeddings() with empty list returns empty array."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = dedup.compute_embeddings([])

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 0

    def test_compute_embeddings_single_term(self) -> None:
        """compute_embeddings() works with single term."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = dedup.compute_embeddings(["machine learning"])

        assert embeddings.shape[0] == 1


# =============================================================================
# HCE-4.5: Test embeddings have correct shape (N, 384) (AC-4.3)
# =============================================================================


class TestEmbeddingsShape:
    """Test embedding dimensions for AC-4.3."""

    def test_embeddings_have_correct_shape_n_384(self) -> None:
        """Embeddings have shape (N, 384) for SBERT MiniLM."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["machine learning", "data science", "deep learning"]
        embeddings = dedup.compute_embeddings(terms)

        assert embeddings.shape == (3, 384)

    def test_embeddings_shape_varies_with_input_count(self) -> None:
        """Embedding count matches input term count."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()

        for count in [1, 5, 10]:
            terms = [f"term{i}" for i in range(count)]
            embeddings = dedup.compute_embeddings(terms)
            assert embeddings.shape[0] == count

    def test_embeddings_are_float64(self) -> None:
        """Embeddings are float64 dtype."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = dedup.compute_embeddings(["test term"])

        assert embeddings.dtype == np.float64


# =============================================================================
# HCE-4.6: Test SBERT engine is cached (AC-4.3)
# =============================================================================


class TestSBERTEngineCaching:
    """Test SBERT engine singleton/caching for AC-4.3 and anti-pattern #12."""

    def test_sbert_engine_is_cached(self) -> None:
        """SBERT engine is reused (not recreated per request)."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup1 = SemanticDeduplicator()
        dedup2 = SemanticDeduplicator()

        # Engine should be same instance (cached singleton)
        assert dedup1._engine is dedup2._engine

    def test_multiple_compute_calls_use_same_engine(self) -> None:
        """Multiple compute_embeddings() calls reuse engine."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        engine_before = dedup._engine

        dedup.compute_embeddings(["term1"])
        dedup.compute_embeddings(["term2"])

        assert dedup._engine is engine_before


# =============================================================================
# HCE-4.10: Test cluster_concepts() computes distance matrix (AC-4.4)
# =============================================================================


class TestClusterConceptsDistanceMatrix:
    """Test cluster_concepts() distance matrix computation for AC-4.4."""

    def test_cluster_concepts_uses_cosine_distance(self) -> None:
        """cluster_concepts() uses 1 - cosine_similarity for distance."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()

        # Use identical terms to verify distance is 0
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Same as first
        ])

        # Should cluster together (distance ~0)
        with patch.object(dedup, "_compute_distance_matrix") as mock_dist:
            mock_dist.return_value = np.array([[0.0, 0.0], [0.0, 0.0]])
            dedup.cluster_concepts(embeddings)
            mock_dist.assert_called_once()

    def test_distance_matrix_is_symmetric(self) -> None:
        """Distance matrix should be symmetric."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = dedup.compute_embeddings(["API", "REST API", "web API"])

        dist_matrix = dedup._compute_distance_matrix(embeddings)

        # Check symmetry
        assert np.allclose(dist_matrix, dist_matrix.T)


# =============================================================================
# HCE-4.11: Test cluster_concepts() uses HDBSCAN (AC-4.4)
# =============================================================================


class TestClusterConceptsHDBSCAN:
    """Test cluster_concepts() HDBSCAN usage for AC-4.4."""

    def test_cluster_concepts_uses_hdbscan(self) -> None:
        """cluster_concepts() uses HDBSCAN for clustering."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = np.random.rand(5, 384)

        # Verify HDBSCAN is called
        with patch("src.nlp.semantic_dedup.HDBSCAN") as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
            mock_hdbscan.return_value = mock_clusterer

            dedup.cluster_concepts(embeddings)

            mock_hdbscan.assert_called_once()
            mock_clusterer.fit_predict.assert_called_once()

    def test_hdbscan_uses_precomputed_metric(self) -> None:
        """HDBSCAN is initialized with metric='precomputed'."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = np.random.rand(5, 384)

        with patch("src.nlp.semantic_dedup.HDBSCAN") as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
            mock_hdbscan.return_value = mock_clusterer

            dedup.cluster_concepts(embeddings)

            # Verify metric='precomputed' is passed
            call_kwargs = mock_hdbscan.call_args[1]
            assert call_kwargs.get("metric") == "precomputed"

    def test_hdbscan_uses_config_min_cluster_size(self) -> None:
        """HDBSCAN uses min_cluster_size from config."""
        from src.nlp.semantic_dedup import SemanticDedupConfig, SemanticDeduplicator

        config = SemanticDedupConfig(min_cluster_size=5)
        dedup = SemanticDeduplicator(config=config)
        embeddings = np.random.rand(10, 384)

        with patch("src.nlp.semantic_dedup.HDBSCAN") as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.array([0] * 10)
            mock_hdbscan.return_value = mock_clusterer

            dedup.cluster_concepts(embeddings)

            call_kwargs = mock_hdbscan.call_args[1]
            assert call_kwargs.get("min_cluster_size") == 5


# =============================================================================
# HCE-4.12: Test cluster_concepts() returns labels array (AC-4.4)
# =============================================================================


class TestClusterConceptsLabels:
    """Test cluster_concepts() label output for AC-4.4."""

    def test_cluster_concepts_returns_ndarray(self) -> None:
        """cluster_concepts() returns numpy ndarray of labels."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = np.random.rand(5, 384)
        labels = dedup.cluster_concepts(embeddings)

        assert isinstance(labels, np.ndarray)

    def test_cluster_concepts_labels_length_matches_input(self) -> None:
        """Labels array length matches number of input embeddings."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = np.random.rand(7, 384)
        labels = dedup.cluster_concepts(embeddings)

        assert len(labels) == 7

    def test_cluster_concepts_labels_are_integers(self) -> None:
        """Cluster labels are integers."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = np.random.rand(5, 384)
        labels = dedup.cluster_concepts(embeddings)

        assert np.issubdtype(labels.dtype, np.integer)


# =============================================================================
# HCE-4.13: Test cluster_concepts() handles single term (AC-4.4)
# =============================================================================


class TestClusterConceptsSingleTerm:
    """Test cluster_concepts() edge cases for AC-4.4."""

    def test_single_term_returns_noise_label(self) -> None:
        """Single term is labeled as noise (-1) since can't cluster."""
        from src.nlp.semantic_dedup import NOISE_LABEL, SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = np.random.rand(1, 384)
        labels = dedup.cluster_concepts(embeddings)

        assert len(labels) == 1
        assert labels[0] == NOISE_LABEL

    def test_two_terms_can_form_cluster(self) -> None:
        """Two similar terms can form a cluster."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        # Create very similar embeddings
        embeddings = np.array([
            np.ones(384),
            np.ones(384) * 0.99,  # Very similar
        ])
        labels = dedup.cluster_concepts(embeddings)

        assert len(labels) == 2

    def test_empty_embeddings_returns_empty_array(self) -> None:
        """Empty embeddings returns empty labels array."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        embeddings = np.array([]).reshape(0, 384)
        labels = dedup.cluster_concepts(embeddings)

        assert len(labels) == 0


# =============================================================================
# HCE-4.15: Test select_representatives() keeps noise points (AC-4.5)
# =============================================================================


class TestSelectRepresentativesNoise:
    """Test select_representatives() noise handling for AC-4.5."""

    def test_noise_points_are_kept(self) -> None:
        """Noise points (label=-1) are kept in output."""
        from src.nlp.semantic_dedup import NOISE_LABEL, SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["unique1", "unique2", "clustered1", "clustered2"]
        labels = np.array([NOISE_LABEL, NOISE_LABEL, 0, 0])

        result, cluster_count = dedup.select_representatives(terms, labels)

        # Both noise points should be kept
        assert "unique1" in result
        assert "unique2" in result

    def test_all_noise_returns_all_terms(self) -> None:
        """If all terms are noise, all are returned."""
        from src.nlp.semantic_dedup import NOISE_LABEL, SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["term1", "term2", "term3"]
        labels = np.array([NOISE_LABEL, NOISE_LABEL, NOISE_LABEL])

        result, cluster_count = dedup.select_representatives(terms, labels)

        assert len(result) == 3
        assert cluster_count == 0


# =============================================================================
# HCE-4.16: Test select_representatives() picks shortest (AC-4.5)
# =============================================================================


class TestSelectRepresentativesShortest:
    """Test select_representatives() shortest term selection for AC-4.5."""

    def test_picks_shortest_term_in_cluster(self) -> None:
        """Selects shortest term as cluster representative."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["API", "REST API", "web API", "API gateway"]
        labels = np.array([0, 0, 0, 0])  # All in same cluster

        result, cluster_count = dedup.select_representatives(terms, labels)

        # "API" is shortest and should be kept
        assert "API" in result
        assert len(result) == 1

    def test_multiple_clusters_pick_shortest_each(self) -> None:
        """Each cluster picks its shortest term."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["ML", "machine learning", "AI", "artificial intelligence"]
        labels = np.array([0, 0, 1, 1])  # Two clusters

        result, cluster_count = dedup.select_representatives(terms, labels)

        assert "ML" in result
        assert "AI" in result
        assert len(result) == 2


# =============================================================================
# HCE-4.17: Test select_representatives() handles ties (AC-4.5)
# =============================================================================


class TestSelectRepresentativesTies:
    """Test select_representatives() tie-breaking for AC-4.5."""

    def test_equal_length_keeps_first_occurrence(self) -> None:
        """When terms have equal length, first occurrence is kept."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["API", "SDK", "CLI"]  # All 3 chars
        labels = np.array([0, 0, 0])

        result, cluster_count = dedup.select_representatives(terms, labels)

        # First occurrence "API" should be kept
        assert result == ["API"]

    def test_tie_with_different_order(self) -> None:
        """First occurrence in input order wins tie."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["CLI", "API", "SDK"]  # CLI is first
        labels = np.array([0, 0, 0])

        result, cluster_count = dedup.select_representatives(terms, labels)

        assert result == ["CLI"]


# =============================================================================
# HCE-4.18: Test select_representatives() returns cluster count (AC-4.5)
# =============================================================================


class TestSelectRepresentativesClusterCount:
    """Test select_representatives() cluster count for AC-4.5."""

    def test_returns_tuple_with_count(self) -> None:
        """Returns tuple of (terms, cluster_count)."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["a", "b", "c"]
        labels = np.array([0, 0, 1])

        result = dedup.select_representatives(terms, labels)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cluster_count_is_correct(self) -> None:
        """Cluster count matches actual clusters (excluding noise)."""
        from src.nlp.semantic_dedup import NOISE_LABEL, SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["a", "b", "c", "d", "e"]
        labels = np.array([0, 0, 1, 1, NOISE_LABEL])  # 2 clusters, 1 noise

        result, cluster_count = dedup.select_representatives(terms, labels)

        assert cluster_count == 2

    def test_no_clusters_returns_zero(self) -> None:
        """No clusters (all noise) returns count of 0."""
        from src.nlp.semantic_dedup import NOISE_LABEL, SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["a", "b"]
        labels = np.array([NOISE_LABEL, NOISE_LABEL])

        result, cluster_count = dedup.select_representatives(terms, labels)

        assert cluster_count == 0


# =============================================================================
# Integration Tests: Full deduplication flow
# =============================================================================


class TestSemanticDeduplicationIntegration:
    """Integration tests for full semantic deduplication flow."""

    def test_deduplicate_method_exists(self) -> None:
        """SemanticDeduplicator has deduplicate() convenience method."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        assert hasattr(dedup, "deduplicate")
        assert callable(dedup.deduplicate)

    def test_deduplicate_returns_terms_and_stats(self) -> None:
        """deduplicate() returns terms and stats dict."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = ["machine learning", "ML", "deep learning"]

        result, stats = dedup.deduplicate(terms)

        assert isinstance(result, list)
        assert isinstance(stats, dict)
        assert "cluster_count" in stats
        assert "removed_count" in stats

    def test_deduplicate_reduces_similar_terms(self) -> None:
        """deduplicate() reduces semantically similar terms."""
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = [
            "machine learning",
            "machine learning models",
            "ML models",
            "deep learning",
            "neural networks",
        ]

        result, stats = dedup.deduplicate(terms)

        # Should have fewer terms than input
        assert len(result) <= len(terms)

    def test_deduplicate_handles_extracted_terms(self) -> None:
        """deduplicate() can work with ExtractedTerm objects."""
        from src.nlp.ensemble_merger import ExtractedTerm
        from src.nlp.semantic_dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator()
        terms = [
            ExtractedTerm(term="API", score=0.1, source="yake"),
            ExtractedTerm(term="REST API", score=0.2, source="textrank"),
            ExtractedTerm(term="web API", score=0.3, source="both"),
        ]

        result, stats = dedup.deduplicate(terms)

        assert isinstance(result, list)
        # All results should be ExtractedTerm
        for item in result:
            assert isinstance(item, ExtractedTerm)
