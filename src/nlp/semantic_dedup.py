"""Semantic Deduplication module using SBERT + HDBSCAN - HCE-4.0.

Provides semantic deduplication for concept terms using sentence embeddings
and density-based clustering. Identifies semantically similar terms and
keeps the shortest (canonical form) as representative.

AC Reference:
- AC-4.2: SemanticDeduplicator module exists
- AC-4.3: compute_embeddings() returns (N, 384) array
- AC-4.4: cluster_concepts() returns labels via HDBSCAN
- AC-4.5: select_representatives() picks shortest per cluster

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Low cognitive complexity through single-responsibility methods
- #12: SBERT engine cached as singleton (no model per request)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, TypeVar, Union

import numpy as np
from hdbscan import HDBSCAN
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from src.models.sbert.semantic_similarity_engine import SemanticSimilarityEngine
from src.nlp.ensemble_merger import ExtractedTerm


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

# Default configuration values
DEFAULT_MIN_CLUSTER_SIZE: Final[int] = 2
DEFAULT_SIMILARITY_THRESHOLD: Final[float] = 0.8

# HDBSCAN constants
NOISE_LABEL: Final[int] = -1
HDBSCAN_METRIC: Final[str] = "precomputed"

# SBERT embedding dimensions (all-MiniLM-L6-v2)
EMBEDDING_DIM: Final[int] = 384


# =============================================================================
# Module-level Singleton (Anti-pattern #12 compliance)
# =============================================================================

# Cached SBERT engine - reused across all SemanticDeduplicator instances
_SBERT_ENGINE: SemanticSimilarityEngine | None = None


def _get_sbert_engine() -> SemanticSimilarityEngine:
    """Get or create the shared SBERT engine singleton.
    
    Returns:
        Shared SemanticSimilarityEngine instance.
    """
    global _SBERT_ENGINE
    if _SBERT_ENGINE is None:
        _SBERT_ENGINE = SemanticSimilarityEngine()
    return _SBERT_ENGINE


# =============================================================================
# Type Aliases
# =============================================================================

T = TypeVar("T", str, ExtractedTerm)


# =============================================================================
# Configuration Dataclass (AC-4.2)
# =============================================================================


@dataclass
class SemanticDedupConfig:
    """Configuration for semantic deduplication.
    
    Attributes:
        min_cluster_size: Minimum samples for HDBSCAN cluster.
        similarity_threshold: Cosine similarity threshold for clustering.
    """
    
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD


# =============================================================================
# SemanticDeduplicator Class (AC-4.2 through AC-4.5)
# =============================================================================


class SemanticDeduplicator:
    """Semantic deduplication using SBERT embeddings and HDBSCAN clustering.
    
    Workflow:
    1. compute_embeddings(): Generate 384-dim SBERT embeddings for terms
    2. cluster_concepts(): Group similar terms using HDBSCAN
    3. select_representatives(): Pick shortest term per cluster
    
    Example:
        >>> dedup = SemanticDeduplicator()
        >>> terms = ["API", "REST API", "web API"]
        >>> result, stats = dedup.deduplicate(terms)
        >>> print(result)  # ["API"] - shortest kept
    """
    
    def __init__(
        self,
        config: SemanticDedupConfig | None = None,
    ) -> None:
        """Initialize the SemanticDeduplicator.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or SemanticDedupConfig()
        self._engine = _get_sbert_engine()
    
    # =========================================================================
    # AC-4.3: Embedding Computation
    # =========================================================================
    
    def compute_embeddings(
        self,
        terms: list[str],
    ) -> NDArray[np.float64]:
        """Compute SBERT embeddings for a list of terms.
        
        Uses the cached SemanticSimilarityEngine (anti-pattern #12).
        
        Args:
            terms: List of term strings to embed.
            
        Returns:
            2D numpy array of shape (N, 384) with embeddings.
            Returns empty array shape (0,) if terms is empty.
        """
        if not terms:
            return np.array([], dtype=np.float64)
        
        return self._engine.compute_embeddings(terms)
    
    # =========================================================================
    # AC-4.4: Concept Clustering
    # =========================================================================
    
    def _compute_distance_matrix(
        self,
        embeddings: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute pairwise cosine distance matrix.
        
        Args:
            embeddings: 2D embedding array of shape (N, D).
            
        Returns:
            2D distance matrix of shape (N, N).
            Distance = 1 - cosine_similarity.
        """
        similarities: NDArray[Any] = cosine_similarity(embeddings)
        # Clip to [0, 2] range to handle floating point errors
        distances = np.clip(1.0 - similarities, 0.0, 2.0)
        return np.array(distances, dtype=np.float64)
    
    def cluster_concepts(
        self,
        embeddings: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Cluster embeddings using HDBSCAN.
        
        Uses precomputed cosine distance matrix for clustering.
        
        Args:
            embeddings: 2D embedding array of shape (N, 384).
            
        Returns:
            1D array of cluster labels. -1 indicates noise (unclustered).
        """
        n_samples = embeddings.shape[0] if len(embeddings.shape) > 1 else len(embeddings)
        
        # Edge cases
        if n_samples == 0:
            return np.array([], dtype=np.int64)
        
        if n_samples == 1:
            return np.array([NOISE_LABEL], dtype=np.int64)
        
        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(embeddings)
        
        # Run HDBSCAN with precomputed distances
        clusterer = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            metric=HDBSCAN_METRIC,
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(distance_matrix)
        
        return np.array(labels, dtype=np.int64)
    
    # =========================================================================
    # AC-4.5: Representative Selection
    # =========================================================================
    
    def select_representatives(
        self,
        terms: list[T],
        labels: NDArray[np.int64],
    ) -> tuple[list[T], int]:
        """Select shortest term as representative for each cluster.
        
        Noise points (label=-1) are always kept.
        For each cluster, the shortest term is selected.
        Ties are broken by keeping the first occurrence.
        
        Args:
            terms: List of terms (strings or ExtractedTerm).
            labels: Cluster labels array from cluster_concepts().
            
        Returns:
            Tuple of (deduplicated_terms, cluster_count).
        """
        if len(terms) == 0:
            return [], 0
        
        result: list[T] = []
        clusters_seen: set[int] = set()
        
        # Group terms by cluster
        cluster_terms: dict[int, list[tuple[int, T]]] = {}
        for idx, (term, label) in enumerate(zip(terms, labels)):
            label_int = int(label)
            if label_int not in cluster_terms:
                cluster_terms[label_int] = []
            cluster_terms[label_int].append((idx, term))
        
        # Process noise points first (keep all)
        if NOISE_LABEL in cluster_terms:
            for idx, term in cluster_terms[NOISE_LABEL]:
                result.append(term)
            del cluster_terms[NOISE_LABEL]
        
        # Process clusters - select shortest term
        for label, term_list in sorted(cluster_terms.items()):
            clusters_seen.add(label)
            
            # Get term text for length comparison
            def get_term_text(item: tuple[int, T]) -> str:
                _, t = item
                if isinstance(t, ExtractedTerm):
                    return t.term
                return str(t)
            
            # Sort by length, then by original index (for tie-breaking)
            sorted_terms = sorted(
                term_list,
                key=lambda x: (len(get_term_text(x)), x[0]),
            )
            
            # Keep shortest (first after sort)
            if sorted_terms:
                result.append(sorted_terms[0][1])
        
        return result, len(clusters_seen)
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def deduplicate(
        self,
        terms: list[T],
    ) -> tuple[list[T], dict[str, int]]:
        """Deduplicate terms using semantic similarity.
        
        Convenience method that runs the full pipeline:
        1. Extract term texts
        2. Compute embeddings
        3. Cluster concepts
        4. Select representatives
        
        Args:
            terms: List of terms (strings or ExtractedTerm).
            
        Returns:
            Tuple of (deduplicated_terms, stats_dict).
            stats_dict contains 'cluster_count' and 'removed_count'.
        """
        if not terms:
            return [], {"cluster_count": 0, "removed_count": 0}
        
        # Extract text from terms
        term_texts = [
            t.term if isinstance(t, ExtractedTerm) else str(t)
            for t in terms
        ]
        
        # Run pipeline
        embeddings = self.compute_embeddings(term_texts)
        labels = self.cluster_concepts(embeddings)
        result, cluster_count = self.select_representatives(terms, labels)
        
        removed_count = len(terms) - len(result)
        
        return result, {
            "cluster_count": cluster_count,
            "removed_count": removed_count,
        }
