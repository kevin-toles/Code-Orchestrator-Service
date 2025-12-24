"""BERTopic-based Concept Discovery module.

Uses BERTopic to automatically discover abstract concepts from extracted terms.
This replaces hardcoded seed concepts with data-driven concept discovery.

Key Insight: BERTopic clusters semantically similar terms into topics.
- Topics that cluster well = True Concepts (abstract ideas)
- Terms in small/singleton clusters = Keywords (just frequent words)

Flow:
1. Collect all extracted terms from metadata
2. Embed with SBERT (reuses existing model)
3. Cluster with HDBSCAN (reuses existing config)
4. Extract topic representations = Concepts
5. Save to config/discovered_concepts.json

AC Reference:
- Replaces hardcoded SEED_PROGRAMMING_CONCEPTS
- Enables data-driven concept validation
- Supports domain-specific concept discovery

Anti-Patterns Avoided:
- #12: BERTopic model cached as singleton
- S1192: Constants in external config files
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Logger
logger = logging.getLogger(__name__)

# =============================================================================
# Module Constants
# =============================================================================

# Default configuration values
DEFAULT_MIN_TOPIC_SIZE: Final[int] = 5
DEFAULT_MIN_CLUSTER_SIZE: Final[int] = 5
DEFAULT_UMAP_N_NEIGHBORS: Final[int] = 15
DEFAULT_UMAP_N_COMPONENTS: Final[int] = 5
DEFAULT_UMAP_MIN_DIST: Final[float] = 0.0
DEFAULT_TOP_N_WORDS: Final[int] = 5
DEFAULT_NR_TOPICS: Final[str] = "auto"  # or int for fixed number

# Model names
DEFAULT_SBERT_MODEL: Final[str] = "all-MiniLM-L6-v2"

# Output paths
DEFAULT_CONCEPTS_OUTPUT: Final[str] = "config/discovered_concepts.json"

# Topic filtering
MIN_CONCEPT_WORD_COUNT: Final[int] = 2  # Concepts must be 2+ words
NOISE_TOPIC_ID: Final[int] = -1  # BERTopic assigns -1 to noise/outliers

# Domain-specific noise patterns to filter from concept names
NOISE_PATTERNS: Final[set[str]] = {
    # Names (from book examples)
    "martin", "scott", "reilly", "robin", "smith", "john", "james",
    # Example placeholders
    "myhost", "mysa", "example", "foo", "bar", "baz", "tmp", "test123",
    # Abbreviations that aren't concepts
    "ee", "ebs", "tf", "t2", "fs", "fpm", "fib", "tldr", "dtype",
    # Non-technical words
    "dragon", "sphinx", "monster", "water", "air", "smoke", "cat",
    "says", "play", "way", "order", "total", "history",
}

# Technical domain indicators that boost concept quality
TECHNICAL_INDICATORS: Final[set[str]] = {
    # Software architecture
    "architecture", "microservice", "monolith", "pattern", "design",
    "service", "api", "gateway", "proxy", "mesh", "sidecar",
    # DevOps/Cloud
    "container", "kubernetes", "docker", "cluster", "pod", "node",
    "deployment", "pipeline", "cicd", "infrastructure",
    # Data
    "database", "schema", "query", "cache", "replication", "sharding",
    # Security
    "authentication", "authorization", "encryption", "secrets", "certificate",
    # Testing
    "testing", "test", "unit", "integration", "mocking",
    # Concepts
    "domain", "bounded", "context", "aggregate", "event", "saga",
    "circuit", "breaker", "resilience", "observability",
}


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class ConceptDiscoveryConfig:
    """Configuration for BERTopic concept discovery.

    Attributes:
        min_topic_size: Minimum documents per topic.
        min_cluster_size: HDBSCAN minimum cluster size.
        umap_n_neighbors: UMAP n_neighbors parameter.
        umap_n_components: UMAP dimensionality reduction target.
        umap_min_dist: UMAP minimum distance.
        top_n_words: Number of words to represent each topic.
        nr_topics: Number of topics ("auto" or int).
        sbert_model: Sentence transformer model name.
        output_path: Path to save discovered concepts.
        min_concept_word_count: Minimum words for a valid concept.
        filter_noise_patterns: Whether to filter noise patterns from topics.
    """

    min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS
    umap_n_components: int = DEFAULT_UMAP_N_COMPONENTS
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST
    top_n_words: int = DEFAULT_TOP_N_WORDS
    nr_topics: str | int = DEFAULT_NR_TOPICS
    sbert_model: str = DEFAULT_SBERT_MODEL
    output_path: str = DEFAULT_CONCEPTS_OUTPUT
    min_concept_word_count: int = MIN_CONCEPT_WORD_COUNT
    filter_noise_patterns: bool = True


@dataclass
class DiscoveredConcept:
    """A concept discovered by BERTopic.

    Attributes:
        topic_id: BERTopic topic ID.
        name: Human-readable concept name (top words joined).
        representative_terms: Top terms in this topic.
        term_count: Number of terms assigned to this topic.
        coherence_score: Topic coherence (if computed).
        quality_score: How "conceptual" the topic is (0-1). Higher = true concept.
    """

    topic_id: int
    name: str
    representative_terms: list[str] = field(default_factory=list)
    term_count: int = 0
    coherence_score: float = 0.0
    quality_score: float = 0.0


@dataclass
class ConceptDiscoveryResult:
    """Result of BERTopic concept discovery.

    Attributes:
        concepts: List of discovered concepts.
        noise_terms: Terms that didn't cluster (keywords, not concepts).
        topic_count: Number of topics discovered.
        total_terms: Total terms processed.
        model_info: BERTopic model metadata.
    """

    concepts: list[DiscoveredConcept] = field(default_factory=list)
    noise_terms: list[str] = field(default_factory=list)
    topic_count: int = 0
    total_terms: int = 0
    model_info: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Singleton Cache
# =============================================================================

_concept_discoverer: "ConceptDiscoverer | None" = None


def get_concept_discoverer(
    config: ConceptDiscoveryConfig | None = None,
) -> "ConceptDiscoverer":
    """Get or create cached ConceptDiscoverer instance.

    Implements singleton pattern per Anti-Pattern #12.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        Cached ConceptDiscoverer instance.
    """
    global _concept_discoverer
    if _concept_discoverer is None:
        _concept_discoverer = ConceptDiscoverer(config)
    return _concept_discoverer


# =============================================================================
# ConceptDiscoverer Class
# =============================================================================


class ConceptDiscoverer:
    """BERTopic-based concept discovery from extracted terms.

    Uses BERTopic to cluster semantically similar terms into topics,
    where well-formed topics represent true concepts and outliers
    are just keywords.

    Components:
    - SBERT: Sentence embeddings (reuses all-MiniLM-L6-v2)
    - UMAP: Dimensionality reduction
    - HDBSCAN: Density-based clustering
    - BERTopic: Topic modeling orchestration
    """

    def __init__(self, config: ConceptDiscoveryConfig | None = None) -> None:
        """Initialize concept discoverer.

        Args:
            config: Configuration for discovery. Defaults to ConceptDiscoveryConfig().
        """
        self.config = config or ConceptDiscoveryConfig()
        self._model: BERTopic | None = None
        self._embedding_model: SentenceTransformer | None = None

    def _get_embedding_model(self) -> SentenceTransformer:
        """Get or create SBERT embedding model (cached)."""
        if self._embedding_model is None:
            logger.info(f"Loading SBERT model: {self.config.sbert_model}")
            self._embedding_model = SentenceTransformer(self.config.sbert_model)
        return self._embedding_model

    def _compute_concept_quality_score(
        self, 
        concept_name: str, 
        representative_terms: list[str],
        term_count: int,
    ) -> float:
        """Compute quality score for a discovered concept.
        
        Higher scores indicate more "conceptual" topics (abstract ideas).
        Lower scores indicate keyword-like topics (just frequent words).
        
        Scoring factors:
        - Multi-word concept names (+)
        - Technical/programming terminology (+)
        - More terms in cluster = more coherent concept (+)
        - Single-word generic terms (-)
        - Noise pattern presence (-)
        
        Args:
            concept_name: The concept name (top words joined).
            representative_terms: All representative terms.
            term_count: Number of terms in the cluster.
            
        Returns:
            Quality score from 0.0 to 1.0.
        """
        score = 0.5  # Base score
        
        # Bonus for multi-word concept names
        word_count = len(concept_name.split())
        if word_count >= 3:
            score += 0.2
        elif word_count >= 2:
            score += 0.1
        
        # Bonus for larger clusters (more coherent)
        if term_count >= 30:
            score += 0.15
        elif term_count >= 15:
            score += 0.1
        elif term_count >= 8:
            score += 0.05
        
        # Bonus for multi-word terms in representatives
        multi_word_terms = sum(1 for t in representative_terms if " " in t)
        if multi_word_terms >= 2:
            score += 0.15
        elif multi_word_terms >= 1:
            score += 0.08
        
        # Penalty for noise patterns in terms
        noise_count = sum(
            1 for t in representative_terms[:5] 
            if t.lower() in NOISE_PATTERNS
        )
        score -= noise_count * 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _create_bertopic_model(self) -> BERTopic:
        """Create configured BERTopic model.

        Returns:
            Configured BERTopic instance.
        """
        # UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            min_dist=self.config.umap_min_dist,
            metric="cosine",
            random_state=42,
        )

        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # CountVectorizer for topic representation
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 3),  # Allow multi-word concepts
            min_df=2,
        )

        # Create BERTopic
        model = BERTopic(
            embedding_model=self._get_embedding_model(),
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=self.config.top_n_words,
            nr_topics=self.config.nr_topics if self.config.nr_topics != "auto" else None,
            min_topic_size=self.config.min_topic_size,
            verbose=False,
        )

        return model

    def discover_concepts(
        self,
        terms: list[str],
    ) -> ConceptDiscoveryResult:
        """Discover concepts from extracted terms using BERTopic.

        Args:
            terms: List of extracted terms (keywords + potential concepts).

        Returns:
            ConceptDiscoveryResult with discovered concepts and noise terms.
        """
        if not terms:
            return ConceptDiscoveryResult()

        logger.info(f"Discovering concepts from {len(terms)} terms")

        # Create fresh model for each discovery run
        model = self._create_bertopic_model()

        # Fit BERTopic
        topics, probs = model.fit_transform(terms)

        # Extract topic info
        topic_info = model.get_topic_info()

        # Build results
        concepts: list[DiscoveredConcept] = []
        noise_terms: list[str] = []

        for idx, topic_id in enumerate(topics):
            term = terms[idx]
            if topic_id == NOISE_TOPIC_ID:
                noise_terms.append(term)
            # Terms in valid topics will be grouped by topic

        # Process each topic (excluding noise topic -1)
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == NOISE_TOPIC_ID:
                continue

            # Get topic representation (top words)
            topic_words = model.get_topic(topic_id)
            if not topic_words:
                continue

            # Build concept name from top words
            top_terms = [word for word, _ in topic_words[:3]]
            
            # Filter noise patterns from topic terms if enabled
            if self.config.filter_noise_patterns:
                filtered_terms = [
                    term for term in top_terms 
                    if term.lower() not in NOISE_PATTERNS
                ]
                # Skip topics dominated by noise patterns
                if len(filtered_terms) < 2:
                    logger.debug(f"Skipping noise topic {topic_id}: {top_terms}")
                    continue
                top_terms = filtered_terms
            
            concept_name = " ".join(top_terms)

            # Filter: concepts should be multi-word or represent multi-word ideas
            if len(concept_name.split()) < self.config.min_concept_word_count:
                # Try to form a better concept name
                if len(top_terms) >= 2:
                    concept_name = f"{top_terms[0]} {top_terms[1]}"

            # Get all representative terms for quality scoring
            all_rep_terms = [w for w, _ in topic_words]
            
            # Compute quality score
            quality_score = self._compute_concept_quality_score(
                concept_name,
                all_rep_terms,
                row["Count"],
            )

            concepts.append(
                DiscoveredConcept(
                    topic_id=topic_id,
                    name=concept_name,
                    representative_terms=all_rep_terms,
                    term_count=row["Count"],
                    coherence_score=0.0,  # Could compute topic coherence
                    quality_score=quality_score,
                )
            )

        # Sort concepts by quality score (highest first)
        concepts.sort(key=lambda c: c.quality_score, reverse=True)

        # Store model for potential reuse
        self._model = model

        result = ConceptDiscoveryResult(
            concepts=concepts,
            noise_terms=noise_terms,
            topic_count=len(concepts),
            total_terms=len(terms),
            model_info={
                "n_topics": len(concepts),
                "n_noise": len(noise_terms),
                "noise_ratio": len(noise_terms) / len(terms) if terms else 0,
            },
        )

        logger.info(
            f"Discovered {result.topic_count} concepts, "
            f"{len(result.noise_terms)} noise terms"
        )

        return result

    def save_concepts(
        self,
        result: ConceptDiscoveryResult,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save discovered concepts to JSON config file.

        Args:
            result: ConceptDiscoveryResult to save.
            output_path: Output path. Defaults to config.output_path.

        Returns:
            Path to saved file.
        """
        path = Path(output_path or self.config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0.0",
            "generated_by": "ConceptDiscoverer (BERTopic)",
            "total_terms_processed": result.total_terms,
            "concepts": [
                {
                    "topic_id": c.topic_id,
                    "name": c.name,
                    "representative_terms": c.representative_terms,
                    "term_count": c.term_count,
                }
                for c in result.concepts
            ],
            "concept_names": [c.name for c in result.concepts],
            "noise_terms_sample": result.noise_terms[:100],  # Sample for reference
            "model_info": result.model_info,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(result.concepts)} concepts to {path}")
        return path

    def load_concepts(
        self,
        input_path: str | Path | None = None,
    ) -> list[str]:
        """Load discovered concepts from JSON config file.

        Args:
            input_path: Input path. Defaults to config.output_path.

        Returns:
            List of concept names for use as validation seeds.
        """
        path = Path(input_path or self.config.output_path)

        if not path.exists():
            logger.warning(f"Concepts file not found: {path}")
            return []

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        concepts = data.get("concept_names", [])
        logger.info(f"Loaded {len(concepts)} concepts from {path}")
        return concepts


# =============================================================================
# Utility Functions
# =============================================================================


def discover_concepts_from_metadata(
    metadata_dir: str | Path,
    output_path: str | Path | None = None,
    config: ConceptDiscoveryConfig | None = None,
) -> ConceptDiscoveryResult:
    """Discover concepts from all metadata files in a directory.

    Convenience function to run concept discovery on extracted metadata.

    Args:
        metadata_dir: Directory containing *_metadata.json files.
        output_path: Where to save discovered concepts.
        config: Optional configuration.

    Returns:
        ConceptDiscoveryResult with discovered concepts.
    """
    metadata_path = Path(metadata_dir)

    # Collect all extracted terms from metadata files
    all_terms: list[str] = []

    for meta_file in metadata_path.glob("*_metadata.json"):
        try:
            with open(meta_file, encoding="utf-8") as f:
                chapters = json.load(f)

            for chapter in chapters:
                # Collect keywords
                keywords = chapter.get("keywords", [])
                all_terms.extend(keywords)

                # Collect concepts (even if empty, for future)
                concepts = chapter.get("concepts", [])
                all_terms.extend(concepts)

        except Exception as e:
            logger.warning(f"Error reading {meta_file}: {e}")
            continue

    logger.info(f"Collected {len(all_terms)} terms from {metadata_path}")

    # Deduplicate while preserving rough order
    seen = set()
    unique_terms = []
    for term in all_terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    logger.info(f"Deduplicated to {len(unique_terms)} unique terms")

    # Run discovery
    discoverer = ConceptDiscoverer(config)
    result = discoverer.discover_concepts(unique_terms)

    # Save results
    if output_path:
        discoverer.save_concepts(result, output_path)

    return result
