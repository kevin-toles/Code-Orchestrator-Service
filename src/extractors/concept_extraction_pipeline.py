"""Concept Extraction Pipeline - HCE-1.0.

5-stage hybrid pipeline for concept extraction:
1. NoiseFilter (pre-filtering)
2. YAKE + TextRank (ensemble extraction) - HCE-2.0
3. Morphological Deduplication (stemming) - HCE-3.0
4. SBERT + HDBSCAN (semantic dedup) - HCE-4.0
5. GraphCodeBERT (optional validation) - existing

AC Reference:
- AC-1.1: Pipeline Instantiation
- AC-1.2: Config Injection
- AC-1.3: Extract Method returns ConceptExtractionResult
- AC-1.4: NoiseFilter Integration
- AC-1.5: Filter Bypass

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Cognitive complexity < 15 (one method per stage)
- S1172: No unused parameters
- #12: NoiseFilter reused (singleton pattern)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from src.validators.noise_filter import (
    BatchFilterResult,
    NoiseFilter,
)


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

# Pipeline stage names
STAGE_NOISE_FILTER: Final[str] = "noise_filter"
STAGE_YAKE: Final[str] = "yake"
STAGE_TEXTRANK: Final[str] = "textrank"
STAGE_STEM_DEDUP: Final[str] = "stem_dedup"
STAGE_SEMANTIC_DEDUP: Final[str] = "semantic_dedup"
STAGE_GRAPHCODEBERT: Final[str] = "graphcodebert"

# Default configuration values
DEFAULT_YAKE_TOP_N: Final[int] = 20
DEFAULT_YAKE_N_GRAM_SIZE: Final[int] = 3
DEFAULT_YAKE_DEDUP_THRESHOLD: Final[float] = 0.9
DEFAULT_TEXTRANK_WORDS: Final[int] = 20
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE: Final[int] = 2
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD: Final[float] = 0.8
DEFAULT_GRAPHCODEBERT_THRESHOLD: Final[float] = 0.7

# Default protected terms for stemming
DEFAULT_PROTECTED_TERMS: Final[list[str]] = [
    "microservices",
    "kubernetes",
    "APIs",
    "GraphQL",
    "REST",
]

# Extraction method identifier
EXTRACTION_METHOD_HYBRID: Final[str] = "hybrid_pipeline"

# Filter stats keys
FILTER_STATS_TERMS_FILTERED: Final[str] = "terms_filtered"
FILTER_STATS_CATEGORIES: Final[str] = "categories"


# =============================================================================
# Data Classes (AC-1.2, AC-1.3)
# =============================================================================


@dataclass
class ConceptExtractionConfig:
    """Configuration for the hybrid concept extraction pipeline.

    Attributes:
        enable_noise_filter: Stage 1 - Filter noise before extraction.
        enable_yake: Stage 2 - Use YAKE for keyword extraction.
        enable_textrank: Stage 2 - Use TextRank for keyword extraction.
        yake_top_n: YAKE max keywords to extract.
        yake_n_gram_size: YAKE max n-gram size.
        yake_dedup_threshold: YAKE deduplication threshold.
        textrank_words: TextRank max words to extract.
        enable_stem_dedup: Stage 3 - Morphological deduplication.
        protected_terms: Terms protected from stemming.
        enable_semantic_dedup: Stage 4 - SBERT+HDBSCAN deduplication.
        hdbscan_min_cluster_size: HDBSCAN minimum cluster size.
        semantic_similarity_threshold: Threshold for semantic similarity.
        enable_graphcodebert: Stage 5 - GraphCodeBERT validation.
        graphcodebert_threshold: GraphCodeBERT confidence threshold.
    """

    # Stage 1: NoiseFilter
    enable_noise_filter: bool = True

    # Stage 2: Ensemble Extraction
    enable_yake: bool = True
    enable_textrank: bool = True
    yake_top_n: int = DEFAULT_YAKE_TOP_N
    yake_n_gram_size: int = DEFAULT_YAKE_N_GRAM_SIZE
    yake_dedup_threshold: float = DEFAULT_YAKE_DEDUP_THRESHOLD
    textrank_words: int = DEFAULT_TEXTRANK_WORDS

    # Stage 3: Morphological Dedup
    enable_stem_dedup: bool = True
    protected_terms: list[str] = field(
        default_factory=lambda: list(DEFAULT_PROTECTED_TERMS)
    )

    # Stage 4: Semantic Dedup
    enable_semantic_dedup: bool = True
    hdbscan_min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
    semantic_similarity_threshold: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD

    # Stage 5: GraphCodeBERT (Optional)
    enable_graphcodebert: bool = False  # Off by default
    graphcodebert_threshold: float = DEFAULT_GRAPHCODEBERT_THRESHOLD


@dataclass
class ExtractedTerm:
    """A single extracted term with metadata.

    Attributes:
        term: The extracted term string.
        score: Extraction score (lower is better for YAKE convention).
        source: Source extractor(s) that found this term.
    """

    term: str
    score: float
    source: Literal["yake", "textrank", "both"]


@dataclass
class ConceptExtractionResult:
    """Result of concept extraction pipeline.

    Attributes:
        concepts: List of extracted terms with scores and sources.
        extraction_stats: Stats from extraction stage (yake_count, etc.).
        filter_stats: Stats from noise filtering (terms_filtered, categories).
        dedup_stats: Stats from deduplication (stem_removed, semantic_clusters).
        pipeline_metadata: Pipeline execution metadata (method, stages, duration).
    """

    concepts: list[ExtractedTerm] = field(default_factory=list)
    extraction_stats: dict[str, int] = field(default_factory=dict)
    filter_stats: dict[str, Any] = field(default_factory=dict)
    dedup_stats: dict[str, int] = field(default_factory=dict)
    pipeline_metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Pipeline Class (AC-1.1, AC-1.3, AC-1.4, AC-1.5)
# =============================================================================


class ConceptExtractionPipeline:
    """Hybrid concept extraction pipeline.

    5-stage pipeline:
    1. NoiseFilter - Pre-filtering (watermarks, URL fragments, etc.)
    2. YAKE + TextRank - Ensemble extraction (HCE-2.0)
    3. Morphological Dedup - Stemming (HCE-3.0)
    4. SBERT + HDBSCAN - Semantic dedup (HCE-4.0)
    5. GraphCodeBERT - Optional validation (existing)

    Anti-Pattern Compliance:
    - #12: NoiseFilter is a singleton (reused instance)
    - S3776: One method per stage, max complexity < 15
    """

    def __init__(
        self,
        config: ConceptExtractionConfig | None = None,
    ) -> None:
        """Initialize the pipeline with optional configuration.

        Args:
            config: Pipeline configuration. Defaults to ConceptExtractionConfig().
        """
        self.config = config or ConceptExtractionConfig()

        # Stage 1: NoiseFilter (AC-1.4)
        self._noise_filter = NoiseFilter()

    def extract(self, text: str) -> ConceptExtractionResult:
        """Extract concepts from text using the hybrid pipeline.

        Args:
            text: Raw text to extract concepts from.

        Returns:
            ConceptExtractionResult with concepts, stats, and metadata.
        """
        start_time = time.time()
        stages_executed: list[str] = []

        # Initialize stats
        filter_stats: dict[str, Any] = {
            FILTER_STATS_TERMS_FILTERED: 0,
            FILTER_STATS_CATEGORIES: {},
        }
        extraction_stats: dict[str, int] = {
            "yake_count": 0,
            "textrank_count": 0,
            "merged_count": 0,
        }
        dedup_stats: dict[str, int] = {
            "stem_removed": 0,
            "semantic_clusters": 0,
            "final_count": 0,
        }

        # Stage 1: NoiseFilter (AC-1.4, AC-1.5)
        filtered_text = text
        if self.config.enable_noise_filter:
            filtered_text, filter_stats = self._apply_noise_filter(text)
            stages_executed.append(STAGE_NOISE_FILTER)

        # TODO: Stage 2 - YAKE + TextRank (HCE-2.0)
        # TODO: Stage 3 - Morphological Dedup (HCE-3.0)
        # TODO: Stage 4 - SBERT + HDBSCAN (HCE-4.0)
        # TODO: Stage 5 - GraphCodeBERT (existing, optional)

        # Placeholder: Return empty concepts until HCE-2.0
        concepts: list[ExtractedTerm] = []

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Build pipeline metadata
        pipeline_metadata: dict[str, Any] = {
            "extraction_method": EXTRACTION_METHOD_HYBRID,
            "stages_executed": stages_executed,
            "duration_ms": duration_ms,
        }

        return ConceptExtractionResult(
            concepts=concepts,
            extraction_stats=extraction_stats,
            filter_stats=filter_stats,
            dedup_stats=dedup_stats,
            pipeline_metadata=pipeline_metadata,
        )

    def _apply_noise_filter(
        self,
        text: str,
    ) -> tuple[str, dict[str, Any]]:
        """Apply NoiseFilter to remove noise terms from text.

        Stage 1 of the pipeline. Filters watermarks, URL fragments,
        generic filler, code artifacts, and page markers.

        Args:
            text: Raw text to filter.

        Returns:
            Tuple of (filtered_text, filter_stats).
        """
        # For text-level filtering, we split into words and filter
        words = text.split()
        batch_result: BatchFilterResult = self._noise_filter.filter_batch(words)

        # Reconstruct filtered text
        filtered_text = " ".join(batch_result.accepted)

        # Build filter stats
        filter_stats: dict[str, Any] = {
            FILTER_STATS_TERMS_FILTERED: len(batch_result.rejected),
            FILTER_STATS_CATEGORIES: batch_result.rejection_reasons,
        }

        return filtered_text, filter_stats
