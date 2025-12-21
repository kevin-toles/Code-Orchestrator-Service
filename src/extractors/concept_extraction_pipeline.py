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
from src.nlp.yake_extractor import YAKEConfig, YAKEExtractor
from src.nlp.textrank_extractor import TextRankConfig, TextRankExtractor
from src.nlp.ensemble_merger import EnsembleMerger, ExtractedTerm as MergerExtractedTerm
from src.nlp.stemmer import deduplicate_by_stem
from src.nlp.semantic_dedup import SemanticDedupConfig, SemanticDeduplicator
from src.nlp.concept_validator import ConceptValidator, ConceptValidationConfig


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

# Pipeline stage names
STAGE_NOISE_FILTER: Final[str] = "noise_filter"
STAGE_YAKE: Final[str] = "yake"
STAGE_TEXTRANK: Final[str] = "textrank"
STAGE_STEM_DEDUP: Final[str] = "stem_dedup"
STAGE_CONCEPT_VALIDATION: Final[str] = "concept_validation"
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
DEFAULT_SBERT_VALIDATION_THRESHOLD: Final[float] = 0.35

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

    # Stage 4.5: Concept Validation (SBERT seed + pattern filter)
    enable_concept_validation: bool = True
    enable_pattern_filter: bool = True  # Author/noise patterns
    enable_sbert_validation: bool = True  # Semantic similarity to seeds
    sbert_validation_threshold: float = DEFAULT_SBERT_VALIDATION_THRESHOLD

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

        # Stage 2: Ensemble Extraction (HCE-2.0, AC-2.8)
        yake_config = YAKEConfig(
            top_n=self.config.yake_top_n,
            n_gram_size=self.config.yake_n_gram_size,
            dedup_threshold=self.config.yake_dedup_threshold,
        )
        self._yake_extractor = YAKEExtractor(config=yake_config)

        textrank_config = TextRankConfig(words=self.config.textrank_words)
        self._textrank_extractor = TextRankExtractor(config=textrank_config)

        self._ensemble_merger = EnsembleMerger()

        # Stage 4: Semantic Deduplication (HCE-4.0, AC-4.6)
        semantic_config = SemanticDedupConfig(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            similarity_threshold=self.config.semantic_similarity_threshold,
        )
        self._semantic_deduplicator = SemanticDeduplicator(config=semantic_config)

        # Stage 4.5: Concept Validation (SBERT seed + pattern filter)
        validation_config = ConceptValidationConfig(
            enable_pattern_filter=self.config.enable_pattern_filter,
            enable_sbert_validation=self.config.enable_sbert_validation,
            sbert_similarity_threshold=self.config.sbert_validation_threshold,
        )
        self._concept_validator = ConceptValidator(
            config=validation_config,
            deduplicator=self._semantic_deduplicator,  # Share SBERT engine
        )

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
            "validation_rejected": 0,
            "semantic_clusters": 0,
            "semantic_removed": 0,
            "final_count": 0,
        }

        # Stage 1: NoiseFilter (AC-1.4, AC-1.5)
        filtered_text = text
        if self.config.enable_noise_filter:
            filtered_text, filter_stats = self._apply_noise_filter(text)
            stages_executed.append(STAGE_NOISE_FILTER)

        # Stage 2: YAKE + TextRank Ensemble Extraction (HCE-2.0, AC-2.8)
        yake_terms: list[tuple[str, float]] = []
        textrank_terms: list[str] = []

        if self.config.enable_yake:
            yake_terms = self._yake_extractor.extract(filtered_text)
            extraction_stats["yake_count"] = len(yake_terms)
            stages_executed.append(STAGE_YAKE)

        if self.config.enable_textrank:
            textrank_terms = self._textrank_extractor.extract(filtered_text)
            extraction_stats["textrank_count"] = len(textrank_terms)
            stages_executed.append(STAGE_TEXTRANK)

        # Merge results using EnsembleMerger
        merged_terms = self._ensemble_merger.merge(yake_terms, textrank_terms)
        extraction_stats["merged_count"] = len(merged_terms)

        # Convert to pipeline's ExtractedTerm format
        concepts: list[ExtractedTerm] = [
            ExtractedTerm(term=t.term, score=t.score, source=t.source)  # type: ignore[arg-type]
            for t in merged_terms
        ]

        # Post-filter: Remove concepts containing noise terms (AC-1.4)
        if self.config.enable_noise_filter:
            concepts = self._filter_noise_concepts(concepts)

        # Stage 3: Morphological Dedup (HCE-3.0, AC-3.5)
        if self.config.enable_stem_dedup:
            concepts, stem_removed = self._apply_stem_deduplication(concepts)
            dedup_stats["stem_removed"] = stem_removed
            stages_executed.append(STAGE_STEM_DEDUP)

        # Stage 4.5: Concept Validation (Pattern + SBERT)
        if self.config.enable_concept_validation:
            concepts, validation_rejected = self._apply_concept_validation(concepts)
            dedup_stats["validation_rejected"] = validation_rejected
            stages_executed.append(STAGE_CONCEPT_VALIDATION)

        # Stage 4: Semantic Dedup (HCE-4.0, AC-4.6)
        if self.config.enable_semantic_dedup:
            concepts, semantic_stats = self._apply_semantic_deduplication(concepts)
            dedup_stats["semantic_clusters"] = semantic_stats["cluster_count"]
            dedup_stats["semantic_removed"] = semantic_stats["removed_count"]
            stages_executed.append(STAGE_SEMANTIC_DEDUP)

        # TODO: Stage 5 - GraphCodeBERT (existing, optional)

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

    def _filter_noise_concepts(
        self,
        concepts: list[ExtractedTerm],
    ) -> list[ExtractedTerm]:
        """Filter out concepts containing noise words.

        Post-extraction filter to remove concepts that contain
        watermarks or other noise (AC-1.4).

        Args:
            concepts: List of extracted terms to filter.

        Returns:
            List of concepts without noise.
        """
        filtered_concepts: list[ExtractedTerm] = []

        for concept in concepts:
            # Split concept term into words and check each
            words = concept.term.split()
            batch_result = self._noise_filter.filter_batch(words)

            # Only keep concept if none of its words were rejected
            if len(batch_result.rejected) == 0:
                filtered_concepts.append(concept)

        return filtered_concepts

    def _apply_stem_deduplication(
        self,
        concepts: list[ExtractedTerm],
    ) -> tuple[list[ExtractedTerm], int]:
        """Apply morphological deduplication using stemming.

        Stage 3 of the pipeline (HCE-3.0, AC-3.5).
        Groups concepts by stem and keeps first occurrence.

        Args:
            concepts: List of extracted terms to deduplicate.

        Returns:
            Tuple of (deduplicated_concepts, removed_count).
        """
        if not concepts:
            return [], 0

        # Convert to MergerExtractedTerm for stemmer compatibility
        merger_terms = [
            MergerExtractedTerm(term=c.term, score=c.score, source=c.source)
            for c in concepts
        ]

        # Apply stem deduplication
        deduped_terms, removed_count = deduplicate_by_stem(merger_terms)

        # Convert back to pipeline's ExtractedTerm format
        result = [
            ExtractedTerm(term=t.term, score=t.score, source=t.source)  # type: ignore[arg-type]
            for t in deduped_terms
        ]

        return result, removed_count

    def _apply_concept_validation(
        self,
        concepts: list[ExtractedTerm],
    ) -> tuple[list[ExtractedTerm], int]:
        """Apply concept validation using pattern filter and SBERT.

        Stage 4.5 of the pipeline. Validates that extracted terms are
        real programming concepts, not author names or noise.

        Args:
            concepts: List of extracted terms to validate.

        Returns:
            Tuple of (validated_concepts, rejected_count).
        """
        if not concepts:
            return [], 0

        # Extract term strings for validation
        term_strings = [c.term for c in concepts]

        # Run validation
        result = self._concept_validator.validate(term_strings)

        # Filter concepts to only valid ones
        valid_set = set(result.valid_concepts)
        validated = [c for c in concepts if c.term in valid_set]

        rejected_count = len(concepts) - len(validated)
        return validated, rejected_count

    def _apply_semantic_deduplication(
        self,
        concepts: list[ExtractedTerm],
    ) -> tuple[list[ExtractedTerm], dict[str, int]]:
        """Apply semantic deduplication using SBERT + HDBSCAN.

        Stage 4 of the pipeline (HCE-4.0, AC-4.6).
        Clusters semantically similar concepts and keeps shortest term.

        Args:
            concepts: List of extracted terms to deduplicate.

        Returns:
            Tuple of (deduplicated_concepts, stats_dict).
        """
        if not concepts:
            return [], {"cluster_count": 0, "removed_count": 0}

        # Convert to MergerExtractedTerm for semantic deduplicator compatibility
        merger_terms = [
            MergerExtractedTerm(term=c.term, score=c.score, source=c.source)
            for c in concepts
        ]

        # Apply semantic deduplication
        deduped_terms, stats = self._semantic_deduplicator.deduplicate(merger_terms)

        # Convert back to pipeline's ExtractedTerm format
        result = [
            ExtractedTerm(term=t.term, score=t.score, source=t.source)  # type: ignore[arg-type]
            for t in deduped_terms
        ]

        return result, stats
