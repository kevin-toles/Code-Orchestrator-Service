"""Metadata Extractor - WBS-1.3.

Combines TF-IDF keyword extraction, concept extraction, domain detection,
and quality scoring for unified metadata extraction.

AC Reference:
- AC-2.2: Keyword Extraction - keywords with term, score, is_technical
- AC-2.3: Concept Extraction - concepts with name, confidence, domain, tier
- AC-2.5: Quality Scoring - quality_score between 0.0-1.0
- AC-2.6: Domain Detection - detected_domain and domain_confidence

Anti-Patterns Avoided:
- Anti-Pattern #12: Singleton pattern for extractors
- S1192: Constants extracted to module level
- S3776: Cognitive complexity < 15 (helper methods)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from src.models.metadata_models import (
    ConceptResult,
    KeywordResult,
    MetadataExtractionOptions,
)
from src.models.tfidf_extractor import (
    KeywordExtractionResult,
    KeywordExtractorConfig,
    TfidfKeywordExtractor,
)
from src.models.concept_extractor import (
    ConceptExtractor,
    ConceptExtractorConfig,
)
from src.validators.noise_filter import (
    NoiseFilter,
    BatchFilterResult,
)


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

# Quality score weights
WEIGHT_KEYWORD_COUNT: Final[float] = 0.3
WEIGHT_CONCEPT_COUNT: Final[float] = 0.3
WEIGHT_DOMAIN_CONFIDENCE: Final[float] = 0.2
WEIGHT_TEXT_LENGTH: Final[float] = 0.2

# Quality score thresholds
MIN_TEXT_LENGTH_FOR_QUALITY: Final[int] = 100
MAX_TEXT_LENGTH_FOR_QUALITY: Final[int] = 10000
MIN_KEYWORDS_FOR_QUALITY: Final[int] = 3
MAX_KEYWORDS_FOR_QUALITY: Final[int] = 20
MIN_CONCEPTS_FOR_QUALITY: Final[int] = 1
MAX_CONCEPTS_FOR_QUALITY: Final[int] = 10

# Stages
STAGE_KEYWORDS: Final[str] = "keywords"
STAGE_CONCEPTS: Final[str] = "concepts"
STAGE_DOMAIN: Final[str] = "domain"
STAGE_NOISE_FILTER: Final[str] = "noise_filter"
STAGE_QUALITY: Final[str] = "quality"

# Default paths
DEFAULT_TAXONOMY_PATH: Final[str] = "config/domain_taxonomy.json"
DEFAULT_STOPWORDS_PATH: Final[str] = "config/technical_stopwords.json"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MetadataExtractorConfig:
    """Configuration for MetadataExtractor.

    Attributes:
        domain_taxonomy_path: Path to domain taxonomy JSON.
        technical_stopwords_path: Path to technical stopwords JSON.
        enable_concepts: Whether to extract concepts.
        enable_domain_detection: Whether to detect domain.
    """

    domain_taxonomy_path: str | None = None
    technical_stopwords_path: str | None = None
    enable_concepts: bool = True
    enable_domain_detection: bool = True


@dataclass
class ExtractionResult:
    """Result of metadata extraction.

    Attributes:
        keywords: List of extracted keywords with scores.
        concepts: List of extracted concepts with confidence.
        detected_domain: Detected domain from concepts.
        domain_confidence: Confidence of domain detection.
        quality_score: Overall quality score 0.0-1.0.
        rejected_keywords: List of rejected keyword terms.
        rejection_reasons: Map of term to rejection reason.
        processing_time_ms: Processing time in milliseconds.
        text_length: Length of input text.
        stages_completed: List of completed processing stages.
    """

    keywords: list[KeywordResult] = field(default_factory=list)
    concepts: list[ConceptResult] = field(default_factory=list)
    detected_domain: str | None = None
    domain_confidence: float | None = None
    quality_score: float = 0.0
    rejected_keywords: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    text_length: int = 0
    stages_completed: list[str] = field(default_factory=list)


# =============================================================================
# Singleton Instance (Anti-Pattern #12)
# =============================================================================

_metadata_extractor: MetadataExtractor | None = None


def get_metadata_extractor(
    config: MetadataExtractorConfig | None = None,
) -> MetadataExtractor:
    """Get or create cached MetadataExtractor instance.

    Implements singleton pattern per Anti-Pattern #12:
    extractors should be cached, not created per request.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        Cached MetadataExtractor instance.
    """
    global _metadata_extractor
    if _metadata_extractor is None:
        _metadata_extractor = MetadataExtractor(config=config)
    return _metadata_extractor


# =============================================================================
# MetadataExtractor Class
# =============================================================================


class MetadataExtractor:
    """Unified metadata extractor combining keywords, concepts, and quality.

    Integrates:
    - TfidfKeywordExtractor for keyword extraction (AC-2.2)
    - ConceptExtractor for concept extraction (AC-2.3)
    - NoiseFilter for noise filtering (AC-2.4)
    - Quality scoring algorithm (AC-2.5)
    - Domain detection from concepts (AC-2.6)

    Attributes:
        config: Extractor configuration.
    """

    def __init__(self, config: MetadataExtractorConfig | None = None) -> None:
        """Initialize MetadataExtractor.

        Args:
            config: Optional configuration.
        """
        self.config = config or MetadataExtractorConfig()
        self._keyword_extractor: TfidfKeywordExtractor | None = None
        self._concept_extractor: ConceptExtractor | None = None
        self._noise_filter: NoiseFilter | None = None

    @property
    def _keyword_extractor_instance(self) -> TfidfKeywordExtractor:
        """Lazy-load keyword extractor (Anti-Pattern #12: cached)."""
        if self._keyword_extractor is None:
            kw_config = KeywordExtractorConfig()
            if self.config.technical_stopwords_path:
                path = Path(self.config.technical_stopwords_path)
                if path.exists():
                    kw_config.custom_stopwords_path = path
            self._keyword_extractor = TfidfKeywordExtractor(config=kw_config)
        return self._keyword_extractor

    @property
    def _concept_extractor_instance(self) -> ConceptExtractor:
        """Lazy-load concept extractor (Anti-Pattern #12: cached)."""
        if self._concept_extractor is None:
            concept_config = ConceptExtractorConfig()
            if self.config.domain_taxonomy_path:
                path = Path(self.config.domain_taxonomy_path)
                if path.exists():
                    concept_config.domain_taxonomy_path = path
            self._concept_extractor = ConceptExtractor(config=concept_config)
        return self._concept_extractor

    @property
    def _noise_filter_instance(self) -> NoiseFilter:
        """Lazy-load noise filter (Anti-Pattern #12: cached)."""
        if self._noise_filter is None:
            self._noise_filter = NoiseFilter()
        return self._noise_filter

    def extract(
        self,
        text: str,
        title: str | None = None,
        book_title: str | None = None,
        options: MetadataExtractionOptions | None = None,
    ) -> ExtractionResult:
        """Extract metadata from text.

        Args:
            text: The text to extract metadata from.
            title: Optional title for context.
            book_title: Optional book title for context.
            options: Extraction options.

        Returns:
            ExtractionResult with keywords, concepts, domain, quality.
        """
        start_time = time.perf_counter()
        options = options or MetadataExtractionOptions()

        result = ExtractionResult(text_length=len(text))

        # Stage 1: Extract keywords
        raw_keywords = self._extract_keywords(text, options)
        result.stages_completed.append(STAGE_KEYWORDS)

        # Stage 2: Filter noise
        if options.filter_noise:
            filtered = self._filter_noise(raw_keywords)
            result.rejected_keywords = filtered.rejected
            result.rejection_reasons = filtered.rejection_reasons
            keyword_terms = filtered.accepted
            result.stages_completed.append(STAGE_NOISE_FILTER)
        else:
            keyword_terms = [kw.keyword for kw in raw_keywords]

        # Stage 3: Build KeywordResult list
        result.keywords = self._build_keyword_results(
            keyword_terms, raw_keywords, options.top_k_keywords
        )

        # Stage 4: Extract concepts
        if self.config.enable_concepts:
            result.concepts = self._extract_concepts(text, options)
            result.stages_completed.append(STAGE_CONCEPTS)

        # Stage 5: Detect domain
        if self.config.enable_domain_detection and result.concepts:
            domain, confidence = self._detect_domain(result.concepts)
            result.detected_domain = domain
            result.domain_confidence = confidence
            result.stages_completed.append(STAGE_DOMAIN)

        # Stage 6: Calculate quality score
        result.quality_score = self._calculate_quality_score(result)
        result.stages_completed.append(STAGE_QUALITY)

        # Record timing
        end_time = time.perf_counter()
        result.processing_time_ms = (end_time - start_time) * 1000

        return result

    def _extract_keywords(
        self,
        text: str,
        options: MetadataExtractionOptions,
    ) -> list[KeywordExtractionResult]:
        """Extract keywords using TF-IDF.

        Args:
            text: Text to extract from.
            options: Extraction options.

        Returns:
            List of keyword extraction results.
        """
        extractor = self._keyword_extractor_instance
        
        # Extract more than needed to allow for filtering
        top_k = options.top_k_keywords * 2
        
        # TfidfKeywordExtractor expects a corpus (list of strings)
        corpus = [text]
        results_per_doc = extractor.extract_keywords_with_scores(
            corpus, top_k=top_k
        )
        
        # Return results for the single document
        return results_per_doc[0] if results_per_doc else []

    def _filter_noise(
        self,
        keywords: list[KeywordExtractionResult],
    ) -> BatchFilterResult:
        """Filter noise from keywords.

        Args:
            keywords: Raw keyword results.

        Returns:
            BatchFilterResult with accepted/rejected terms.
        """
        terms = [kw.keyword for kw in keywords]
        return self._noise_filter_instance.filter_batch(terms)

    def _build_keyword_results(
        self,
        filtered_terms: list[str],
        raw_keywords: list[KeywordExtractionResult],
        top_k: int,
    ) -> list[KeywordResult]:
        """Build KeywordResult list from filtered terms.

        Args:
            filtered_terms: Terms that passed noise filter.
            raw_keywords: Original keyword results with scores.
            top_k: Maximum keywords to return.

        Returns:
            List of KeywordResult sorted by score.
        """
        # Create score lookup
        score_map = {kw.keyword: kw.score for kw in raw_keywords}
        
        results: list[KeywordResult] = []
        for term in filtered_terms:
            score = score_map.get(term, 0.0)
            results.append(
                KeywordResult(
                    term=term,
                    score=score,
                    is_technical=self._is_technical_term(term),
                    sources=["tfidf"],
                )
            )
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]

    def _is_technical_term(self, term: str) -> bool:
        """Check if term is technical.

        Simple heuristic: contains underscore, hyphen, or common
        technical patterns.

        Args:
            term: Term to check.

        Returns:
            True if term appears technical.
        """
        technical_patterns = [
            "_",  # snake_case
            "-",  # kebab-case
            "api",
            "http",
            "json",
            "xml",
            "sql",
            "ml",
            "ai",
            "llm",
            "gpu",
            "cpu",
        ]
        term_lower = term.lower()
        return any(p in term_lower for p in technical_patterns)

    def _extract_concepts(
        self,
        text: str,
        options: MetadataExtractionOptions,
    ) -> list[ConceptResult]:
        """Extract concepts using ConceptExtractor.

        Args:
            text: Text to extract from.
            options: Extraction options.

        Returns:
            List of ConceptResult.
        """
        extractor = self._concept_extractor_instance
        
        # ConceptExtractor.extract_concepts returns ConceptExtractionResult
        extraction_result = extractor.extract_concepts(text)
        
        results: list[ConceptResult] = []
        for concept in extraction_result.concepts[:options.top_k_concepts]:
            # Filter by confidence
            if concept.confidence >= options.min_concept_confidence:
                results.append(
                    ConceptResult(
                        name=concept.name,
                        confidence=concept.confidence,
                        domain=concept.domain,
                        tier=concept.tier or "",
                    )
                )
        
        return results

    def _detect_domain(
        self,
        concepts: list[ConceptResult],
    ) -> tuple[str | None, float | None]:
        """Detect domain from concepts.

        Uses most common domain among extracted concepts.

        Args:
            concepts: Extracted concepts.

        Returns:
            Tuple of (domain, confidence).
        """
        if not concepts:
            return None, None

        # Count domain occurrences, weighted by confidence
        domain_scores: dict[str, float] = {}
        for concept in concepts:
            if concept.domain:
                domain_scores[concept.domain] = (
                    domain_scores.get(concept.domain, 0.0) + concept.confidence
                )

        if not domain_scores:
            return None, None

        # Find domain with highest weighted score
        best_domain = max(domain_scores, key=lambda d: domain_scores[d])
        total_confidence = sum(domain_scores.values())
        confidence = domain_scores[best_domain] / total_confidence if total_confidence > 0 else 0.0

        return best_domain, round(confidence, 3)

    def _calculate_quality_score(self, result: ExtractionResult) -> float:
        """Calculate quality score for extraction result.

        Formula per CME_ARCHITECTURE.md Section 5:
        quality = (kw_score * 0.3) + (concept_score * 0.3) +
                  (domain_score * 0.2) + (length_score * 0.2)

        Args:
            result: Extraction result.

        Returns:
            Quality score 0.0-1.0.
        """
        # Keyword score: ratio of keywords to max expected
        kw_count = len(result.keywords)
        kw_score = min(
            kw_count / MAX_KEYWORDS_FOR_QUALITY,
            1.0,
        ) if kw_count >= MIN_KEYWORDS_FOR_QUALITY else kw_count / MIN_KEYWORDS_FOR_QUALITY * 0.5

        # Concept score: ratio of concepts to max expected
        concept_count = len(result.concepts)
        if concept_count >= MIN_CONCEPTS_FOR_QUALITY:
            concept_score = min(
                concept_count / MAX_CONCEPTS_FOR_QUALITY,
                1.0,
            )
        else:
            concept_score = 0.0

        # Domain score: domain confidence or 0
        domain_score = result.domain_confidence or 0.0

        # Length score: normalized text length
        length = result.text_length
        if length < MIN_TEXT_LENGTH_FOR_QUALITY:
            length_score = length / MIN_TEXT_LENGTH_FOR_QUALITY * 0.5
        elif length > MAX_TEXT_LENGTH_FOR_QUALITY:
            length_score = 1.0
        else:
            length_score = (length - MIN_TEXT_LENGTH_FOR_QUALITY) / (
                MAX_TEXT_LENGTH_FOR_QUALITY - MIN_TEXT_LENGTH_FOR_QUALITY
            ) * 0.5 + 0.5

        # Weighted sum
        quality = (
            kw_score * WEIGHT_KEYWORD_COUNT
            + concept_score * WEIGHT_CONCEPT_COUNT
            + domain_score * WEIGHT_DOMAIN_CONFIDENCE
            + length_score * WEIGHT_TEXT_LENGTH
        )

        return round(min(max(quality, 0.0), 1.0), 3)
