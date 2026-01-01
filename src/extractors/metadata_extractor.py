"""Metadata Extractor - WBS-1.3.

Combines TF-IDF keyword extraction, concept extraction, domain detection,
and quality scoring for unified metadata extraction.

AC Reference:
- AC-2.2: Keyword Extraction - keywords with term, score, is_technical
- AC-2.3: Concept Extraction - concepts with name, confidence, domain, tier
- AC-2.5: Quality Scoring - quality_score between 0.0-1.0
- AC-2.6: Domain Detection - detected_domain and domain_confidence
- AC-5.1: Pipeline Integration - delegates to ConceptExtractionPipeline
- HTC-1.0: Classification - validates terms through 4-tier classifier

Anti-Patterns Avoided:
- Anti-Pattern #12: Singleton pattern for extractors
- S1192: Constants extracted to module level
- S3776: Cognitive complexity < 15 (helper methods)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from src.classifiers import (
    AliasLookup,
    ClassificationResponse,
    HeuristicFilter,
    SyncTieredClassifier,
    TrainedClassifier,
)
from src.extractors.concept_extraction_pipeline import (
    ConceptExtractionConfig,
    ConceptExtractionPipeline,
    ConceptExtractionResult,
)
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
from src.validators.dictionary_validator import (
    DictionaryValidator,
    DictionaryValidationResult,
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
STAGE_DICTIONARY: Final[str] = "dictionary_validation"
STAGE_QUALITY: Final[str] = "quality"
STAGE_CLASSIFICATION: Final[str] = "classification"
STAGE_SUMMARY: Final[str] = "summary"

# Default paths
DEFAULT_TAXONOMY_PATH: Final[str] = "config/domain_taxonomy.json"
DEFAULT_STOPWORDS_PATH: Final[str] = "config/technical_stopwords.json"
PRE_FILTER_LOG_PATH: Final[str] = "logs/pre_filter_extraction.jsonl"
DEFAULT_ALIAS_LOOKUP_PATH: Final[str] = "config/alias_lookup.json"
DEFAULT_MODEL_PATH: Final[str] = "models/concept_classifier.joblib"

# Classification constants
CLASSIFICATION_CONCEPT: Final[str] = "concept"
CLASSIFICATION_KEYWORD: Final[str] = "keyword"
CLASSIFICATION_REJECTED: Final[str] = "rejected"
TIER_ALIAS_LOOKUP: Final[int] = 1


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
        enable_classification: Whether to use HTC to classify terms.
        enable_summary: Whether to generate LLM summary via inference-service.
        alias_lookup_path: Path to alias lookup JSON.
        model_path: Path to trained classifier model.
        inference_service_url: URL of inference-service for summary generation.
    """

    domain_taxonomy_path: str | None = None
    technical_stopwords_path: str | None = None
    enable_concepts: bool = True
    enable_domain_detection: bool = True
    enable_classification: bool = True
    enable_summary: bool = True
    alias_lookup_path: str | None = None
    model_path: str | None = None
    inference_service_url: str | None = None


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
        pipeline_metadata: Metadata from hybrid extraction pipeline.
        dedup_stats: Deduplication statistics from hybrid pipeline.
        classification_stats: Statistics from HTC classification.
    """

    keywords: list[KeywordResult] = field(default_factory=list)
    concepts: list[ConceptResult] = field(default_factory=list)
    summary: str | None = None
    detected_domain: str | None = None
    domain_confidence: float | None = None
    quality_score: float = 0.0
    rejected_keywords: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    text_length: int = 0
    stages_completed: list[str] = field(default_factory=list)
    pipeline_metadata: dict[str, Any] | None = None
    dedup_stats: dict[str, int] | None = None
    classification_stats: dict[str, int] | None = None
    summary_model: str | None = None
    summary_tokens: int = 0


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
    - SyncTieredClassifier for term classification (HTC-1.0)

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
        self._dictionary_validator: DictionaryValidator | None = None
        self._concept_pipeline: ConceptExtractionPipeline | None = None
        self._classifier: SyncTieredClassifier | None = None

    @property
    def _concept_pipeline_instance(self) -> ConceptExtractionPipeline:
        """Lazy-load concept extraction pipeline (Anti-Pattern #12: cached).
        
        Returns:
            Cached ConceptExtractionPipeline instance.
        """
        if self._concept_pipeline is None:
            self._concept_pipeline = ConceptExtractionPipeline()
        return self._concept_pipeline

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

    @property
    def _dictionary_validator_instance(self) -> DictionaryValidator:
        """Lazy-load dictionary validator (Anti-Pattern #12: cached)."""
        if self._dictionary_validator is None:
            self._dictionary_validator = DictionaryValidator()
        return self._dictionary_validator

    @property
    def _classifier_instance(self) -> SyncTieredClassifier | None:
        """Lazy-load sync tiered classifier (Anti-Pattern #12: cached).

        Returns None if classification is disabled or models not available.
        """
        if not self.config.enable_classification:
            return None

        if self._classifier is None:
            # Determine paths
            alias_path = Path(
                self.config.alias_lookup_path or DEFAULT_ALIAS_LOOKUP_PATH
            )
            model_path = Path(self.config.model_path or DEFAULT_MODEL_PATH)

            # Check if files exist
            if not alias_path.exists() or not model_path.exists():
                return None

            try:
                # Load components
                alias_lookup = AliasLookup(lookup_path=alias_path)
                trained_classifier = TrainedClassifier(model_path=model_path)
                heuristic_filter = HeuristicFilter()

                self._classifier = SyncTieredClassifier(
                    alias_lookup=alias_lookup,
                    trained_classifier=trained_classifier,
                    heuristic_filter=heuristic_filter,
                )
            except Exception:
                # If loading fails, disable classification
                return None

        return self._classifier

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
            book_title: Optional book title for domain inference.
            options: Extraction options.

        Returns:
            ExtractionResult with keywords, concepts, domain, quality.
        """
        start_time = time.perf_counter()
        options = options or MetadataExtractionOptions()
        
        # Store source info for pre-filter logging
        source_book = book_title or "unknown_book"
        source_chapter = title or "unknown_chapter"

        result = ExtractionResult(text_length=len(text))

        # Stage 1: Extract keywords
        raw_keywords = self._extract_keywords(text, options)
        result.stages_completed.append(STAGE_KEYWORDS)
        
        # Capture ALL raw keywords BEFORE any filtering
        raw_keyword_terms = [kw.keyword for kw in raw_keywords]

        # Stage 2: Filter noise
        if options.filter_noise:
            filtered = self._filter_noise(raw_keywords)
            result.rejected_keywords = filtered.rejected
            result.rejection_reasons = filtered.rejection_reasons
            keyword_terms = filtered.accepted
            result.stages_completed.append(STAGE_NOISE_FILTER)
        else:
            keyword_terms = [kw.keyword for kw in raw_keywords]
        
        # Capture keywords after noise filter but before dictionary validation
        post_noise_keyword_terms = keyword_terms.copy()

        # Stage 3: Dictionary validation - only accept approved keywords
        if options.validate_dictionary:
            dict_result = self._dictionary_validator_instance.validate_keywords(keyword_terms)
            # Add rejected terms to result
            for term in dict_result.rejected:
                if term not in result.rejected_keywords:
                    result.rejected_keywords.append(term)
            result.rejection_reasons.update(dict_result.rejection_reasons)
            keyword_terms = dict_result.accepted
            result.stages_completed.append(STAGE_DICTIONARY)

        # Stage 4: Build KeywordResult list (no top_k limit - extract ALL)
        result.keywords = self._build_keyword_results(keyword_terms, raw_keywords)

        # Stage 5: Extract concepts
        raw_concept_names: list[str] = []
        if self.config.enable_concepts:
            concepts, pipeline_meta, dedup = self._extract_concepts(text, options)
            
            # Capture ALL raw concepts BEFORE dictionary validation
            raw_concept_names = [c.name for c in concepts]
            
            # Dictionary validation for concepts - only accept approved concepts
            if options.validate_dictionary:
                concept_names = [c.name for c in concepts]
                concept_dict_result = self._dictionary_validator_instance.validate_concepts(concept_names)
                
                # Filter concepts to only those in approved list
                approved_concept_names = set(name.lower() for name in concept_dict_result.accepted)
                concepts = [c for c in concepts if c.name.lower() in approved_concept_names]
                
                # Add rejected concepts to rejection tracking
                for term in concept_dict_result.rejected:
                    if term not in result.rejected_keywords:
                        result.rejected_keywords.append(term)
                result.rejection_reasons.update(concept_dict_result.rejection_reasons)
            
            result.concepts = concepts
            result.pipeline_metadata = pipeline_meta
            result.dedup_stats = dedup
            result.stages_completed.append(STAGE_CONCEPTS)
        
        # MANDATORY: Log pre-filter extraction data
        self._log_pre_filter_extraction(
            book_title=source_book,
            chapter_title=source_chapter,
            raw_keywords=raw_keyword_terms,
            post_noise_keywords=post_noise_keyword_terms,
            final_keywords=[kw.term for kw in result.keywords],
            raw_concepts=raw_concept_names,
            final_concepts=[c.name for c in result.concepts],
        )

        # Stage 6: Classify terms using Hybrid Tiered Classifier
        if self.config.enable_classification:
            classification_result = self._classify_terms(
                result.keywords, result.concepts
            )
            result.keywords = classification_result["keywords"]
            result.concepts = classification_result["concepts"]
            # Add newly rejected terms
            for term, reason in classification_result["rejected"].items():
                if term not in result.rejected_keywords:
                    result.rejected_keywords.append(term)
                    result.rejection_reasons[term] = reason
            result.classification_stats = classification_result["stats"]
            result.stages_completed.append(STAGE_CLASSIFICATION)

        # Stage 6: Detect domain
        if self.config.enable_domain_detection and result.concepts:
            domain, confidence = self._detect_domain(result.concepts)
            result.detected_domain = domain
            result.domain_confidence = confidence
            result.stages_completed.append(STAGE_DOMAIN)

        # Stage 7: Calculate quality score
        result.quality_score = self._calculate_quality_score(result)
        result.stages_completed.append(STAGE_QUALITY)

        # Record timing
        end_time = time.perf_counter()
        result.processing_time_ms = (end_time - start_time) * 1000

        return result

    async def extract_async(
        self,
        text: str,
        title: str | None = None,
        book_title: str | None = None,
        options: MetadataExtractionOptions | None = None,
    ) -> ExtractionResult:
        """Extract metadata from text with async summary generation.

        Same as extract() but adds LLM-generated summary via inference-service.
        Internal microservice communication - does NOT go through Gateway.

        Args:
            text: The text to extract metadata from.
            title: Optional title for context.
            book_title: Optional book title for domain inference.
            options: Extraction options (enable_summary controls summary generation).

        Returns:
            ExtractionResult with keywords, concepts, domain, quality, and summary.
        """
        # Run synchronous extraction first
        result = self.extract(text, title, book_title, options)
        
        # Generate summary if enabled
        opts = options or MetadataExtractionOptions()
        if opts.enable_summary and self.config.enable_summary:
            summary_result = await self._generate_summary_async(text, title)
            if summary_result:
                result.summary = summary_result.summary
                result.summary_model = summary_result.model
                result.summary_tokens = summary_result.tokens_used
                result.stages_completed.append(STAGE_SUMMARY)

        return result

    async def _generate_summary_async(
        self,
        text: str,
        title: str | None = None,
    ) -> "SummaryResult | None":
        """Generate summary using inference-service.

        Internal call to inference-service (port 8085) - does NOT use Gateway.
        
        Args:
            text: Chapter text to summarize.
            title: Optional chapter title for context.

        Returns:
            SummaryResult or None if generation fails.
        """
        try:
            from src.clients.inference_client import InferenceClient
            
            inference_url = self.config.inference_service_url or "http://localhost:8085"
            async with InferenceClient(base_url=inference_url) as client:
                return await client.generate_summary(text, title)
        except Exception as e:
            # Log error but don't fail extraction
            import logging
            logging.warning(f"Summary generation failed: {e}")
            return None

    def _log_pre_filter_extraction(
        self,
        book_title: str,
        chapter_title: str,
        raw_keywords: list[str],
        post_noise_keywords: list[str],
        final_keywords: list[str],
        raw_concepts: list[str],
        final_concepts: list[str],
    ) -> None:
        """Log pre-filter extraction data to JSONL file.
        
        MANDATORY logging of all extracted terms before and after filtering
        for pipeline analysis.
        
        Args:
            book_title: Source book name.
            chapter_title: Source chapter name/number.
            raw_keywords: ALL keywords before any filtering.
            post_noise_keywords: Keywords after noise filter, before dictionary.
            final_keywords: Keywords after dictionary validation.
            raw_concepts: ALL concepts before dictionary validation.
            final_concepts: Concepts after dictionary validation.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "book": book_title,
            "chapter": chapter_title,
            "extraction": {
                "keywords": {
                    "raw_count": len(raw_keywords),
                    "raw_terms": raw_keywords,
                    "post_noise_count": len(post_noise_keywords),
                    "post_noise_terms": post_noise_keywords,
                    "final_count": len(final_keywords),
                    "final_terms": final_keywords,
                    "noise_filtered": len(raw_keywords) - len(post_noise_keywords),
                    "dict_filtered": len(post_noise_keywords) - len(final_keywords),
                },
                "concepts": {
                    "raw_count": len(raw_concepts),
                    "raw_terms": raw_concepts,
                    "final_count": len(final_concepts),
                    "final_terms": final_concepts,
                    "dict_filtered": len(raw_concepts) - len(final_concepts),
                },
            },
        }
        
        # Ensure logs directory exists
        log_path = Path(PRE_FILTER_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to JSONL file
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

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
        
        # Extract ALL keywords - no top_k limit
        # Use a very high number to get all possible candidates
        top_k = 1000000
        
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
    ) -> list[KeywordResult]:
        """Build KeywordResult list from filtered terms.

        Args:
            filtered_terms: Terms that passed noise filter.
            raw_keywords: Original keyword results with scores.

        Returns:
            List of KeywordResult sorted by score (ALL terms, no limit).
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
        
        # Sort by score descending - return ALL (no top_k limit)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results

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

    def _classify_terms(
        self,
        keywords: list[KeywordResult],
        concepts: list[ConceptResult],
    ) -> dict[str, Any]:
        """Classify all terms using Hybrid Tiered Classifier.

        This method:
        1. Runs all keywords through the classifier
        2. Promotes keywords classified as "concept" to concepts
        3. Runs all concepts through the classifier for validation
        4. Rejects any terms classified as "rejected"

        Args:
            keywords: List of keyword results to classify
            concepts: List of concept results to validate

        Returns:
            Dict with keys: keywords, concepts, rejected, stats
        """
        classifier = self._classifier_instance
        if classifier is None:
            # Classification disabled or unavailable
            return {
                "keywords": keywords,
                "concepts": concepts,
                "rejected": {},
                "stats": None,
            }

        # Track stats
        stats: dict[str, int] = {
            "total_terms": len(keywords) + len(concepts),
            "tier_1_hits": 0,
            "tier_2_hits": 0,
            "tier_3_rejections": 0,
            "keywords_promoted": 0,
        }
        rejected: dict[str, str] = {}
        final_keywords: list[KeywordResult] = []
        final_concepts: list[ConceptResult] = []

        # Track which terms are already concepts (by name) to avoid duplicates
        concept_names: set[str] = {c.name.lower() for c in concepts}

        # Classify keywords
        for kw in keywords:
            result = classifier.classify(kw.term)
            self._update_tier_stats(stats, result)

            if result.classification == CLASSIFICATION_REJECTED:
                rejected[kw.term] = result.rejection_reason or "tier_3_noise"
                stats["tier_3_rejections"] += 1
            elif result.classification == CLASSIFICATION_CONCEPT:
                # Promote to concept if not already there
                if result.canonical_term.lower() not in concept_names:
                    final_concepts.append(
                        ConceptResult(
                            name=result.canonical_term,
                            confidence=result.confidence,
                            domain="",  # Will be set by domain detection
                            tier=f"T{result.tier_used}",
                        )
                    )
                    concept_names.add(result.canonical_term.lower())
                    stats["keywords_promoted"] += 1
            else:
                # Keep as keyword
                final_keywords.append(kw)

        # Validate existing concepts
        for concept in concepts:
            result = classifier.classify(concept.name)
            self._update_tier_stats(stats, result)

            if result.classification == CLASSIFICATION_REJECTED:
                rejected[concept.name] = result.rejection_reason or "tier_3_noise"
                stats["tier_3_rejections"] += 1
            else:
                # Update with validated info
                final_concepts.append(
                    ConceptResult(
                        name=result.canonical_term,
                        confidence=max(concept.confidence, result.confidence),
                        domain=concept.domain,
                        tier=f"T{result.tier_used}" if result.tier_used == TIER_ALIAS_LOOKUP else concept.tier,
                    )
                )

        return {
            "keywords": final_keywords,
            "concepts": final_concepts,
            "rejected": rejected,
            "stats": stats,
        }

    def _update_tier_stats(
        self,
        stats: dict[str, int],
        result: ClassificationResponse,
    ) -> None:
        """Update tier hit statistics.

        Args:
            stats: Stats dict to update
            result: Classification result
        """
        if result.tier_used == TIER_ALIAS_LOOKUP:
            stats["tier_1_hits"] += 1
        elif result.tier_used == 2:
            stats["tier_2_hits"] += 1

    def _extract_concepts(
        self,
        text: str,
        options: MetadataExtractionOptions,
    ) -> tuple[list[ConceptResult], dict[str, Any] | None, dict[str, int] | None]:
        """Extract concepts using legacy extractor or hybrid pipeline.

        Args:
            text: Text to extract from.
            options: Extraction options including use_hybrid_extraction flag.

        Returns:
            Tuple of (concepts, pipeline_metadata, dedup_stats).
            pipeline_metadata and dedup_stats are None for legacy extraction.
        """
        if options.use_hybrid_extraction:
            return self._extract_concepts_hybrid(text, options)
        return self._extract_concepts_legacy(text, options)

    def _extract_concepts_legacy(
        self,
        text: str,
        options: MetadataExtractionOptions,
    ) -> tuple[list[ConceptResult], dict[str, Any] | None, dict[str, int] | None]:
        """Extract concepts using legacy ConceptExtractor.

        Args:
            text: Text to extract from.
            options: Extraction options.

        Returns:
            Tuple of (concepts, None, None) - no pipeline metadata for legacy.
        """
        extractor = self._concept_extractor_instance
        
        # ConceptExtractor.extract_concepts returns ConceptExtractionResult
        extraction_result = extractor.extract_concepts(text)
        
        # Extract ALL concepts - no top_k limit
        results: list[ConceptResult] = []
        for concept in extraction_result.concepts:
            # Filter by confidence only
            if concept.confidence >= options.min_concept_confidence:
                results.append(
                    ConceptResult(
                        name=concept.name,
                        confidence=concept.confidence,
                        domain=concept.domain,
                        tier=concept.tier or "",
                    )
                )
        
        return results, None, None

    def _extract_concepts_hybrid(
        self,
        text: str,
        options: MetadataExtractionOptions,
    ) -> tuple[list[ConceptResult], dict[str, Any], dict[str, int]]:
        """Extract concepts using hybrid ConceptExtractionPipeline.

        Args:
            text: Text to extract from.
            options: Extraction options.

        Returns:
            Tuple of (concepts, pipeline_metadata, dedup_stats).
        """
        pipeline = self._concept_pipeline_instance
        
        # Run hybrid pipeline extraction
        pipeline_result = pipeline.extract(text)
        
        # Map pipeline ExtractedTerms to ConceptResults - ALL concepts (no top_k limit)
        results: list[ConceptResult] = []
        for term in pipeline_result.concepts:
            # Invert YAKE score (lower = better) to confidence (higher = better)
            confidence = 1.0 - min(term.score, 1.0)
            if confidence >= options.min_concept_confidence:
                results.append(
                    ConceptResult(
                        name=term.term,
                        confidence=confidence,
                        domain="",  # Hybrid pipeline doesn't assign domains yet
                        tier="",
                    )
                )
        
        return results, pipeline_result.pipeline_metadata, pipeline_result.dedup_stats

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
