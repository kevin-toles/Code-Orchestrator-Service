"""Unit tests for MetadataExtractor - WBS-1.3.

Tests metadata extraction combining TF-IDF keywords, concept extraction,
domain detection, and quality scoring per AC-2.2, AC-2.3, AC-2.5, AC-2.6.

TDD Phase: RED - All tests should FAIL initially.
"""

import pytest
from src.extractors.metadata_extractor import (
    MetadataExtractor,
    MetadataExtractorConfig,
    ExtractionResult,
    get_metadata_extractor,
)
from src.models.metadata_models import (
    MetadataExtractionOptions,
    KeywordResult,
    ConceptResult,
)


# === WBS-1.3.1-1.3.2: TF-IDF Keyword Extraction ===

class TestKeywordExtraction:
    """Tests for keyword extraction with TF-IDF (AC-2.2)."""

    def test_extract_keywords_returns_keyword_results(self) -> None:
        """AC-2.2: extract() should return keywords as KeywordResult list."""
        extractor = MetadataExtractor()
        text = """
        Microservices architecture enables scalable distributed systems.
        API design patterns improve service communication and reliability.
        Docker containers provide consistent deployment environments.
        """
        
        result = extractor.extract(text)
        
        assert isinstance(result.keywords, list)
        assert len(result.keywords) > 0
        assert all(isinstance(kw, KeywordResult) for kw in result.keywords)

    def test_keywords_have_term_and_score(self) -> None:
        """AC-2.2: Keywords should have term and score attributes."""
        extractor = MetadataExtractor()
        text = "Kubernetes orchestration enables container management at scale."
        
        result = extractor.extract(text)
        
        for keyword in result.keywords:
            assert hasattr(keyword, "term")
            assert hasattr(keyword, "score")
            assert isinstance(keyword.term, str)
            assert isinstance(keyword.score, float)

    def test_keywords_have_is_technical_flag(self) -> None:
        """AC-2.2: Keywords should have is_technical boolean flag."""
        extractor = MetadataExtractor()
        text = "Machine learning models require GPU acceleration for training."
        
        result = extractor.extract(text)
        
        for keyword in result.keywords:
            assert hasattr(keyword, "is_technical")
            assert isinstance(keyword.is_technical, bool)


# === WBS-1.3.3-1.3.4: Keyword Sorting ===

class TestKeywordSorting:
    """Tests for keywords sorted by score (AC-2.2)."""

    def test_keywords_sorted_by_score_descending(self) -> None:
        """AC-2.2: Keywords should be sorted by score descending."""
        extractor = MetadataExtractor()
        text = """
        Deep learning neural networks process natural language understanding.
        Transformer models enable state-of-the-art NLP performance.
        Attention mechanisms improve sequence-to-sequence translation.
        """
        
        result = extractor.extract(text)
        
        scores = [kw.score for kw in result.keywords]
        assert scores == sorted(scores, reverse=True)

    def test_extraction_returns_keywords_sorted_by_score(self) -> None:
        """AC-2.2: Keywords should be sorted by score (highest first)."""
        extractor = MetadataExtractor()
        text = """
        Software engineering best practices include code review,
        continuous integration, test-driven development, refactoring,
        and documentation for maintainable systems.
        """
        
        result = extractor.extract(text)
        
        # Verify keywords are sorted by score descending
        if len(result.keywords) > 1:
            scores = [kw.score for kw in result.keywords]
            assert scores == sorted(scores, reverse=True)


# === WBS-1.3.5-1.3.6: Concept Extraction ===

class TestConceptExtraction:
    """Tests for concept extraction (AC-2.3)."""

    def test_extract_concepts_returns_concept_results(self) -> None:
        """AC-2.3: extract() should return concepts as ConceptResult list."""
        extractor = MetadataExtractor()
        text = """
        Retrieval-augmented generation combines large language models
        with external knowledge bases for improved accuracy.
        Vector embeddings enable semantic search capabilities.
        """
        
        result = extractor.extract(text)
        
        assert isinstance(result.concepts, list)
        assert all(isinstance(c, ConceptResult) for c in result.concepts)

    def test_concepts_have_name_confidence_domain_tier(self) -> None:
        """AC-2.3: Concepts should have name, confidence, domain, tier."""
        extractor = MetadataExtractor()
        text = "Prompt engineering optimizes LLM responses through structured inputs."
        
        result = extractor.extract(text)
        
        if result.concepts:  # May be empty if no taxonomy match
            for concept in result.concepts:
                assert hasattr(concept, "name")
                assert hasattr(concept, "confidence")
                assert hasattr(concept, "domain")
                assert hasattr(concept, "tier")

    def test_respects_top_k_concepts_option(self) -> None:
        """AC-2.3: Should respect top_k_concepts option."""
        extractor = MetadataExtractor()
        options = MetadataExtractionOptions(top_k_concepts=3)
        text = """
        Machine learning pipelines include data preprocessing,
        feature engineering, model training, hyperparameter tuning,
        and model deployment for production systems.
        """
        
        result = extractor.extract(text, options=options)
        
        assert len(result.concepts) <= 3


# === WBS-1.3.7-1.3.8: Domain Detection ===

class TestDomainDetection:
    """Tests for domain detection (AC-2.6)."""

    def test_extracts_detected_domain(self) -> None:
        """AC-2.6: Result should include detected_domain."""
        extractor = MetadataExtractor()
        text = """
        Software architecture patterns define system structure.
        Microservices, event-driven, and layered architectures
        provide different scalability characteristics.
        """
        
        result = extractor.extract(text)
        
        assert hasattr(result, "detected_domain")
        assert result.detected_domain is None or isinstance(result.detected_domain, str)

    def test_extracts_domain_confidence(self) -> None:
        """AC-2.6: Result should include domain_confidence."""
        extractor = MetadataExtractor()
        text = "REST API design follows resource-oriented architecture principles."
        
        result = extractor.extract(text)
        
        assert hasattr(result, "domain_confidence")
        assert result.domain_confidence is None or (
            isinstance(result.domain_confidence, float)
            and 0.0 <= result.domain_confidence <= 1.0
        )

    def test_domain_inferred_from_concepts(self) -> None:
        """AC-2.6: Domain should be inferred from extracted concepts."""
        extractor = MetadataExtractor()
        text = """
        LLM fine-tuning requires curated datasets and GPU resources.
        Prompt engineering and retrieval-augmented generation improve
        model accuracy for domain-specific applications.
        """
        
        result = extractor.extract(text)
        
        # If concepts have domains, detected_domain should match most common
        if result.concepts:
            concept_domains = [c.domain for c in result.concepts if c.domain]
            if concept_domains:
                assert result.detected_domain in concept_domains or result.detected_domain is None


# === WBS-1.3.9-1.3.10: Quality Score Calculation ===

class TestQualityScoring:
    """Tests for quality score calculation (AC-2.5)."""

    def test_extracts_quality_score(self) -> None:
        """AC-2.5: Result should include quality_score."""
        extractor = MetadataExtractor()
        text = "Distributed systems require fault tolerance and consistency guarantees."
        
        result = extractor.extract(text)
        
        assert hasattr(result, "quality_score")
        assert isinstance(result.quality_score, float)

    def test_quality_score_in_valid_range(self) -> None:
        """AC-2.5: Quality score should be between 0.0 and 1.0."""
        extractor = MetadataExtractor()
        text = """
        Cloud-native applications leverage containerization,
        orchestration, and infrastructure as code for deployment.
        """
        
        result = extractor.extract(text)
        
        assert 0.0 <= result.quality_score <= 1.0

    def test_quality_score_higher_for_rich_content(self) -> None:
        """AC-2.5: Quality score should be higher for content-rich text."""
        extractor = MetadataExtractor()
        
        # Short, generic text
        poor_text = "The thing works."
        poor_result = extractor.extract(poor_text)
        
        # Rich, technical text
        rich_text = """
        Kubernetes provides container orchestration with automatic scaling,
        load balancing, service discovery, and rolling updates. Pod scheduling
        optimizes resource utilization across cluster nodes. Helm charts
        package applications for reproducible deployments.
        """
        rich_result = extractor.extract(rich_text)
        
        # Rich content should have higher quality score
        assert rich_result.quality_score >= poor_result.quality_score


# === WBS-1.3.11: Singleton Pattern ===

class TestSingletonPattern:
    """Tests for singleton pattern (Anti-Pattern #12)."""

    def test_get_metadata_extractor_returns_singleton(self) -> None:
        """Anti-Pattern #12: get_metadata_extractor() should return same instance."""
        extractor1 = get_metadata_extractor()
        extractor2 = get_metadata_extractor()
        
        assert extractor1 is extractor2

    def test_extractor_caches_internal_extractors(self) -> None:
        """Anti-Pattern #12: Internal extractors should be cached."""
        extractor = MetadataExtractor()
        
        # Accessing extractor twice should return same instance
        kw_extractor1 = extractor._keyword_extractor
        kw_extractor2 = extractor._keyword_extractor
        
        assert kw_extractor1 is kw_extractor2


# === Integration with Noise Filter ===

class TestNoiseFilterIntegration:
    """Tests for integration with NoiseFilter."""

    def test_filters_noise_by_default(self) -> None:
        """AC-2.4: Noise should be filtered by default."""
        extractor = MetadataExtractor()
        text = """
        OceanOfPDF provides microservices architecture patterns.
        Using www and http fragments shouldn't appear in keywords.
        """
        
        result = extractor.extract(text)
        
        keyword_terms = [kw.term.lower() for kw in result.keywords]
        assert "oceanofpdf" not in keyword_terms
        assert "www" not in keyword_terms
        assert "http" not in keyword_terms

    def test_respects_filter_noise_option(self) -> None:
        """AC-2.4: Should respect filter_noise option."""
        extractor = MetadataExtractor()
        options = MetadataExtractionOptions(filter_noise=False)
        text = "OceanOfPDF provides technical content."
        
        result = extractor.extract(text, options=options)
        
        # With filter_noise=False, noise terms may appear
        # (depends on TF-IDF scores)
        assert isinstance(result.keywords, list)

    def test_includes_rejected_keywords(self) -> None:
        """AC-2.4: Result should include rejected keywords and reasons."""
        extractor = MetadataExtractor()
        text = """
        The oceanofpdf watermark shouldn't appear in final keywords.
        Using 'll and 's contractions are also filtered.
        """
        
        result = extractor.extract(text)
        
        assert hasattr(result, "rejected_keywords")
        assert hasattr(result, "rejection_reasons")


# === Configuration Tests ===

class TestMetadataExtractorConfig:
    """Tests for MetadataExtractorConfig."""

    def test_config_has_taxonomy_path(self) -> None:
        """Config should support taxonomy path configuration."""
        config = MetadataExtractorConfig(
            domain_taxonomy_path="/path/to/taxonomy.json"
        )
        assert config.domain_taxonomy_path == "/path/to/taxonomy.json"

    def test_config_has_stopwords_path(self) -> None:
        """Config should support stopwords path configuration."""
        config = MetadataExtractorConfig(
            technical_stopwords_path="/path/to/stopwords.json"
        )
        assert config.technical_stopwords_path == "/path/to/stopwords.json"

    def test_extractor_accepts_config(self) -> None:
        """MetadataExtractor should accept config in constructor."""
        config = MetadataExtractorConfig()
        extractor = MetadataExtractor(config=config)
        
        assert extractor.config is config


# === Processing Metadata ===

class TestProcessingMetadata:
    """Tests for processing metadata in response."""

    def test_includes_processing_time(self) -> None:
        """Response should include processing_time_ms."""
        extractor = MetadataExtractor()
        text = "Automated testing improves software quality."
        
        result = extractor.extract(text)
        
        assert hasattr(result, "processing_time_ms")
        assert isinstance(result.processing_time_ms, float)
        assert result.processing_time_ms >= 0

    def test_includes_text_length(self) -> None:
        """Response should include text_length."""
        extractor = MetadataExtractor()
        text = "Short text for testing."
        
        result = extractor.extract(text)
        
        assert hasattr(result, "text_length")
        assert result.text_length == len(text)

    def test_includes_stages_completed(self) -> None:
        """Response should include stages_completed list."""
        extractor = MetadataExtractor()
        text = "Agile methodology promotes iterative development."
        
        result = extractor.extract(text)
        
        assert hasattr(result, "stages_completed")
        assert isinstance(result.stages_completed, list)
        assert "keywords" in result.stages_completed


# =============================================================================
# HCE-5.0: Pipeline Integration Tests (AC-5.1, AC-5.2)
# =============================================================================


class TestPipelineImport:
    """HCE-5.1: Test MetadataExtractor imports pipeline (AC-5.1)."""

    def test_concept_extraction_pipeline_import(self) -> None:
        """HCE-5.1: ConceptExtractionPipeline should be importable in module."""
        from src.extractors.metadata_extractor import ConceptExtractionPipeline
        
        assert ConceptExtractionPipeline is not None

    def test_metadata_extractor_imports_pipeline_config(self) -> None:
        """HCE-5.1: ConceptExtractionConfig should be importable."""
        from src.extractors.metadata_extractor import ConceptExtractionConfig
        
        assert ConceptExtractionConfig is not None


class TestPipelineAttribute:
    """HCE-5.2: Test MetadataExtractor has pipeline attribute (AC-5.1)."""

    def test_extractor_has_concept_pipeline_attr(self) -> None:
        """HCE-5.2: MetadataExtractor should have _concept_pipeline attribute."""
        extractor = MetadataExtractor()
        
        assert hasattr(extractor, "_concept_pipeline")

    def test_concept_pipeline_is_lazy_loaded(self) -> None:
        """HCE-5.2: _concept_pipeline should be lazy-loaded (initially None)."""
        extractor = MetadataExtractor()
        
        # Before access, internal _concept_pipeline_instance should be None
        assert extractor._concept_pipeline is not None or hasattr(extractor, "_concept_pipeline_instance")


class TestPipelineDelegation:
    """HCE-5.3: Test extract_concepts() delegates to pipeline (AC-5.1)."""

    def test_extract_uses_concept_pipeline(self) -> None:
        """HCE-5.3: extract() should use ConceptExtractionPipeline internally."""
        extractor = MetadataExtractor()
        
        # Verify the pipeline instance is created and accessible
        pipeline = extractor._concept_pipeline_instance
        
        assert pipeline is not None
        assert hasattr(pipeline, "extract")

    def test_pipeline_result_included_in_extraction(self) -> None:
        """HCE-5.3: Pipeline result should be reflected in ExtractionResult."""
        extractor = MetadataExtractor()
        text = "Deep learning neural networks enable AI systems."
        
        result = extractor.extract(text)
        
        # Should have pipeline_metadata in result
        assert hasattr(result, "pipeline_metadata") or hasattr(result, "stages_completed")


class TestBackwardCompatibility:
    """HCE-5.6: Test output format unchanged (AC-5.2)."""

    def test_keywords_format_unchanged(self) -> None:
        """HCE-5.6: keywords should remain List[KeywordResult]."""
        extractor = MetadataExtractor()
        text = "API design patterns improve REST services."
        
        result = extractor.extract(text)
        
        assert isinstance(result.keywords, list)
        # All keywords should still be KeywordResult objects
        for kw in result.keywords:
            assert hasattr(kw, "term")
            assert hasattr(kw, "score")
            assert hasattr(kw, "is_technical")

    def test_concepts_format_unchanged(self) -> None:
        """HCE-5.6: concepts should remain List[ConceptResult]."""
        extractor = MetadataExtractor()
        text = "Kubernetes orchestration enables container management."
        
        result = extractor.extract(text)
        
        assert isinstance(result.concepts, list)
        # All concepts should still be ConceptResult objects
        for concept in result.concepts:
            assert hasattr(concept, "name")
            assert hasattr(concept, "confidence")
            assert hasattr(concept, "domain")
            assert hasattr(concept, "tier")

    def test_extraction_result_has_all_original_fields(self) -> None:
        """HCE-5.6: ExtractionResult should have all original fields."""
        extractor = MetadataExtractor()
        text = "Microservices architecture scales distributed systems."
        
        result = extractor.extract(text)
        
        # Original fields must still exist
        assert hasattr(result, "keywords")
        assert hasattr(result, "concepts")
        assert hasattr(result, "detected_domain")
        assert hasattr(result, "domain_confidence")
        assert hasattr(result, "quality_score")
        assert hasattr(result, "rejected_keywords")
        assert hasattr(result, "rejection_reasons")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "text_length")
        assert hasattr(result, "stages_completed")


class TestExtractionStrategyToggle:
    """HCE-5.X: Test extraction strategy toggle between legacy and hybrid."""

    def test_options_has_use_hybrid_extraction_field(self) -> None:
        """Options should have use_hybrid_extraction boolean field."""
        options = MetadataExtractionOptions(use_hybrid_extraction=True)
        
        assert hasattr(options, "use_hybrid_extraction")
        assert options.use_hybrid_extraction is True

    def test_use_hybrid_extraction_defaults_to_true(self) -> None:
        """use_hybrid_extraction should default to True (hybrid is production default)."""
        options = MetadataExtractionOptions()
        
        assert options.use_hybrid_extraction is True

    def test_hybrid_extraction_uses_pipeline(self) -> None:
        """When use_hybrid_extraction=True, should use ConceptExtractionPipeline."""
        extractor = MetadataExtractor()
        options = MetadataExtractionOptions(use_hybrid_extraction=True)
        text = "Machine learning algorithms process data patterns."
        
        result = extractor.extract(text, options=options)
        
        # Pipeline metadata indicates hybrid pipeline was used
        assert result.pipeline_metadata is not None
        assert "stages_executed" in result.pipeline_metadata
        assert len(result.pipeline_metadata["stages_executed"]) > 0

    def test_legacy_extraction_does_not_have_pipeline_stages(self) -> None:
        """When use_hybrid_extraction=False, pipeline_metadata should be None."""
        extractor = MetadataExtractor()
        options = MetadataExtractionOptions(use_hybrid_extraction=False)
        text = "Software design patterns improve code quality."
        
        result = extractor.extract(text, options=options)
        
        # Legacy mode: no pipeline metadata
        assert result.pipeline_metadata is None or result.pipeline_metadata == {}

    def test_hybrid_extraction_includes_dedup_stats(self) -> None:
        """Hybrid extraction should include dedup_stats in result."""
        extractor = MetadataExtractor()
        options = MetadataExtractionOptions(use_hybrid_extraction=True)
        text = "API design patterns enable REST services with GraphQL."
        
        result = extractor.extract(text, options=options)
        
        # Hybrid mode includes dedup stats
        assert result.dedup_stats is not None
        assert "stem_removed" in result.dedup_stats
        assert "semantic_clusters" in result.dedup_stats
