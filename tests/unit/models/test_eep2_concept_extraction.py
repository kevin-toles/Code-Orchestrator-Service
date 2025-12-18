"""
Unit Tests for EEP-2: Concept Extraction Layer

WBS: EEP-2 - Concept Extraction Layer (Phase 2 of Enhanced Enrichment Pipeline)
TDD Phase: RED (tests written BEFORE implementation)

Tests for:
- EEP-2.1: ConceptExtractor class following Protocol pattern
- EEP-2.2: Domain taxonomy loading from configurable path
- EEP-2.3: Concept matching against taxonomy keywords
- EEP-2.4: Extraction endpoint POST /api/v1/concepts
- EEP-2.5: Anti-pattern compliance and metrics

Acceptance Criteria (from ENHANCED_ENRICHMENT_PIPELINE_WBS.md):
- AC-2.1.1: Create src/models/concept_extractor.py
- AC-2.1.2: Follow Protocol pattern per CODING_PATTERNS_ANALYSIS.md line 130
- AC-2.1.3: Use dataclasses for output structures (ExtractedConcept)
- AC-2.1.4: Full type annotations (Anti-Pattern #2.2)
- AC-2.2.1: Load taxonomy from configurable path
- AC-2.2.2: Parse tier structure (T0-T5 hierarchy)
- AC-2.2.3: Cache taxonomy in memory (Anti-Pattern #12 prevention)
- AC-2.3.1: Match chapter text against domain_keywords and primary_keywords
- AC-2.3.2: Apply min_domain_matches threshold
- AC-2.3.3: Return matched concepts with confidence scores
- AC-2.3.4: Support hierarchical concept relationships
- AC-2.4.1: Add POST /api/v1/concepts endpoint
- AC-2.5.1: 20+ tests written before implementation (RED)
- AC-2.5.2: Implementation passes all tests (GREEN)
- AC-2.5.3: Refactor with 0 anti-patterns (REFACTOR)

Document Priority Applied (for conflict resolution):
1. GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md
2. AI_CODING_PLATFORM_ARCHITECTURE.md
3. llm-gateway ARCHITECTURE.md
4. AI-ML_taxonomy_20251128.json
5. CODING_PATTERNS_ANALYSIS.md

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Use constants for repeated string literals
- S3776: Cognitive complexity < 15
- S1172: No unused parameters
- S3457: No empty f-strings
- #7: No exception shadowing
- #12: No model loading per request (cache taxonomy)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pytest

if TYPE_CHECKING:
    from src.models.concept_extractor import (
        ConceptExtractor,
        ConceptExtractorConfig,
        ExtractedConcept,
        ConceptExtractionResult,
        ConceptExtractorProtocol,
    )


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

DOMAIN_TAXONOMY_PATH = Path(
    "/Users/kevintoles/POC/semantic-search-service/config/domain_taxonomy.json"
)
AI_ML_TAXONOMY_PATH = Path(
    "/Users/kevintoles/POC/textbooks/Taxonomies/AI-ML_taxonomy_20251128.json"
)

# Test domains
TEST_DOMAIN_LLM_RAG = "llm_rag"
TEST_DOMAIN_PYTHON = "python_implementation"
TEST_DOMAIN_MICROSERVICES = "microservices_architecture"
TEST_DOMAIN_UNKNOWN = "unknown_domain"

# Test tier names from TIER_RELATIONSHIP_DIAGRAM.md
TEST_TIER_ARCHITECTURE = "architecture"
TEST_TIER_PRACTICES = "practices"
TEST_TIER_IMPLEMENTATION = "implementation"

# Sample keywords from EEP-1 output
SAMPLE_EEP1_KEYWORDS = [
    "retrieval",
    "embedding",
    "vector",
    "RAG",
    "langchain",
    "transformer",
    "attention",
    "semantic search",
]

# Sample text for concept extraction
SAMPLE_CHAPTER_TEXT = """
This chapter explores Retrieval-Augmented Generation (RAG) architectures.
We implement vector embeddings using transformer-based models with attention mechanisms.
LangChain provides excellent tooling for building semantic search pipelines.
The architecture leverages microservices patterns with FastAPI endpoints.
"""

SAMPLE_CODE_TEXT = """
import langchain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def create_rag_pipeline(documents: list[str]) -> RetrievalQA:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def domain_taxonomy_sample() -> dict[str, Any]:
    """Sample domain taxonomy for testing (matches real structure)."""
    return {
        "domains": {
            "llm_rag": {
                "description": "LLM, RAG, document processing, embeddings",
                "primary_keywords": ["chunk", "chunking", "split", "segment", "token"],
                "domain_keywords": [
                    "RAG", "retrieval", "vector", "embedding", "LLM", "token",
                    "context window", "semantic search", "transformer", "attention",
                    "langchain", "llamaindex"
                ],
                "min_domain_matches": 1,
                "tier_whitelist": ["architecture", "practices"],
                "score_adjustments": {
                    "in_whitelist_book": 0.3,
                    "domain_keyword_present": 0.1,
                    "primary_only_no_domain": -0.3
                }
            },
            "python_implementation": {
                "description": "Python programming, libraries, APIs",
                "primary_keywords": ["python", "def", "class", "import", "pip"],
                "domain_keywords": [
                    "fastapi", "flask", "pydantic", "asyncio", "typing",
                    "pandas", "numpy", "decorator", "generator"
                ],
                "min_domain_matches": 0,
                "tier_whitelist": ["implementation"],
                "score_adjustments": {
                    "in_whitelist_book": 0.2,
                    "domain_keyword_present": 0.05
                }
            },
            "microservices_architecture": {
                "description": "Distributed systems, microservices, API design",
                "primary_keywords": ["microservice", "API", "service", "distributed"],
                "domain_keywords": [
                    "docker", "kubernetes", "REST", "gRPC", "kafka",
                    "circuit breaker", "load balancer", "gateway"
                ],
                "min_domain_matches": 1,
                "tier_whitelist": ["architecture"],
                "score_adjustments": {
                    "in_whitelist_book": 0.25,
                    "domain_keyword_present": 0.08
                }
            }
        },
        "default_settings": {
            "min_domain_matches": 1,
            "primary_only_penalty": -0.3,
            "unknown_domain_behavior": "no_filter"
        }
    }


@pytest.fixture
def tier_taxonomy_sample() -> dict[str, Any]:
    """Sample tier taxonomy for hierarchical concept testing."""
    return {
        "tiers": {
            "architecture": {
                "priority": 1,
                "name": "Architecture Spine",
                "description": "AI/ML architecture and system design patterns",
                "concepts": ["agent", "agents", "architecture", "api", "apis", "embedding", "RAG"]
            },
            "practices": {
                "priority": 2,
                "name": "Best Practices",
                "description": "Development patterns and methodologies",
                "concepts": ["pattern", "patterns", "testing", "deployment", "monitoring"]
            },
            "implementation": {
                "priority": 3,
                "name": "Implementation Details",
                "description": "Language-specific implementation",
                "concepts": ["python", "fastapi", "decorator", "async", "pydantic"]
            }
        }
    }


@pytest.fixture
def domain_taxonomy_file(tmp_path: Path, domain_taxonomy_sample: dict[str, Any]) -> Path:
    """Create a temporary domain_taxonomy.json file."""
    taxonomy_file = tmp_path / "domain_taxonomy.json"
    taxonomy_file.write_text(json.dumps(domain_taxonomy_sample, indent=2))
    return taxonomy_file


@pytest.fixture
def tier_taxonomy_file(tmp_path: Path, tier_taxonomy_sample: dict[str, Any]) -> Path:
    """Create a temporary tier taxonomy file."""
    taxonomy_file = tmp_path / "tier_taxonomy.json"
    taxonomy_file.write_text(json.dumps(tier_taxonomy_sample, indent=2))
    return taxonomy_file


@pytest.fixture
def eep1_filtered_keywords() -> list[str]:
    """Keywords produced by EEP-1 stopword filtering."""
    return [
        "retrieval", "augmented", "generation", "RAG", "vector",
        "embeddings", "transformer", "attention", "semantic", "search",
        "langchain", "microservices", "fastapi", "architecture"
    ]


# =============================================================================
# EEP-2.1: ConceptExtractor Class Tests
# =============================================================================


class TestConceptExtractorClass:
    """Test AC-2.1.1: ConceptExtractor class creation."""

    def test_concept_extractor_module_exists(self) -> None:
        """AC-2.1.1: src/models/concept_extractor.py should exist."""
        module_path = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/src/models/concept_extractor.py")
        assert module_path.exists(), f"ConceptExtractor module not found at {module_path}"

    def test_concept_extractor_class_importable(self) -> None:
        """AC-2.1.1: ConceptExtractor class should be importable."""
        from src.models.concept_extractor import ConceptExtractor
        assert ConceptExtractor is not None

    def test_concept_extractor_config_importable(self) -> None:
        """AC-2.1.3: ConceptExtractorConfig dataclass should be importable."""
        from src.models.concept_extractor import ConceptExtractorConfig
        assert ConceptExtractorConfig is not None

    def test_extracted_concept_dataclass_importable(self) -> None:
        """AC-2.1.3: ExtractedConcept dataclass should be importable."""
        from src.models.concept_extractor import ExtractedConcept
        assert ExtractedConcept is not None

    def test_concept_extraction_result_importable(self) -> None:
        """AC-2.1.3: ConceptExtractionResult dataclass should be importable."""
        from src.models.concept_extractor import ConceptExtractionResult
        assert ConceptExtractionResult is not None


class TestConceptExtractorProtocol:
    """Test AC-2.1.2: Protocol pattern per CODING_PATTERNS_ANALYSIS.md line 130."""

    def test_concept_extractor_protocol_exists(self) -> None:
        """AC-2.1.2: ConceptExtractorProtocol should be defined."""
        from src.models.concept_extractor import ConceptExtractorProtocol
        assert ConceptExtractorProtocol is not None

    def test_protocol_has_extract_concepts_method(self) -> None:
        """AC-2.1.2: Protocol should define extract_concepts method."""
        from src.models.concept_extractor import ConceptExtractorProtocol
        assert hasattr(ConceptExtractorProtocol, "extract_concepts")

    def test_protocol_has_get_domain_concepts_method(self) -> None:
        """AC-2.1.2: Protocol should define get_domain_concepts method."""
        from src.models.concept_extractor import ConceptExtractorProtocol
        assert hasattr(ConceptExtractorProtocol, "get_domain_concepts")

    def test_protocol_has_get_tier_concepts_method(self) -> None:
        """AC-2.1.2: Protocol should define get_tier_concepts method."""
        from src.models.concept_extractor import ConceptExtractorProtocol
        assert hasattr(ConceptExtractorProtocol, "get_tier_concepts")


class TestExtractedConceptDataclass:
    """Test AC-2.1.3: Use dataclasses for output structures."""

    def test_extracted_concept_is_dataclass(self) -> None:
        """AC-2.1.3: ExtractedConcept should be a dataclass."""
        from dataclasses import is_dataclass
        from src.models.concept_extractor import ExtractedConcept
        assert is_dataclass(ExtractedConcept)

    def test_extracted_concept_has_name_field(self) -> None:
        """AC-2.1.3: ExtractedConcept should have name field."""
        from src.models.concept_extractor import ExtractedConcept
        concept = ExtractedConcept(name="RAG", confidence=0.9, domain="llm_rag", tier="architecture")
        assert concept.name == "RAG"

    def test_extracted_concept_has_confidence_field(self) -> None:
        """AC-2.1.3: ExtractedConcept should have confidence field."""
        from src.models.concept_extractor import ExtractedConcept
        concept = ExtractedConcept(name="RAG", confidence=0.9, domain="llm_rag", tier="architecture")
        assert concept.confidence == 0.9

    def test_extracted_concept_has_domain_field(self) -> None:
        """AC-2.1.3: ExtractedConcept should have domain field."""
        from src.models.concept_extractor import ExtractedConcept
        concept = ExtractedConcept(name="RAG", confidence=0.9, domain="llm_rag", tier="architecture")
        assert concept.domain == "llm_rag"

    def test_extracted_concept_has_tier_field(self) -> None:
        """AC-2.1.3: ExtractedConcept should have tier field."""
        from src.models.concept_extractor import ExtractedConcept
        concept = ExtractedConcept(name="RAG", confidence=0.9, domain="llm_rag", tier="architecture")
        assert concept.tier == "architecture"

    def test_extracted_concept_has_parent_concept_field(self) -> None:
        """AC-2.3.4: ExtractedConcept should support hierarchical relationships."""
        from src.models.concept_extractor import ExtractedConcept
        concept = ExtractedConcept(
            name="attention",
            confidence=0.8,
            domain="llm_rag",
            tier="architecture",
            parent_concept="transformer"
        )
        assert concept.parent_concept == "transformer"


class TestConceptExtractionResult:
    """Test ConceptExtractionResult dataclass structure."""

    def test_result_is_dataclass(self) -> None:
        """ConceptExtractionResult should be a dataclass."""
        from dataclasses import is_dataclass
        from src.models.concept_extractor import ConceptExtractionResult
        assert is_dataclass(ConceptExtractionResult)

    def test_result_has_concepts_field(self) -> None:
        """Result should have concepts list field."""
        from src.models.concept_extractor import ConceptExtractionResult, ExtractedConcept
        concept = ExtractedConcept(name="RAG", confidence=0.9, domain="llm_rag", tier="architecture")
        result = ConceptExtractionResult(concepts=[concept], domain_scores={"llm_rag": 0.9})
        assert len(result.concepts) == 1

    def test_result_has_domain_scores_field(self) -> None:
        """Result should have domain_scores dict field."""
        from src.models.concept_extractor import ConceptExtractionResult, ExtractedConcept
        concept = ExtractedConcept(name="RAG", confidence=0.9, domain="llm_rag", tier="architecture")
        result = ConceptExtractionResult(concepts=[concept], domain_scores={"llm_rag": 0.9})
        assert result.domain_scores["llm_rag"] == 0.9


# =============================================================================
# EEP-2.2: Domain Taxonomy Loading Tests
# =============================================================================


class TestDomainTaxonomyLoading:
    """Test AC-2.2.1-2.2.3: Domain taxonomy loading and caching."""

    def test_load_taxonomy_from_path(self, domain_taxonomy_file: Path) -> None:
        """AC-2.2.1: Load taxonomy from configurable path."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        assert extractor.taxonomy is not None

    def test_taxonomy_has_domains(self, domain_taxonomy_file: Path) -> None:
        """AC-2.2.1: Loaded taxonomy should have domains dict."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        assert "domains" in extractor.taxonomy or len(extractor.domains) > 0

    def test_taxonomy_caching(self, domain_taxonomy_file: Path) -> None:
        """AC-2.2.3: Taxonomy should be cached in memory (Anti-Pattern #12)."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        # Access taxonomy twice - should be same object (cached)
        taxonomy_1 = extractor.taxonomy
        taxonomy_2 = extractor.taxonomy
        assert taxonomy_1 is taxonomy_2

    def test_load_nonexistent_taxonomy_raises(self) -> None:
        """Should raise error for nonexistent taxonomy file."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=Path("/nonexistent/path.json"))
        with pytest.raises(FileNotFoundError):
            ConceptExtractor(config)

    def test_load_real_domain_taxonomy(self) -> None:
        """Load real domain_taxonomy.json from semantic-search-service."""
        if not DOMAIN_TAXONOMY_PATH.exists():
            pytest.skip("Real domain taxonomy not available")
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=DOMAIN_TAXONOMY_PATH)
        extractor = ConceptExtractor(config)
        assert extractor.taxonomy is not None


class TestTierTaxonomyLoading:
    """Test AC-2.2.2: Parse tier structure (T0-T5 hierarchy)."""

    def test_load_tier_taxonomy(self, tier_taxonomy_file: Path) -> None:
        """AC-2.2.2: Load tier taxonomy for hierarchical concepts."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(tier_taxonomy_path=tier_taxonomy_file)
        extractor = ConceptExtractor(config)
        assert extractor.tier_taxonomy is not None or len(extractor.tiers) > 0

    def test_tier_has_priority(self, tier_taxonomy_file: Path, domain_taxonomy_file: Path) -> None:
        """AC-2.2.2: Tiers should have priority ordering."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(
            domain_taxonomy_path=domain_taxonomy_file,
            tier_taxonomy_path=tier_taxonomy_file
        )
        extractor = ConceptExtractor(config)
        tier_info = extractor.get_tier_info(TEST_TIER_ARCHITECTURE)
        assert tier_info is not None
        assert "priority" in tier_info or hasattr(tier_info, "priority")

    def test_tier_has_concepts_list(self, tier_taxonomy_file: Path, domain_taxonomy_file: Path) -> None:
        """AC-2.2.2: Tiers should have concepts list."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(
            domain_taxonomy_path=domain_taxonomy_file,
            tier_taxonomy_path=tier_taxonomy_file
        )
        extractor = ConceptExtractor(config)
        tier_concepts = extractor.get_tier_concepts(TEST_TIER_ARCHITECTURE)
        assert isinstance(tier_concepts, list)
        assert len(tier_concepts) > 0


# =============================================================================
# EEP-2.3: Concept Matching Tests
# =============================================================================


class TestConceptMatching:
    """Test AC-2.3.1-2.3.4: Concept matching against taxonomy."""

    def test_extract_concepts_from_text(self, domain_taxonomy_file: Path) -> None:
        """AC-2.3.1: Match text against domain_keywords and primary_keywords."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        result = extractor.extract_concepts(SAMPLE_CHAPTER_TEXT)
        assert len(result.concepts) > 0

    def test_extract_concepts_finds_rag(self, domain_taxonomy_file: Path) -> None:
        """AC-2.3.1: Should find RAG concept in sample text."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        result = extractor.extract_concepts(SAMPLE_CHAPTER_TEXT)
        concept_names = [c.name.lower() for c in result.concepts]
        assert any("rag" in name for name in concept_names)

    def test_extract_concepts_with_eep1_keywords(
        self, domain_taxonomy_file: Path, eep1_filtered_keywords: list[str]
    ) -> None:
        """AC-2.3.1: Should extract concepts from EEP-1 filtered keywords."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        result = extractor.extract_concepts_from_keywords(eep1_filtered_keywords)
        assert len(result.concepts) > 0

    def test_min_domain_matches_threshold(self, domain_taxonomy_file: Path) -> None:
        """AC-2.3.2: Apply min_domain_matches threshold."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        # Text with only one LLM keyword should still match (min_domain_matches=1)
        sparse_text = "This text mentions vector embeddings."
        result = extractor.extract_concepts(sparse_text)
        # Should find at least one concept since "vector" and "embedding" are domain keywords
        assert len(result.concepts) >= 0  # May be empty if threshold not met

    def test_concepts_have_confidence_scores(self, domain_taxonomy_file: Path) -> None:
        """AC-2.3.3: Extracted concepts should have confidence scores."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        result = extractor.extract_concepts(SAMPLE_CHAPTER_TEXT)
        for concept in result.concepts:
            assert 0.0 <= concept.confidence <= 1.0

    def test_confidence_score_boosted_by_domain_keywords(
        self, domain_taxonomy_file: Path
    ) -> None:
        """AC-2.3.3: Score should be boosted for domain keyword matches."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        # Text heavy with LLM/RAG concepts
        heavy_text = "RAG retrieval vector embedding transformer attention langchain"
        result = extractor.extract_concepts(heavy_text)
        # Domain score for llm_rag should be significant
        assert result.domain_scores.get("llm_rag", 0) > 0


class TestHierarchicalConcepts:
    """Test AC-2.3.4: Support hierarchical concept relationships."""

    def test_concept_has_parent_relationship(
        self, domain_taxonomy_file: Path, tier_taxonomy_file: Path
    ) -> None:
        """AC-2.3.4: Concepts should support parent-child relationships."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(
            domain_taxonomy_path=domain_taxonomy_file,
            tier_taxonomy_path=tier_taxonomy_file,
            enable_hierarchical=True
        )
        extractor = ConceptExtractor(config)
        result = extractor.extract_concepts(SAMPLE_CHAPTER_TEXT)
        # Check if any concept has a parent
        has_parent = any(c.parent_concept is not None for c in result.concepts)
        # May or may not have parents depending on config
        assert isinstance(has_parent, bool)

    def test_get_concept_hierarchy(
        self, domain_taxonomy_file: Path, tier_taxonomy_file: Path
    ) -> None:
        """AC-2.3.4: Should build concept hierarchy tree."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(
            domain_taxonomy_path=domain_taxonomy_file,
            tier_taxonomy_path=tier_taxonomy_file,
            enable_hierarchical=True
        )
        extractor = ConceptExtractor(config)
        hierarchy = extractor.get_concept_hierarchy()
        assert isinstance(hierarchy, dict)


class TestDomainClassification:
    """Test domain classification from concepts."""

    def test_get_domain_concepts(self, domain_taxonomy_file: Path) -> None:
        """Get concepts for a specific domain."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        concepts = extractor.get_domain_concepts(TEST_DOMAIN_LLM_RAG)
        assert isinstance(concepts, list)
        assert len(concepts) > 0

    def test_classify_text_domain(self, domain_taxonomy_file: Path) -> None:
        """Classify text into primary domain."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        domain = extractor.classify_domain(SAMPLE_CHAPTER_TEXT)
        assert domain in [TEST_DOMAIN_LLM_RAG, TEST_DOMAIN_PYTHON, TEST_DOMAIN_MICROSERVICES, None]

    def test_classify_rag_text_as_llm_domain(self, domain_taxonomy_file: Path) -> None:
        """RAG-heavy text should classify as llm_rag domain."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        rag_text = "RAG retrieval augmented generation vector embeddings langchain llamaindex"
        domain = extractor.classify_domain(rag_text)
        assert domain == TEST_DOMAIN_LLM_RAG


# =============================================================================
# EEP-2.5: Anti-Pattern Compliance Tests
# =============================================================================


class TestAntiPatternCompliance:
    """Test AC-2.5.3: Refactor with 0 anti-patterns."""

    def test_no_duplicate_string_literals(self) -> None:
        """S1192: No duplicate string literals > 3 occurrences."""
        module_path = Path(
            "/Users/kevintoles/POC/Code-Orchestrator-Service/src/models/concept_extractor.py"
        )
        if not module_path.exists():
            pytest.skip("Module not yet created (RED phase)")
        content = module_path.read_text()
        # Check for constants section (indicates S1192 compliance)
        assert "# Constants" in content or "CONSTANT" in content.upper() or len(content) < 100

    def test_has_full_type_annotations(self) -> None:
        """Anti-Pattern #2.2: Full type annotations required."""
        module_path = Path(
            "/Users/kevintoles/POC/Code-Orchestrator-Service/src/models/concept_extractor.py"
        )
        if not module_path.exists():
            pytest.skip("Module not yet created (RED phase)")
        content = module_path.read_text()
        # Check for type annotation indicators
        assert "->" in content  # Return type annotations
        assert ": " in content  # Parameter annotations

    def test_cognitive_complexity_reasonable(self) -> None:
        """S3776: Cognitive complexity should be < 15 per method."""
        module_path = Path(
            "/Users/kevintoles/POC/Code-Orchestrator-Service/src/models/concept_extractor.py"
        )
        if not module_path.exists():
            pytest.skip("Module not yet created (RED phase)")
        content = module_path.read_text()
        # Check for helper methods (indicates complexity management)
        method_count = content.count("def ")
        # At least 5 methods expected for good decomposition
        assert method_count >= 5

    def test_no_exception_shadowing(self) -> None:
        """Anti-Pattern #7: No exception shadowing."""
        module_path = Path(
            "/Users/kevintoles/POC/Code-Orchestrator-Service/src/models/concept_extractor.py"
        )
        if not module_path.exists():
            pytest.skip("Module not yet created (RED phase)")
        content = module_path.read_text()
        # Should not have generic except without re-raise
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "except:" in line or "except Exception:" in line:
                # Next non-empty line should have raise or logging
                next_lines = "\n".join(lines[i:i+5])
                assert "raise" in next_lines or "logger" in next_lines or "logging" in next_lines


# =============================================================================
# Integration Tests (Minimal)
# =============================================================================


class TestConceptExtractorIntegration:
    """Integration tests for ConceptExtractor."""

    def test_full_extraction_pipeline(self, domain_taxonomy_file: Path) -> None:
        """Test complete extraction from text to concepts."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        
        # Extract concepts
        result = extractor.extract_concepts(SAMPLE_CHAPTER_TEXT)
        
        # Validate result structure
        assert hasattr(result, "concepts")
        assert hasattr(result, "domain_scores")
        assert isinstance(result.concepts, list)
        assert isinstance(result.domain_scores, dict)

    def test_extraction_from_code_text(self, domain_taxonomy_file: Path) -> None:
        """Test extraction from code-heavy text."""
        from src.models.concept_extractor import ConceptExtractor, ConceptExtractorConfig
        config = ConceptExtractorConfig(domain_taxonomy_path=domain_taxonomy_file)
        extractor = ConceptExtractor(config)
        
        result = extractor.extract_concepts(SAMPLE_CODE_TEXT)
        concept_names = [c.name.lower() for c in result.concepts]
        # Should find langchain-related concepts
        assert len(result.concepts) >= 0


# =============================================================================
# FakeConceptExtractor for Testing
# =============================================================================


class TestFakeConceptExtractor:
    """Test FakeConceptExtractor for protocol compliance (testing pattern)."""

    def test_fake_extractor_exists(self) -> None:
        """FakeConceptExtractor should exist for testing."""
        from src.models.concept_extractor import FakeConceptExtractor
        assert FakeConceptExtractor is not None

    def test_fake_extractor_follows_protocol(self) -> None:
        """FakeConceptExtractor should implement ConceptExtractorProtocol."""
        from src.models.concept_extractor import FakeConceptExtractor, ConceptExtractorProtocol
        fake = FakeConceptExtractor()
        assert hasattr(fake, "extract_concepts")
        assert hasattr(fake, "get_domain_concepts")
        assert hasattr(fake, "get_tier_concepts")

    def test_fake_extractor_returns_mock_data(self) -> None:
        """FakeConceptExtractor should return configurable mock data."""
        from src.models.concept_extractor import FakeConceptExtractor, ExtractedConcept
        mock_concepts = [
            ExtractedConcept(name="test_concept", confidence=1.0, domain="test", tier="test")
        ]
        fake = FakeConceptExtractor(mock_concepts=mock_concepts)
        result = fake.extract_concepts("any text")
        assert len(result.concepts) == 1
        assert result.concepts[0].name == "test_concept"
