"""End-to-end tests for ConceptExtractionPipeline - HCE-5.16 to HCE-5.18.

Tests full pipeline execution with real chapter text per AC-5.5.

TDD Phase: RED - Tests should FAIL initially.
"""

import time

import pytest

from src.extractors.concept_extraction_pipeline import (
    ConceptExtractionConfig,
    ConceptExtractionPipeline,
    ConceptExtractionResult,
)


# =============================================================================
# Test Data - Real Chapter Excerpts
# =============================================================================

# APOSD-style chapter excerpt (software design concepts)
APOSD_CHAPTER_TEXT = """
The most fundamental problem in computer science is problem decomposition:
how to take a complex problem and divide it up into pieces that can be 
solved independently. Problem decomposition is the central design task that
programmers face every day, and yet, other than the work described here,
I have not been able to find a single class in any university where problem
decomposition is a central topic. We teach for loops and object-oriented 
programming, but not software design.

The most important technique for achieving deep modules is information hiding.
This technique was first described by David Parnas. The basic idea is that
each module should encapsulate a few pieces of knowledge, which represent
design decisions. The knowledge is embedded in the module's implementation
but does not appear in its interface, so it is not visible to other modules.

Complexity is caused by two things: dependencies and obscurity. A dependency
exists when a given piece of code can not be understood and modified in 
isolation; the code relates in some way to other code, and the other code
must be considered and/or modified if the given code is changed.
"""

# Code-heavy chapter excerpt (systems programming)
CODE_HEAVY_CHAPTER_TEXT = """
The Linux kernel uses a red-black tree implementation for the CPU scheduler's
completely fair scheduler (CFS). The struct rb_node and struct rb_root are
defined in include/linux/rbtree.h:

    struct rb_node {
        unsigned long __rb_parent_color;
        struct rb_node *rb_right;
        struct rb_node *rb_left;
    };

The O(log n) insertion and deletion operations make red-black trees ideal
for managing process run queues. The kernel also uses hash tables extensively
for the page cache and dentry cache lookups. The hash_long() function in
include/linux/hash.h implements a multiplicative hash using the golden ratio:

    #define GOLDEN_RATIO_PRIME 0x9e37fffffffc0001UL
    
Memory allocation in the kernel uses the slab allocator, which maintains
caches of commonly-sized objects. The kmem_cache_alloc() function allocates
objects from these caches, avoiding the overhead of general-purpose allocation.
"""

# Short text for performance testing
SHORT_TEXT = """
Machine learning enables computers to learn from data. Deep neural networks
are a type of machine learning model with multiple layers. Gradient descent
optimizes model parameters during training.
"""


# =============================================================================
# HCE-5.16: E2E Test with APOSD Chapter (AC-5.5)
# =============================================================================


class TestAPOSDExtraction:
    """HCE-5.16: Test E2E with APOSD-style chapter."""

    def test_extracts_software_design_concepts(self) -> None:
        """HCE-5.16: Should extract software design concepts from APOSD text."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        
        # Should return ConceptExtractionResult
        assert isinstance(result, ConceptExtractionResult)
        
        # Should have extracted concepts
        assert len(result.concepts) > 0
        
        # Extract concept terms for checking
        concept_terms = [c.term.lower() for c in result.concepts]
        
        # Should find key software design concepts
        design_terms = ["decomposition", "module", "complexity", "interface", 
                       "design", "information hiding", "dependency", "abstraction"]
        found_design_terms = [t for t in design_terms if any(t in ct for ct in concept_terms)]
        
        # Should find at least some design concepts
        assert len(found_design_terms) > 0, f"Expected design concepts, got: {concept_terms[:10]}"

    def test_aposd_returns_non_empty_concepts(self) -> None:
        """HCE-5.16: APOSD chapter should return non-empty concepts."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        
        assert len(result.concepts) > 0
        assert all(c.term for c in result.concepts)

    def test_aposd_concepts_have_scores(self) -> None:
        """HCE-5.16: Extracted concepts should have valid scores."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        
        for concept in result.concepts:
            assert isinstance(concept.score, float)


# =============================================================================
# HCE-5.17: E2E Test with Code-Heavy Chapter (AC-5.5)
# =============================================================================


class TestCodeHeavyExtraction:
    """HCE-5.17: Test E2E with code-heavy chapter."""

    def test_extracts_systems_concepts(self) -> None:
        """HCE-5.17: Should extract systems concepts from code-heavy text."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(CODE_HEAVY_CHAPTER_TEXT)
        
        assert isinstance(result, ConceptExtractionResult)
        assert len(result.concepts) > 0
        
        concept_terms = [c.term.lower() for c in result.concepts]
        
        # Should find systems programming concepts
        systems_terms = ["kernel", "scheduler", "memory", "hash", "cache", 
                        "allocation", "tree", "red-black"]
        found_systems_terms = [t for t in systems_terms if any(t in ct for ct in concept_terms)]
        
        assert len(found_systems_terms) > 0, f"Expected systems concepts, got: {concept_terms[:10]}"

    def test_handles_code_snippets_gracefully(self) -> None:
        """HCE-5.17: Pipeline should handle code snippets without errors."""
        pipeline = ConceptExtractionPipeline()
        
        # Should not raise exception
        result = pipeline.extract(CODE_HEAVY_CHAPTER_TEXT)
        
        assert result is not None
        # Should complete pipeline stages
        assert len(result.pipeline_metadata.get("stages_executed", [])) > 0

    def test_code_heavy_returns_non_empty_concepts(self) -> None:
        """HCE-5.17: Code-heavy chapter should return non-empty concepts."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(CODE_HEAVY_CHAPTER_TEXT)
        
        assert len(result.concepts) > 0


# =============================================================================
# HCE-5.18: Performance Test (AC-5.5)
# =============================================================================


class TestPipelinePerformance:
    """HCE-5.18: Test E2E completes in < 5s (AC-5.5)."""

    def test_short_text_under_5_seconds(self) -> None:
        """HCE-5.18: Short text extraction should complete in < 5s."""
        pipeline = ConceptExtractionPipeline()
        
        start = time.perf_counter()
        result = pipeline.extract(SHORT_TEXT)
        duration = time.perf_counter() - start
        
        assert duration < 5.0, f"Extraction took {duration:.2f}s, expected < 5s"
        assert result is not None

    def test_aposd_chapter_under_5_seconds(self) -> None:
        """HCE-5.18: APOSD chapter extraction should complete in < 5s."""
        pipeline = ConceptExtractionPipeline()
        
        start = time.perf_counter()
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        duration = time.perf_counter() - start
        
        assert duration < 5.0, f"Extraction took {duration:.2f}s, expected < 5s"

    def test_code_heavy_chapter_under_5_seconds(self) -> None:
        """HCE-5.18: Code-heavy chapter extraction should complete in < 5s."""
        pipeline = ConceptExtractionPipeline()
        
        start = time.perf_counter()
        result = pipeline.extract(CODE_HEAVY_CHAPTER_TEXT)
        duration = time.perf_counter() - start
        
        assert duration < 5.0, f"Extraction took {duration:.2f}s, expected < 5s"

    def test_pipeline_reports_duration_ms(self) -> None:
        """HCE-5.18: Pipeline should report duration_ms in metadata."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(SHORT_TEXT)
        
        metadata = result.pipeline_metadata
        assert "duration_ms" in metadata
        assert isinstance(metadata["duration_ms"], (int, float))
        assert metadata["duration_ms"] > 0
        assert metadata["duration_ms"] < 5000  # < 5s = 5000ms


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestPipelineStagesExecuted:
    """Test that all pipeline stages execute correctly."""

    def test_all_default_stages_executed(self) -> None:
        """All default-enabled stages should be in stages_executed."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        
        stages = result.pipeline_metadata.get("stages_executed", [])
        
        # Default stages should all be present
        assert "noise_filter" in stages
        assert "yake" in stages
        assert "textrank" in stages
        assert "stem_dedup" in stages
        assert "semantic_dedup" in stages

    def test_stages_in_correct_order(self) -> None:
        """Stages should execute in correct order."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(SHORT_TEXT)
        
        stages = result.pipeline_metadata.get("stages_executed", [])
        
        # Check relative ordering
        if "noise_filter" in stages and "yake" in stages:
            assert stages.index("noise_filter") < stages.index("yake")
        
        if "stem_dedup" in stages and "semantic_dedup" in stages:
            assert stages.index("stem_dedup") < stages.index("semantic_dedup")


class TestDedupStatsIntegration:
    """Test dedup stats are correctly reported."""

    def test_dedup_stats_present(self) -> None:
        """dedup_stats should be present in result."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        
        assert hasattr(result, "dedup_stats")
        assert isinstance(result.dedup_stats, dict)

    def test_stem_removed_reported(self) -> None:
        """stem_removed count should be reported."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        
        assert "stem_removed" in result.dedup_stats
        assert isinstance(result.dedup_stats["stem_removed"], int)

    def test_semantic_removed_reported(self) -> None:
        """semantic_removed count should be reported."""
        pipeline = ConceptExtractionPipeline()
        
        result = pipeline.extract(APOSD_CHAPTER_TEXT)
        
        assert "semantic_removed" in result.dedup_stats
        assert isinstance(result.dedup_stats["semantic_removed"], int)
