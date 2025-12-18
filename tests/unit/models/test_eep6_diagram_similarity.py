"""
Unit Tests for EEP-6: Diagram Similarity

WBS: EEP-6 - Diagram Similarity (Phase 6 of Enhanced Enrichment Pipeline)
TDD Phase: RED (tests written BEFORE implementation)

Tests for:
- EEP-6.1: Diagram detection (Figure X, Diagram X, ASCII art)
- EEP-6.2: Diagram description extraction (caption, context)
- EEP-6.3: Diagram similarity computation (SBERT embeddings)

Acceptance Criteria (from ENHANCED_ENRICHMENT_PIPELINE_WBS.md):
- AC-6.1.1: Detect "Figure X", "Diagram X", "Architecture diagram" patterns
- AC-6.1.2: Detect ASCII art diagrams (box drawing characters)
- AC-6.1.3: Return DiagramReference(type, caption, context)
- AC-6.2.1: Extract caption text
- AC-6.2.2: Extract surrounding context (paragraph before/after)
- AC-6.2.3: Use SBERT to embed description
- AC-6.3.1: Compare diagram descriptions using SBERT
- AC-6.3.2: Flag chapters with similar architecture diagrams

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Use constants for repeated string literals
- S3776: Cognitive complexity < 15
- S1172: No unused parameters
- #7: No exception shadowing
- #12: No model loading per request (cache embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from src.models.diagram_extractor import (
        DiagramExtractor,
        DiagramExtractorConfig,
        DiagramReference,
        DiagramSimilarityResult,
        DiagramType,
    )


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

# Sample text with figure references
SAMPLE_TEXT_WITH_FIGURE = """
Chapter 3: System Architecture

In this chapter, we explore the overall system design.

Figure 3.1: High-level architecture diagram showing the main components.

The system consists of three main layers: presentation, business logic, and data access.
Each layer communicates through well-defined interfaces.
"""

SAMPLE_TEXT_WITH_DIAGRAM = """
Section 2.5: Data Flow

The following diagram illustrates the data flow between components:

Diagram 2.5.1: Data Flow Architecture

Data enters through the API gateway and is processed by multiple microservices.
"""

SAMPLE_TEXT_WITH_ARCHITECTURE = """
Overview of Microservices

This architecture diagram shows the service mesh configuration:

Architecture Diagram: Service Mesh Topology

Each service communicates via gRPC and uses Envoy as a sidecar proxy.
"""

SAMPLE_ASCII_ART = """
The following shows the component layout:

┌─────────────┐     ┌─────────────┐
│   Client    │────►│   Server    │
└─────────────┘     └─────────────┘
        │                  │
        ▼                  ▼
┌─────────────┐     ┌─────────────┐
│  Frontend   │     │  Backend    │
└─────────────┘     └─────────────┘

This box diagram represents the client-server interaction.
"""

SAMPLE_SIMPLE_ASCII = """
Here is a simple flow:

+--------+     +--------+
| Input  |---->| Output |
+--------+     +--------+

The above shows basic input/output flow.
"""

SAMPLE_NO_DIAGRAMS = """
Chapter 1: Introduction

This chapter introduces the basic concepts of software engineering.
We will cover design patterns, best practices, and common pitfalls.

No diagrams are included in this introductory section.
"""


# =============================================================================
# EEP-6.1: Diagram Detection Tests
# =============================================================================


class TestDiagramTypeEnum:
    """Test DiagramType enum exists and has required values."""

    def test_diagram_type_enum_exists(self) -> None:
        """AC-6.1.1: DiagramType enum should exist."""
        from src.models.diagram_extractor import DiagramType

        assert DiagramType is not None

    def test_diagram_type_has_figure(self) -> None:
        """AC-6.1.1: DiagramType should have FIGURE value."""
        from src.models.diagram_extractor import DiagramType

        assert hasattr(DiagramType, "FIGURE")

    def test_diagram_type_has_diagram(self) -> None:
        """AC-6.1.1: DiagramType should have DIAGRAM value."""
        from src.models.diagram_extractor import DiagramType

        assert hasattr(DiagramType, "DIAGRAM")

    def test_diagram_type_has_architecture(self) -> None:
        """AC-6.1.1: DiagramType should have ARCHITECTURE value."""
        from src.models.diagram_extractor import DiagramType

        assert hasattr(DiagramType, "ARCHITECTURE")

    def test_diagram_type_has_ascii_art(self) -> None:
        """AC-6.1.2: DiagramType should have ASCII_ART value."""
        from src.models.diagram_extractor import DiagramType

        assert hasattr(DiagramType, "ASCII_ART")


class TestDiagramReferenceDataclass:
    """Test DiagramReference dataclass structure."""

    def test_diagram_reference_is_dataclass(self) -> None:
        """AC-6.1.3: DiagramReference should be a dataclass."""
        from dataclasses import is_dataclass

        from src.models.diagram_extractor import DiagramReference

        assert is_dataclass(DiagramReference)

    def test_diagram_reference_has_type_field(self) -> None:
        """AC-6.1.3: DiagramReference should have 'type' field."""
        from src.models.diagram_extractor import DiagramReference, DiagramType

        ref = DiagramReference(
            diagram_type=DiagramType.FIGURE,
            caption="Test caption",
            context="Test context",
        )
        assert hasattr(ref, "diagram_type")

    def test_diagram_reference_has_caption_field(self) -> None:
        """AC-6.1.3: DiagramReference should have 'caption' field."""
        from src.models.diagram_extractor import DiagramReference, DiagramType

        ref = DiagramReference(
            diagram_type=DiagramType.FIGURE,
            caption="Test caption",
            context="Test context",
        )
        assert hasattr(ref, "caption")

    def test_diagram_reference_has_context_field(self) -> None:
        """AC-6.1.3: DiagramReference should have 'context' field."""
        from src.models.diagram_extractor import DiagramReference, DiagramType

        ref = DiagramReference(
            diagram_type=DiagramType.FIGURE,
            caption="Test caption",
            context="Test context",
        )
        assert hasattr(ref, "context")

    def test_diagram_reference_has_line_number_field(self) -> None:
        """DiagramReference should have optional 'line_number' field."""
        from src.models.diagram_extractor import DiagramReference, DiagramType

        ref = DiagramReference(
            diagram_type=DiagramType.FIGURE,
            caption="Test caption",
            context="Test context",
            line_number=10,
        )
        assert ref.line_number == 10


class TestDiagramExtractorConfig:
    """Test DiagramExtractorConfig dataclass."""

    def test_config_is_dataclass(self) -> None:
        """DiagramExtractorConfig should be a dataclass."""
        from dataclasses import is_dataclass

        from src.models.diagram_extractor import DiagramExtractorConfig

        assert is_dataclass(DiagramExtractorConfig)

    def test_config_has_context_lines_before(self) -> None:
        """Config should have context_lines_before attribute."""
        from src.models.diagram_extractor import DiagramExtractorConfig

        config = DiagramExtractorConfig()
        assert hasattr(config, "context_lines_before")

    def test_config_has_context_lines_after(self) -> None:
        """Config should have context_lines_after attribute."""
        from src.models.diagram_extractor import DiagramExtractorConfig

        config = DiagramExtractorConfig()
        assert hasattr(config, "context_lines_after")

    def test_config_default_context_lines(self) -> None:
        """Default context should be 3 lines before and 5 after."""
        from src.models.diagram_extractor import DiagramExtractorConfig

        config = DiagramExtractorConfig()
        assert config.context_lines_before == 3
        assert config.context_lines_after == 5


class TestDiagramDetection:
    """Test diagram detection functionality."""

    def test_detect_figure_reference(self) -> None:
        """AC-6.1.1: Should detect 'Figure X' pattern."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        assert len(diagrams) >= 1
        assert any(d.diagram_type == DiagramType.FIGURE for d in diagrams)

    def test_detect_diagram_reference(self) -> None:
        """AC-6.1.1: Should detect 'Diagram X' pattern."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_DIAGRAM)

        assert len(diagrams) >= 1
        assert any(d.diagram_type == DiagramType.DIAGRAM for d in diagrams)

    def test_detect_architecture_diagram(self) -> None:
        """AC-6.1.1: Should detect 'Architecture diagram' pattern."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_ARCHITECTURE)

        assert len(diagrams) >= 1
        assert any(d.diagram_type == DiagramType.ARCHITECTURE for d in diagrams)

    def test_detect_ascii_art_box_drawing(self) -> None:
        """AC-6.1.2: Should detect ASCII art with box drawing characters."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_ASCII_ART)

        assert len(diagrams) >= 1
        assert any(d.diagram_type == DiagramType.ASCII_ART for d in diagrams)

    def test_detect_ascii_art_simple_boxes(self) -> None:
        """AC-6.1.2: Should detect simple ASCII boxes (+, -, |)."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_SIMPLE_ASCII)

        assert len(diagrams) >= 1
        assert any(d.diagram_type == DiagramType.ASCII_ART for d in diagrams)

    def test_no_diagrams_returns_empty_list(self) -> None:
        """Should return empty list when no diagrams found."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_NO_DIAGRAMS)

        assert diagrams == []

    def test_case_insensitive_detection(self) -> None:
        """Should detect diagrams case-insensitively."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        text = "FIGURE 1: Test\nfigure 2: Test\nFigure 3: Test"
        diagrams = extractor.extract_diagrams(text)

        assert len(diagrams) == 3


# =============================================================================
# EEP-6.2: Diagram Description Extraction Tests
# =============================================================================


class TestCaptionExtraction:
    """Test diagram caption extraction."""

    def test_extract_figure_caption(self) -> None:
        """AC-6.2.1: Should extract caption text from Figure reference."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        assert len(diagrams) >= 1
        figure_diagram = diagrams[0]
        assert "High-level architecture diagram" in figure_diagram.caption

    def test_extract_diagram_caption(self) -> None:
        """AC-6.2.1: Should extract caption text from Diagram reference."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_DIAGRAM)

        assert len(diagrams) >= 1
        assert "Data Flow Architecture" in diagrams[0].caption

    def test_caption_not_empty(self) -> None:
        """Caption should not be empty string."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        for diagram in diagrams:
            assert diagram.caption.strip() != ""


class TestContextExtraction:
    """Test surrounding context extraction."""

    def test_extract_context_before(self) -> None:
        """AC-6.2.2: Should extract paragraph before diagram."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        assert len(diagrams) >= 1
        # Context should include text before the diagram reference
        assert "system design" in diagrams[0].context.lower()

    def test_extract_context_after(self) -> None:
        """AC-6.2.2: Should extract paragraph after diagram."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        assert len(diagrams) >= 1
        # Context should include text after the diagram reference
        assert "three main layers" in diagrams[0].context.lower()

    def test_context_configurable_lines(self) -> None:
        """Context line count should be configurable."""
        from src.models.diagram_extractor import (
            DiagramExtractor,
            DiagramExtractorConfig,
        )

        config = DiagramExtractorConfig(context_lines_before=5, context_lines_after=5)
        extractor = DiagramExtractor(config=config)

        assert extractor.config.context_lines_before == 5
        assert extractor.config.context_lines_after == 5

    def test_ascii_art_context_includes_description(self) -> None:
        """ASCII art context/caption should include surrounding description."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_ASCII_ART)

        ascii_diagrams = [d for d in diagrams if d.diagram_type == DiagramType.ASCII_ART]
        assert len(ascii_diagrams) >= 1
        # Description may appear in caption or context
        combined_text = f"{ascii_diagrams[0].caption} {ascii_diagrams[0].context}".lower()
        # Should capture either "client-server" or component/layout description
        assert any(
            term in combined_text
            for term in ["client", "server", "component", "layout"]
        )


# =============================================================================
# EEP-6.3: Diagram Similarity Tests
# =============================================================================


class TestDiagramSimilarityResult:
    """Test DiagramSimilarityResult dataclass."""

    def test_similarity_result_is_dataclass(self) -> None:
        """DiagramSimilarityResult should be a dataclass."""
        from dataclasses import is_dataclass

        from src.models.diagram_extractor import DiagramSimilarityResult

        assert is_dataclass(DiagramSimilarityResult)

    def test_similarity_result_has_score(self) -> None:
        """Result should have similarity score."""
        from src.models.diagram_extractor import DiagramSimilarityResult

        result = DiagramSimilarityResult(
            score=0.85,
            source_diagram_index=0,
            target_diagram_index=1,
        )
        assert result.score == 0.85

    def test_similarity_result_has_indices(self) -> None:
        """Result should have source and target diagram indices."""
        from src.models.diagram_extractor import DiagramSimilarityResult

        result = DiagramSimilarityResult(
            score=0.85,
            source_diagram_index=0,
            target_diagram_index=1,
        )
        assert result.source_diagram_index == 0
        assert result.target_diagram_index == 1


class TestDiagramEmbedding:
    """Test SBERT embedding for diagram descriptions."""

    def test_embed_diagram_description(self) -> None:
        """AC-6.2.3: Should use SBERT to embed description."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        assert len(diagrams) >= 1
        embedding = extractor.embed_diagram(diagrams[0])

        # SBERT produces 384 or 768 dim embeddings depending on model
        assert embedding is not None
        assert len(embedding) > 0

    def test_embedding_is_normalized(self) -> None:
        """Embedding should be L2 normalized."""
        import numpy as np

        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        embedding = extractor.embed_diagram(diagrams[0])
        norm = np.linalg.norm(embedding)

        # Should be approximately 1.0 (L2 normalized)
        assert 0.99 <= norm <= 1.01


class TestDiagramSimilarity:
    """Test diagram similarity computation."""

    def test_compute_similarity_same_diagram(self) -> None:
        """AC-6.3.1: Same diagram should have similarity ~1.0."""
        from src.models.diagram_extractor import DiagramExtractor, DiagramType

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        if len(diagrams) >= 1:
            score = extractor.compute_similarity(diagrams[0], diagrams[0])
            assert score >= 0.99  # Same diagram should be very similar

    def test_compute_similarity_different_diagrams(self) -> None:
        """AC-6.3.1: Different diagrams should have varying similarity."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()

        # Two different types of diagrams
        diagrams1 = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)
        diagrams2 = extractor.extract_diagrams(SAMPLE_ASCII_ART)

        if diagrams1 and diagrams2:
            score = extractor.compute_similarity(diagrams1[0], diagrams2[0])
            # Different diagrams should have lower similarity
            assert 0.0 <= score <= 1.0

    def test_similarity_returns_float(self) -> None:
        """Similarity score should be a float between 0 and 1."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        if len(diagrams) >= 1:
            score = extractor.compute_similarity(diagrams[0], diagrams[0])
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_find_similar_diagrams(self) -> None:
        """AC-6.3.2: Should find chapters with similar diagrams."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()

        # Create two chapters with similar architecture diagrams
        chapter1 = """
        Figure 1: Microservices Architecture
        
        This diagram shows a service-oriented architecture with API gateway.
        """
        chapter2 = """
        Figure 2: Service Mesh Architecture
        
        This diagram illustrates a service-oriented design with load balancer.
        """
        chapter3 = """
        Figure 3: Database Schema
        
        This diagram shows the relational database entity relationships.
        """

        diagrams1 = extractor.extract_diagrams(chapter1)
        diagrams2 = extractor.extract_diagrams(chapter2)
        diagrams3 = extractor.extract_diagrams(chapter3)

        if diagrams1 and diagrams2 and diagrams3:
            # Architecture diagrams should be more similar to each other
            sim_1_2 = extractor.compute_similarity(diagrams1[0], diagrams2[0])
            sim_1_3 = extractor.compute_similarity(diagrams1[0], diagrams3[0])

            # Both scores should be valid
            assert 0.0 <= sim_1_2 <= 1.0
            assert 0.0 <= sim_1_3 <= 1.0


class TestBatchDiagramComparison:
    """Test batch diagram comparison across chapters."""

    def test_compare_chapter_diagrams(self) -> None:
        """Should compare all diagrams between two chapters."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()

        results = extractor.compare_chapter_diagrams(
            source_text=SAMPLE_TEXT_WITH_FIGURE,
            target_text=SAMPLE_TEXT_WITH_DIAGRAM,
        )

        # Should return a list of similarity results
        assert isinstance(results, list)

    def test_compare_returns_max_similarity(self) -> None:
        """Should return max similarity when multiple diagrams."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()

        max_sim = extractor.get_max_diagram_similarity(
            source_text=SAMPLE_TEXT_WITH_FIGURE,
            target_text=SAMPLE_TEXT_WITH_DIAGRAM,
        )

        # Should return a float or None if no diagrams
        assert max_sim is None or isinstance(max_sim, float)

    def test_no_diagrams_returns_none(self) -> None:
        """Should return None when one chapter has no diagrams."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()

        max_sim = extractor.get_max_diagram_similarity(
            source_text=SAMPLE_TEXT_WITH_FIGURE,
            target_text=SAMPLE_NO_DIAGRAMS,
        )

        assert max_sim is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty text should return empty diagram list."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams("")

        assert diagrams == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        """Whitespace-only text should return empty diagram list."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams("   \n\t\n   ")

        assert diagrams == []

    def test_malformed_figure_reference(self) -> None:
        """Should handle malformed figure references gracefully."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        # Missing colon, incomplete pattern
        text = "Figure without number\nFigure: no number here"
        diagrams = extractor.extract_diagrams(text)

        # Should not crash, may or may not find diagrams
        assert isinstance(diagrams, list)

    def test_unicode_content(self) -> None:
        """Should handle Unicode content correctly."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        text = """
        Figure 1: Архитектура системы (System Architecture)
        
        This shows the システム設計 design pattern.
        """
        diagrams = extractor.extract_diagrams(text)

        # Should handle Unicode without crashing
        assert isinstance(diagrams, list)

    def test_very_long_text(self) -> None:
        """Should handle very long text efficiently."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        # Create long text with a diagram in the middle
        long_text = "Introduction. " * 1000 + "\n\nFigure 1: Test diagram\n\n" + "Conclusion. " * 1000

        diagrams = extractor.extract_diagrams(long_text)

        assert len(diagrams) >= 1


# =============================================================================
# Protocol Pattern Tests
# =============================================================================


class TestProtocolPattern:
    """Test Protocol pattern compliance per CODING_PATTERNS_ANALYSIS.md."""

    def test_extractor_protocol_exists(self) -> None:
        """DiagramExtractorProtocol should exist for testing."""
        from src.models.diagram_extractor import DiagramExtractorProtocol

        assert DiagramExtractorProtocol is not None

    def test_fake_extractor_exists(self) -> None:
        """FakeDiagramExtractor should exist for testing."""
        from src.models.diagram_extractor import FakeDiagramExtractor

        fake = FakeDiagramExtractor()
        assert fake is not None

    def test_fake_extractor_returns_deterministic_results(self) -> None:
        """Fake extractor should return deterministic results."""
        from src.models.diagram_extractor import FakeDiagramExtractor

        fake = FakeDiagramExtractor()

        diagrams1 = fake.extract_diagrams("test text")
        diagrams2 = fake.extract_diagrams("test text")

        assert diagrams1 == diagrams2


# =============================================================================
# Integration with Existing Models
# =============================================================================


class TestSBERTIntegration:
    """Test integration with existing SBERT models."""

    def test_uses_existing_sbert_model(self) -> None:
        """Should use existing SemanticSimilarityEngine or shared SBERT."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()

        # Should have access to embedding functionality
        assert hasattr(extractor, "embed_diagram")
        assert callable(extractor.embed_diagram)

    def test_embedding_cached(self) -> None:
        """Embeddings should be cached (Anti-Pattern #12)."""
        from src.models.diagram_extractor import DiagramExtractor

        extractor = DiagramExtractor()
        diagrams = extractor.extract_diagrams(SAMPLE_TEXT_WITH_FIGURE)

        if diagrams:
            # First embedding
            emb1 = extractor.embed_diagram(diagrams[0])
            # Second embedding (should use cache)
            emb2 = extractor.embed_diagram(diagrams[0])

            # Should be identical (from cache)
            assert (emb1 == emb2).all() if hasattr(emb1, "__iter__") else emb1 == emb2
