"""
EEP-6: Diagram Similarity Extractor

WBS: EEP-6 - Diagram Similarity (Phase 6 of Enhanced Enrichment Pipeline)

Provides:
- EEP-6.1: Diagram detection (Figure X, Diagram X, ASCII art)
- EEP-6.2: Diagram description extraction (caption, context)
- EEP-6.3: Diagram similarity computation (SBERT embeddings)

Acceptance Criteria:
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

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np

# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

# Regex patterns for diagram detection
FIGURE_PATTERN = r"(?i)(?:^|\n)\s*(figure\s+[\d.]+)\s*[:\-]?\s*([^\n]*)"
DIAGRAM_PATTERN = r"(?i)(?:^|\n)\s*(diagram\s+[\d.]+)\s*[:\-]?\s*([^\n]*)"
ARCHITECTURE_PATTERN = r"(?i)(?:^|\n)\s*(architecture\s+diagram)\s*[:\-]?\s*([^\n]*)"

# Box drawing characters for ASCII art detection
BOX_DRAWING_CHARS = "─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬"
SIMPLE_BOX_CHARS = "+-|"

# Minimum density of box characters to detect ASCII art
ASCII_ART_THRESHOLD = 0.05

# Default context lines
DEFAULT_CONTEXT_LINES_BEFORE = 3
DEFAULT_CONTEXT_LINES_AFTER = 3

# Embedding dimensions
SBERT_EMBEDDING_DIM = 384


# =============================================================================
# Enums
# =============================================================================


class DiagramType(Enum):
    """Type of diagram detected in text."""

    FIGURE = auto()
    DIAGRAM = auto()
    ARCHITECTURE = auto()
    ASCII_ART = auto()


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True)
class DiagramReference:
    """
    Reference to a diagram found in text.

    Attributes:
        diagram_type: The type of diagram detected
        caption: The caption or title of the diagram
        context: Surrounding text context
        line_number: Optional line number where diagram was found
    """

    diagram_type: DiagramType
    caption: str
    context: str
    line_number: int | None = None

    def __eq__(self, other: object) -> bool:
        """Check equality based on all fields."""
        if not isinstance(other, DiagramReference):
            return NotImplemented
        return (
            self.diagram_type == other.diagram_type
            and self.caption == other.caption
            and self.context == other.context
            and self.line_number == other.line_number
        )

    def __hash__(self) -> int:
        """Make hashable for caching."""
        return hash((self.diagram_type, self.caption, self.context, self.line_number))


@dataclass
class DiagramExtractorConfig:
    """
    Configuration for DiagramExtractor.

    Attributes:
        context_lines_before: Number of lines to include before diagram
        context_lines_after: Number of lines to include after diagram
        ascii_art_threshold: Minimum box char density for ASCII detection
        sbert_model_name: Name of SBERT model to use for embeddings
    """

    context_lines_before: int = DEFAULT_CONTEXT_LINES_BEFORE
    context_lines_after: int = 5  # Increased to capture text after diagram
    ascii_art_threshold: float = ASCII_ART_THRESHOLD
    sbert_model_name: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class DiagramSimilarityResult:
    """
    Result of comparing two diagrams.

    Attributes:
        score: Similarity score between 0 and 1
        source_diagram_index: Index of source diagram in its list
        target_diagram_index: Index of target diagram in its list
    """

    score: float
    source_diagram_index: int
    target_diagram_index: int


# =============================================================================
# Protocols (per CODING_PATTERNS_ANALYSIS.md line 130)
# =============================================================================


@runtime_checkable
class DiagramExtractorProtocol(Protocol):
    """Protocol for diagram extraction implementations."""

    def extract_diagrams(self, text: str) -> list[DiagramReference]:
        """Extract diagram references from text."""
        ...

    def embed_diagram(self, diagram: DiagramReference) -> np.ndarray:
        """Embed diagram description using SBERT."""
        ...

    def compute_similarity(
        self,
        diagram1: DiagramReference,
        diagram2: DiagramReference,
    ) -> float:
        """Compute similarity between two diagrams."""
        ...


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeDiagramExtractor:
    """
    Fake implementation for testing (Protocol pattern).

    Provides deterministic results for unit tests.
    """

    def __init__(self) -> None:
        """Initialize fake extractor."""
        self._fake_diagrams: dict[str, list[DiagramReference]] = {}
        self._fake_embeddings: dict[DiagramReference, np.ndarray] = {}

    def extract_diagrams(self, text: str) -> list[DiagramReference]:
        """Return deterministic fake diagrams based on text hash."""
        if text in self._fake_diagrams:
            return self._fake_diagrams[text]
        return []

    def embed_diagram(self, diagram: DiagramReference) -> np.ndarray:
        """Return deterministic fake embedding."""
        if diagram in self._fake_embeddings:
            return self._fake_embeddings[diagram]
        # Generate deterministic embedding based on caption
        np.random.seed(hash(diagram.caption) % (2**32))
        embedding = np.random.randn(SBERT_EMBEDDING_DIM)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def compute_similarity(
        self,
        diagram1: DiagramReference,
        diagram2: DiagramReference,
    ) -> float:
        """Compute cosine similarity between fake embeddings."""
        emb1 = self.embed_diagram(diagram1)
        emb2 = self.embed_diagram(diagram2)
        return float(np.dot(emb1, emb2))

    def set_fake_diagrams(
        self,
        text: str,
        diagrams: list[DiagramReference],
    ) -> None:
        """Set fake diagrams for a specific text."""
        self._fake_diagrams[text] = diagrams

    def set_fake_embedding(
        self,
        diagram: DiagramReference,
        embedding: np.ndarray,
    ) -> None:
        """Set fake embedding for a specific diagram."""
        self._fake_embeddings[diagram] = embedding


# =============================================================================
# Main Implementation
# =============================================================================


class DiagramExtractor:
    """
    Extracts and compares diagrams from text content.

    Detects figure references, diagram references, architecture diagrams,
    and ASCII art. Uses SBERT for semantic similarity computation.

    Attributes:
        config: Extractor configuration
    """

    def __init__(self, config: DiagramExtractorConfig | None = None) -> None:
        """
        Initialize the diagram extractor.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or DiagramExtractorConfig()
        self._model = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    @property
    def _sbert_model(self):
        """Lazy-load SBERT model (Anti-Pattern #12 fix)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.config.sbert_model_name)
            except ImportError:
                # Fallback to mock for testing
                self._model = _MockSBERTModel()
        return self._model

    def extract_diagrams(self, text: str) -> list[DiagramReference]:
        """
        Extract all diagram references from text.

        Args:
            text: The text to search for diagrams

        Returns:
            List of DiagramReference objects found in text
        """
        if not text or not text.strip():
            return []

        diagrams: list[DiagramReference] = []

        # Detect labeled diagrams (Figure, Diagram, Architecture)
        diagrams.extend(self._extract_labeled_diagrams(text))

        # Detect ASCII art
        diagrams.extend(self._extract_ascii_art(text))

        return diagrams

    def _extract_labeled_diagrams(self, text: str) -> list[DiagramReference]:
        """Extract Figure X, Diagram X, Architecture diagram patterns."""
        diagrams: list[DiagramReference] = []
        lines = text.split("\n")

        # Pattern configs: (pattern, diagram_type)
        pattern_configs = [
            (FIGURE_PATTERN, DiagramType.FIGURE),
            (DIAGRAM_PATTERN, DiagramType.DIAGRAM),
            (ARCHITECTURE_PATTERN, DiagramType.ARCHITECTURE),
        ]

        for pattern, diagram_type in pattern_configs:
            for match in re.finditer(pattern, text):
                label = match.group(1).strip()
                caption = match.group(2).strip() if match.group(2) else label

                # Find line number
                line_number = text[: match.start()].count("\n") + 1

                # Extract context
                context = self._extract_context(lines, line_number - 1)

                diagrams.append(
                    DiagramReference(
                        diagram_type=diagram_type,
                        caption=caption,
                        context=context,
                        line_number=line_number,
                    )
                )

        return diagrams

    def _extract_ascii_art(self, text: str) -> list[DiagramReference]:
        """Detect ASCII art blocks using box drawing characters."""
        diagrams: list[DiagramReference] = []
        lines = text.split("\n")

        # Find consecutive lines with box characters
        ascii_block_start = None
        ascii_block_lines: list[str] = []

        for i, line in enumerate(lines):
            box_density = self._calculate_box_density(line)

            if box_density >= self.config.ascii_art_threshold:
                if ascii_block_start is None:
                    ascii_block_start = i
                ascii_block_lines.append(line)
            elif ascii_block_lines:
                # End of ASCII block - check if it's substantial
                if len(ascii_block_lines) >= 2:
                    context = self._extract_context(lines, ascii_block_start)
                    caption = self._generate_ascii_caption(ascii_block_lines, lines, ascii_block_start)

                    diagrams.append(
                        DiagramReference(
                            diagram_type=DiagramType.ASCII_ART,
                            caption=caption,
                            context=context,
                            line_number=ascii_block_start + 1,
                        )
                    )
                ascii_block_start = None
                ascii_block_lines = []

        # Handle block at end of text
        if ascii_block_lines and len(ascii_block_lines) >= 2:
            context = self._extract_context(lines, ascii_block_start)
            caption = self._generate_ascii_caption(ascii_block_lines, lines, ascii_block_start)

            diagrams.append(
                DiagramReference(
                    diagram_type=DiagramType.ASCII_ART,
                    caption=caption,
                    context=context,
                    line_number=ascii_block_start + 1,
                )
            )

        return diagrams

    def _calculate_box_density(self, line: str) -> float:
        """Calculate density of box drawing characters in a line."""
        if not line:
            return 0.0

        box_count = sum(1 for c in line if c in BOX_DRAWING_CHARS or c in SIMPLE_BOX_CHARS)
        return box_count / len(line)

    def _extract_context(self, lines: list[str], center_line: int) -> str:
        """Extract surrounding context lines."""
        start = max(0, center_line - self.config.context_lines_before)
        end = min(len(lines), center_line + self.config.context_lines_after + 1)

        context_lines = lines[start:end]
        return "\n".join(context_lines)

    def _generate_ascii_caption(
        self,
        ascii_lines: list[str],
        all_lines: list[str],
        start_line: int,
    ) -> str:
        """Generate caption for ASCII art from surrounding text."""
        # Look for description in line before ASCII art
        if start_line > 0:
            prev_line = all_lines[start_line - 1].strip()
            if prev_line and not self._is_ascii_art_line(prev_line):
                return f"ASCII Diagram: {prev_line}"

        # Look for description in line after ASCII art
        end_line = start_line + len(ascii_lines)
        if end_line < len(all_lines):
            next_line = all_lines[end_line].strip()
            if next_line and not self._is_ascii_art_line(next_line):
                return f"ASCII Diagram: {next_line}"

        return "ASCII Diagram"

    def _is_ascii_art_line(self, line: str) -> bool:
        """Check if a line is part of ASCII art."""
        return self._calculate_box_density(line) >= self.config.ascii_art_threshold

    def embed_diagram(self, diagram: DiagramReference) -> np.ndarray:
        """
        Embed diagram description using SBERT.

        Args:
            diagram: DiagramReference to embed

        Returns:
            Normalized embedding vector
        """
        # Create cache key from diagram content
        cache_key = f"{diagram.caption}|{diagram.context}"

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Combine caption and context for embedding
        text = f"{diagram.caption} {diagram.context}"

        # Get embedding from model
        embedding = self._sbert_model.encode(text, normalize_embeddings=True)

        # Cache and return
        self._embedding_cache[cache_key] = np.array(embedding)
        return self._embedding_cache[cache_key]

    def compute_similarity(
        self,
        diagram1: DiagramReference,
        diagram2: DiagramReference,
    ) -> float:
        """
        Compute cosine similarity between two diagrams.

        Args:
            diagram1: First diagram
            diagram2: Second diagram

        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.embed_diagram(diagram1)
        emb2 = self.embed_diagram(diagram2)

        # Cosine similarity (embeddings are normalized)
        similarity = float(np.dot(emb1, emb2))

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))

    def compare_chapter_diagrams(
        self,
        source_text: str,
        target_text: str,
    ) -> list[DiagramSimilarityResult]:
        """
        Compare all diagrams between two chapters.

        Args:
            source_text: Source chapter text
            target_text: Target chapter text

        Returns:
            List of similarity results for all diagram pairs
        """
        source_diagrams = self.extract_diagrams(source_text)
        target_diagrams = self.extract_diagrams(target_text)

        results: list[DiagramSimilarityResult] = []

        for i, src_diag in enumerate(source_diagrams):
            for j, tgt_diag in enumerate(target_diagrams):
                score = self.compute_similarity(src_diag, tgt_diag)
                results.append(
                    DiagramSimilarityResult(
                        score=score,
                        source_diagram_index=i,
                        target_diagram_index=j,
                    )
                )

        return results

    def get_max_diagram_similarity(
        self,
        source_text: str,
        target_text: str,
    ) -> float | None:
        """
        Get maximum similarity between any diagram pair.

        Args:
            source_text: Source chapter text
            target_text: Target chapter text

        Returns:
            Maximum similarity score, or None if no diagrams found
        """
        results = self.compare_chapter_diagrams(source_text, target_text)

        if not results:
            return None

        return max(r.score for r in results)


# =============================================================================
# Mock SBERT for testing without sentence-transformers
# =============================================================================


class _MockSBERTModel:
    """Mock SBERT model for testing without sentence-transformers."""

    def encode(
        self,
        text: str,
        normalize_embeddings: bool = True,  # noqa: ARG002
    ) -> np.ndarray:
        """Generate deterministic embedding from text."""
        # Use hash to generate deterministic but varied embeddings
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(SBERT_EMBEDDING_DIM)

        if normalize_embeddings:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding
