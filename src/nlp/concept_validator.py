"""Concept Validation module - Post-extraction filtering and SBERT seed validation.

Validates extracted concepts to ensure they are real programming/technical terms
rather than author names, copyright notices, or generic filler words.

Two-stage validation:
1. Pattern Filter: Regex-based removal of author/copyright patterns (fast, no model)
2. SBERT Seed Validation: Semantic similarity to known programming concepts

AC Reference:
- HCE Target Outcome: Produce high-quality concept lists that maximize cross-book
  relationship discovery while minimizing noise and redundancy.

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- #12: SBERT engine cached as singleton (reuses SemanticDeduplicator's engine)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp.semantic_dedup import SemanticDeduplicator


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

# Default validation thresholds
DEFAULT_SBERT_SIMILARITY_THRESHOLD: Final[float] = 0.35
DEFAULT_MIN_CONCEPT_LENGTH: Final[int] = 2

# Validation stage names
STAGE_PATTERN_FILTER: Final[str] = "pattern_filter"
STAGE_SBERT_VALIDATE: Final[str] = "sbert_validate"

# Rejection reason codes
REJECTION_AUTHOR_PATTERN: Final[str] = "author_pattern"
REJECTION_NOISE_TERM: Final[str] = "noise_term"
REJECTION_LOW_SIMILARITY: Final[str] = "low_similarity"


# =============================================================================
# Pattern Filter Constants
# =============================================================================

# Patterns for author/copyright detection
AUTHOR_PATTERNS: Final[tuple[str, ...]] = (
    r"Copyright",                              # Any mention of copyright
    r"©",                                      # Copyright symbol
    r"^[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+$",  # Full name with middle initial
    r"All\s+Rights\s+Reserved",                # Common copyright phrase
)

# Known author names from textbooks (add as needed)
KNOWN_AUTHORS: Final[frozenset[str]] = frozenset({
    "ousterhout",      # A Philosophy of Software Design
    "john",            # Common first name (John Ousterhout)
    "fowler",          # Refactoring, Patterns of Enterprise Application Architecture
    "martin",          # Clean Code, Clean Architecture
    "gamma",           # Design Patterns (Gang of Four)
    "helm",            # Design Patterns (Gang of Four)
    "johnson",         # Design Patterns (Gang of Four)
    "vlissides",       # Design Patterns (Gang of Four)
    "knuth",           # The Art of Computer Programming
    "stroustrup",      # The C++ Programming Language
    "kernighan",       # The C Programming Language
    "ritchie",         # The C Programming Language
    "bloch",           # Effective Java
    "meyers",          # Effective C++
    "sutter",          # Exceptional C++
    "alexandrescu",    # Modern C++ Design
    "kleppmann",       # Designing Data-Intensive Applications
    "nygard",          # Release It!
    "newman",          # Building Microservices
})

# Generic noise terms without technical meaning
NOISE_TERMS: Final[frozenset[str]] = frozenset({
    # Common filler words
    "taking", "little", "people", "things", "way", "time", "part", "bring",
    "kind", "sure", "long", "high", "low", "good", "bad", "big", "small",
    "names", "john", "much", "many", "well", "even", "just", "also",
    # Generic verbs
    "using", "used", "use", "make", "made", "get", "got", "set", "put",
    "come", "came", "go", "went", "take", "took", "give", "gave",
})


# =============================================================================
# SBERT Seed Concepts
# =============================================================================

# Seed concepts representing valid programming/software terms
# These are used for semantic similarity validation
SEED_PROGRAMMING_CONCEPTS: Final[tuple[str, ...]] = (
    # Core CS concepts
    "algorithm", "data structure", "design pattern", "API", "interface",
    "complexity", "abstraction", "module", "dependency", "coupling",
    "cohesion", "encapsulation", "inheritance", "polymorphism",
    # Software design
    "code", "software", "design", "programming", "development",
    "documentation", "refactoring", "debugging", "testing",
    "architecture", "scalability", "performance", "optimization",
    # APOSD-specific concepts
    "information hiding", "deep module", "shallow module", "comments",
    "exception handling", "error handling", "layer", "pass-through",
    # Development practices
    "version control", "deployment", "monitoring", "logging",
    "continuous integration", "code review", "unit testing",
    # System concepts
    "concurrency", "threading", "caching", "database", "network",
    "protocol", "memory", "storage", "security", "authentication",
)


# =============================================================================
# Configuration and Result Dataclasses
# =============================================================================


@dataclass
class ConceptValidationConfig:
    """Configuration for concept validation.

    Attributes:
        enable_pattern_filter: Enable regex-based author/noise filtering.
        enable_sbert_validation: Enable SBERT semantic similarity validation.
        sbert_similarity_threshold: Minimum similarity to seed concepts (0-1).
        min_concept_length: Minimum characters for valid concept.
        additional_noise_terms: Extra terms to filter (merged with defaults).
        additional_authors: Extra author names to filter (merged with defaults).
    """

    enable_pattern_filter: bool = True
    enable_sbert_validation: bool = True
    sbert_similarity_threshold: float = DEFAULT_SBERT_SIMILARITY_THRESHOLD
    min_concept_length: int = DEFAULT_MIN_CONCEPT_LENGTH
    additional_noise_terms: frozenset[str] = field(default_factory=frozenset)
    additional_authors: frozenset[str] = field(default_factory=frozenset)


@dataclass
class ConceptValidationResult:
    """Result of concept validation.

    Attributes:
        valid_concepts: Concepts that passed all validation stages.
        rejected_concepts: Concepts that were rejected.
        rejection_reasons: Map of rejected concept -> reason code.
        similarity_scores: Map of valid concept -> SBERT similarity score.
        stages_executed: List of validation stages that ran.
    """

    valid_concepts: list[str] = field(default_factory=list)
    rejected_concepts: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)
    similarity_scores: dict[str, float] = field(default_factory=dict)
    stages_executed: list[str] = field(default_factory=list)


# =============================================================================
# ConceptValidator Class
# =============================================================================


class ConceptValidator:
    """Validates extracted concepts using pattern matching and SBERT similarity.

    Two-stage validation pipeline:
    1. Pattern Filter (fast): Remove author names, copyright notices, noise terms
    2. SBERT Validation (semantic): Keep only concepts similar to programming seeds

    Anti-Pattern Compliance:
    - #12: Reuses SemanticDeduplicator's cached SBERT engine
    - S1192: All constants at module level
    """

    def __init__(
        self,
        config: ConceptValidationConfig | None = None,
        deduplicator: SemanticDeduplicator | None = None,
    ) -> None:
        """Initialize the concept validator.

        Args:
            config: Validation configuration. Defaults to ConceptValidationConfig().
            deduplicator: SemanticDeduplicator instance for SBERT access.
                          If None, creates new instance (shares SBERT singleton).
        """
        self.config = config or ConceptValidationConfig()
        self._deduplicator = deduplicator or SemanticDeduplicator()

        # Compile regex patterns
        self._author_patterns = [
            re.compile(p, re.IGNORECASE) for p in AUTHOR_PATTERNS
        ]

        # Merge noise terms and authors with config additions
        self._noise_terms = NOISE_TERMS | self.config.additional_noise_terms
        self._known_authors = KNOWN_AUTHORS | self.config.additional_authors

        # Cache for seed embeddings
        self._seed_embeddings: NDArray[np.floating[Any]] | None = None

    def _get_seed_embeddings(self) -> NDArray[np.floating[Any]]:
        """Get or compute seed concept embeddings (cached).

        Returns:
            Numpy array of shape (num_seeds, embedding_dim).
        """
        if self._seed_embeddings is None:
            self._seed_embeddings = self._deduplicator.compute_embeddings(
                list(SEED_PROGRAMMING_CONCEPTS)
            )
        return self._seed_embeddings

    def _is_author_pattern(self, concept: str) -> bool:
        """Check if concept matches author/copyright patterns.

        Args:
            concept: Concept string to check.

        Returns:
            True if concept matches any author pattern.
        """
        # Check regex patterns
        for pattern in self._author_patterns:
            if pattern.search(concept):
                return True

        # Check known authors (any word in concept)
        concept_words = set(concept.lower().split())
        if concept_words & self._known_authors:
            return True

        return False

    def _is_noise_term(self, concept: str) -> bool:
        """Check if concept is a generic noise term.

        Args:
            concept: Concept string to check.

        Returns:
            True if concept is a noise term.
        """
        return concept.lower() in self._noise_terms

    def _pattern_filter(
        self,
        concepts: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        """Filter concepts using pattern matching.

        Stage 1: Fast pattern-based filtering (no model inference).

        Args:
            concepts: List of concepts to filter.

        Returns:
            Tuple of (valid_concepts, rejection_reasons).
        """
        valid = []
        rejections: dict[str, str] = {}

        for concept in concepts:
            # Check minimum length
            if len(concept) < self.config.min_concept_length:
                rejections[concept] = REJECTION_NOISE_TERM
                continue

            # Check author patterns
            if self._is_author_pattern(concept):
                rejections[concept] = REJECTION_AUTHOR_PATTERN
                continue

            # Check noise terms
            if self._is_noise_term(concept):
                rejections[concept] = REJECTION_NOISE_TERM
                continue

            valid.append(concept)

        return valid, rejections

    def _sbert_validate(
        self,
        concepts: list[str],
    ) -> tuple[list[str], dict[str, str], dict[str, float]]:
        """Validate concepts using SBERT semantic similarity.

        Stage 2: Semantic validation against seed programming concepts.

        Args:
            concepts: List of concepts to validate.

        Returns:
            Tuple of (valid_concepts, rejection_reasons, similarity_scores).
        """
        if not concepts:
            return [], {}, {}

        # Get embeddings
        seed_embeddings = self._get_seed_embeddings()
        concept_embeddings = self._deduplicator.compute_embeddings(concepts)

        # Compute similarity matrix
        sim_matrix = cosine_similarity(concept_embeddings, seed_embeddings)
        max_similarities = sim_matrix.max(axis=1)

        # Filter by threshold
        valid = []
        rejections: dict[str, str] = {}
        scores: dict[str, float] = {}

        for concept, similarity in zip(concepts, max_similarities):
            sim_float = float(similarity)
            if sim_float >= self.config.sbert_similarity_threshold:
                valid.append(concept)
                scores[concept] = sim_float
            else:
                rejections[concept] = REJECTION_LOW_SIMILARITY

        return valid, rejections, scores

    def validate(self, concepts: list[str]) -> ConceptValidationResult:
        """Validate concepts through all configured stages.

        Pipeline:
        1. Pattern Filter (if enabled): Remove author/noise patterns
        2. SBERT Validation (if enabled): Keep semantically similar concepts

        Args:
            concepts: List of concept strings to validate.

        Returns:
            ConceptValidationResult with valid/rejected concepts and metadata.
        """
        result = ConceptValidationResult()
        current_concepts = concepts.copy()

        # Stage 1: Pattern Filter
        if self.config.enable_pattern_filter:
            current_concepts, pattern_rejections = self._pattern_filter(
                current_concepts
            )
            result.rejection_reasons.update(pattern_rejections)
            result.stages_executed.append(STAGE_PATTERN_FILTER)

        # Stage 2: SBERT Validation
        if self.config.enable_sbert_validation and current_concepts:
            current_concepts, sbert_rejections, scores = self._sbert_validate(
                current_concepts
            )
            result.rejection_reasons.update(sbert_rejections)
            result.similarity_scores = scores
            result.stages_executed.append(STAGE_SBERT_VALIDATE)

        # Finalize results
        result.valid_concepts = current_concepts
        result.rejected_concepts = list(result.rejection_reasons.keys())

        return result
