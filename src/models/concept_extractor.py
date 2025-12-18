"""
Code-Orchestrator-Service - Concept Extractor

EEP-2: Concept Extraction Layer (Phase 2 of Enhanced Enrichment Pipeline)

Extracts domain concepts from text and EEP-1 filtered keywords by matching
against taxonomy keywords. Supports hierarchical concept relationships and
domain classification.

WBS Acceptance Criteria:
- AC-2.1.1: src/models/concept_extractor.py created ✓
- AC-2.1.2: Protocol pattern per CODING_PATTERNS_ANALYSIS.md line 130
- AC-2.1.3: Dataclasses for output structures (ExtractedConcept)
- AC-2.1.4: Full type annotations (Anti-Pattern #2.2)
- AC-2.2.1: Load taxonomy from configurable path
- AC-2.2.2: Parse tier structure (T0-T5 hierarchy)
- AC-2.2.3: Cache taxonomy in memory (Anti-Pattern #12)
- AC-2.3.1: Match text against domain_keywords and primary_keywords
- AC-2.3.2: Apply min_domain_matches threshold
- AC-2.3.3: Return matched concepts with confidence scores
- AC-2.3.4: Support hierarchical concept relationships

Document Priority Applied:
1. GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md
2. AI_CODING_PLATFORM_ARCHITECTURE.md
3. domain_taxonomy.json structure

Patterns Applied:
- Protocol typing for duck typing (CODING_PATTERNS_ANALYSIS.md line 130)
- Dataclasses for structured output
- FakeConceptExtractor for testing

Anti-Patterns Avoided:
- S1192: Constants for repeated string literals
- S3776: Cognitive complexity < 15 (helper methods)
- S1172: No unused parameters
- #7: No exception shadowing (explicit re-raise)
- #12: Taxonomy cached at __init__ (not per request)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# =============================================================================
# Constants (S1192 compliance - no duplicate string literals)
# =============================================================================

# Taxonomy structure keys
KEY_DOMAINS = "domains"
KEY_TIERS = "tiers"
KEY_PRIMARY_KEYWORDS = "primary_keywords"
KEY_DOMAIN_KEYWORDS = "domain_keywords"
KEY_MIN_DOMAIN_MATCHES = "min_domain_matches"
KEY_SCORE_ADJUSTMENTS = "score_adjustments"
KEY_TIER_WHITELIST = "tier_whitelist"
KEY_CONCEPTS = "concepts"
KEY_PRIORITY = "priority"
KEY_DEFAULT_SETTINGS = "default_settings"

# Score adjustment keys
SCORE_DOMAIN_KEYWORD_PRESENT = "domain_keyword_present"
SCORE_PRIMARY_ONLY_NO_DOMAIN = "primary_only_no_domain"
SCORE_IN_WHITELIST_TIER = "in_whitelist_tier"

# Default values
DEFAULT_MIN_DOMAIN_MATCHES = 1
DEFAULT_CONFIDENCE_BASE = 0.5
DEFAULT_CONFIDENCE_BOOST = 0.1
DEFAULT_UNKNOWN_DOMAIN = "unknown"

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes (AC-2.1.3)
# =============================================================================


@dataclass(frozen=True)
class ExtractedConcept:
    """A concept extracted from text with confidence and domain information.

    Attributes:
        name: The concept name/term
        confidence: Confidence score 0.0-1.0
        domain: Domain classification (e.g., 'llm_rag', 'python_implementation')
        tier: Tier classification (e.g., 'architecture', 'practices')
        parent_concept: Optional parent concept for hierarchy (AC-2.3.4)
    """

    name: str
    confidence: float
    domain: str
    tier: str
    parent_concept: str | None = None


@dataclass
class ConceptExtractionResult:
    """Result of concept extraction containing matched concepts and domain scores.

    Attributes:
        concepts: List of extracted concepts
        domain_scores: Domain-level confidence scores
        primary_domain: Primary domain classification
        total_matches: Total number of keyword matches
    """

    concepts: list[ExtractedConcept] = field(default_factory=list)
    domain_scores: dict[str, float] = field(default_factory=dict)
    primary_domain: str | None = None
    total_matches: int = 0


@dataclass
class ConceptExtractorConfig:
    """Configuration for ConceptExtractor.

    Attributes:
        domain_taxonomy_path: Path to domain_taxonomy.json
        tier_taxonomy_path: Optional path to tier taxonomy
        enable_hierarchical: Whether to build concept hierarchies
        min_confidence: Minimum confidence threshold for concepts
    """

    domain_taxonomy_path: Path | None = None
    tier_taxonomy_path: Path | None = None
    enable_hierarchical: bool = False
    min_confidence: float = 0.3


# =============================================================================
# Protocol (AC-2.1.2 - per CODING_PATTERNS_ANALYSIS.md line 130)
# =============================================================================


class ConceptExtractorProtocol(Protocol):
    """Protocol for ConceptExtractor duck typing.

    Enables FakeConceptExtractor for testing without real taxonomy files.
    Pattern: Protocol typing per CODING_PATTERNS_ANALYSIS.md line 130
    """

    def extract_concepts(self, text: str) -> ConceptExtractionResult:
        """Extract concepts from text."""
        ...

    def extract_concepts_from_keywords(
        self, keywords: list[str]
    ) -> ConceptExtractionResult:
        """Extract concepts from EEP-1 filtered keywords."""
        ...

    def get_domain_concepts(self, domain: str) -> list[str]:
        """Get all concepts for a domain."""
        ...

    def get_tier_concepts(self, tier: str) -> list[str]:
        """Get all concepts for a tier."""
        ...

    def classify_domain(self, text: str) -> str | None:
        """Classify text into primary domain."""
        ...


# =============================================================================
# ConceptExtractor Implementation
# =============================================================================


class ConceptExtractor:
    """Extracts domain concepts from text by matching against taxonomy keywords.

    Implements ConceptExtractorProtocol for duck typing support.
    Caches taxonomy at initialization (Anti-Pattern #12 prevention).

    Example:
        config = ConceptExtractorConfig(domain_taxonomy_path=Path("taxonomy.json"))
        extractor = ConceptExtractor(config)
        result = extractor.extract_concepts("RAG architecture with embeddings")
    """

    def __init__(self, config: ConceptExtractorConfig) -> None:
        """Initialize ConceptExtractor with configuration.

        Args:
            config: ConceptExtractorConfig with taxonomy paths

        Raises:
            FileNotFoundError: If taxonomy file doesn't exist
        """
        self._config = config
        self._taxonomy: dict[str, Any] | None = None
        self._tier_taxonomy: dict[str, Any] | None = None
        self._domains: dict[str, dict[str, Any]] = {}
        self._tiers: dict[str, dict[str, Any]] = {}

        # Load taxonomies at init (AC-2.2.3 - cache, Anti-Pattern #12)
        self._load_taxonomies()

    def _load_taxonomies(self) -> None:
        """Load and cache taxonomy files (Anti-Pattern #12 prevention)."""
        if self._config.domain_taxonomy_path:
            self._taxonomy = self._load_json_file(self._config.domain_taxonomy_path)
            self._domains = self._taxonomy.get(KEY_DOMAINS, {})

        if self._config.tier_taxonomy_path:
            self._tier_taxonomy = self._load_json_file(self._config.tier_taxonomy_path)
            self._tiers = self._tier_taxonomy.get(KEY_TIERS, {})

    def _load_json_file(self, path: Path) -> dict[str, Any]:
        """Load JSON file with proper error handling.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON as dict

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {path}")

        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in taxonomy file %s: %s", path, e)
            raise  # Re-raise, no shadowing (Anti-Pattern #7)

    @property
    def taxonomy(self) -> dict[str, Any] | None:
        """Get cached taxonomy (same object on repeated access)."""
        return self._taxonomy

    @property
    def tier_taxonomy(self) -> dict[str, Any] | None:
        """Get cached tier taxonomy."""
        return self._tier_taxonomy

    @property
    def domains(self) -> dict[str, dict[str, Any]]:
        """Get domains dict from taxonomy."""
        return self._domains

    @property
    def tiers(self) -> dict[str, dict[str, Any]]:
        """Get tiers dict from tier taxonomy."""
        return self._tiers

    def extract_concepts(self, text: str) -> ConceptExtractionResult:
        """Extract concepts from text by matching taxonomy keywords.

        AC-2.3.1: Match text against domain_keywords and primary_keywords
        AC-2.3.2: Apply min_domain_matches threshold
        AC-2.3.3: Return matched concepts with confidence scores

        Args:
            text: Text to extract concepts from

        Returns:
            ConceptExtractionResult with matched concepts and scores
        """
        # Normalize text for matching
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        concepts: list[ExtractedConcept] = []
        domain_scores: dict[str, float] = {}
        total_matches = 0

        for domain_name, domain_config in self._domains.items():
            domain_result = self._match_domain_concepts(
                words, text_lower, domain_name, domain_config
            )
            if domain_result.concepts:
                concepts.extend(domain_result.concepts)
                domain_scores[domain_name] = domain_result.domain_scores.get(domain_name, 0)
                total_matches += domain_result.total_matches

        # Determine primary domain
        primary_domain = self._get_primary_domain(domain_scores)

        # Filter by confidence threshold
        concepts = [c for c in concepts if c.confidence >= self._config.min_confidence]

        return ConceptExtractionResult(
            concepts=concepts,
            domain_scores=domain_scores,
            primary_domain=primary_domain,
            total_matches=total_matches,
        )

    def _match_domain_concepts(
        self,
        words: set[str],
        text_lower: str,
        domain_name: str,
        domain_config: dict[str, Any],
    ) -> ConceptExtractionResult:
        """Match concepts for a single domain (S3776 - reduced complexity).

        Args:
            words: Set of words from text
            text_lower: Lowercase text for phrase matching
            domain_name: Name of domain being matched
            domain_config: Domain configuration from taxonomy

        Returns:
            ConceptExtractionResult for this domain
        """
        primary_keywords = {
            k.lower() for k in domain_config.get(KEY_PRIMARY_KEYWORDS, [])
        }
        domain_keywords = {
            k.lower() for k in domain_config.get(KEY_DOMAIN_KEYWORDS, [])
        }
        min_matches = domain_config.get(KEY_MIN_DOMAIN_MATCHES, DEFAULT_MIN_DOMAIN_MATCHES)
        score_adjustments = domain_config.get(KEY_SCORE_ADJUSTMENTS, {})
        tier_whitelist = domain_config.get(KEY_TIER_WHITELIST, [])

        # Find matches
        primary_matches = words & primary_keywords
        domain_matches = self._find_keyword_matches(words, text_lower, domain_keywords)

        # Check threshold (AC-2.3.2)
        if len(domain_matches) < min_matches:
            return ConceptExtractionResult()

        # Build concepts with confidence scores (AC-2.3.3)
        concepts = self._build_concepts_from_matches(
            primary_matches,
            domain_matches,
            domain_name,
            tier_whitelist,
            score_adjustments,
        )

        # Calculate domain score
        domain_score = self._calculate_domain_score(
            primary_matches, domain_matches, score_adjustments
        )

        return ConceptExtractionResult(
            concepts=concepts,
            domain_scores={domain_name: domain_score},
            total_matches=len(primary_matches) + len(domain_matches),
        )

    def _find_keyword_matches(
        self, words: set[str], text_lower: str, keywords: set[str]
    ) -> set[str]:
        """Find keyword matches including multi-word phrases.

        Args:
            words: Set of single words from text
            text_lower: Full lowercase text for phrase matching
            keywords: Keywords to match

        Returns:
            Set of matched keywords
        """
        matches = words & keywords

        # Check multi-word keywords
        for keyword in keywords:
            if " " in keyword and keyword in text_lower:
                matches.add(keyword)

        return matches

    def _build_concepts_from_matches(
        self,
        primary_matches: set[str],
        domain_matches: set[str],
        domain_name: str,
        tier_whitelist: list[str],
        score_adjustments: dict[str, Any],
    ) -> list[ExtractedConcept]:
        """Build ExtractedConcept instances from matches.

        Args:
            primary_matches: Matched primary keywords
            domain_matches: Matched domain keywords
            domain_name: Domain name
            tier_whitelist: Allowed tiers for this domain
            score_adjustments: Score adjustment configuration

        Returns:
            List of ExtractedConcept instances
        """
        concepts: list[ExtractedConcept] = []
        tier = tier_whitelist[0] if tier_whitelist else DEFAULT_UNKNOWN_DOMAIN

        # Add domain keyword concepts (higher confidence)
        domain_boost = score_adjustments.get(SCORE_DOMAIN_KEYWORD_PRESENT, DEFAULT_CONFIDENCE_BOOST)
        for keyword in domain_matches:
            confidence = min(1.0, DEFAULT_CONFIDENCE_BASE + domain_boost * 2)
            parent = self._find_parent_concept(keyword) if self._config.enable_hierarchical else None
            concepts.append(ExtractedConcept(
                name=keyword,
                confidence=confidence,
                domain=domain_name,
                tier=tier,
                parent_concept=parent,
            ))

        # Add primary keyword concepts (lower confidence if no domain match)
        for keyword in primary_matches:
            if keyword not in domain_matches:
                penalty = score_adjustments.get(SCORE_PRIMARY_ONLY_NO_DOMAIN, 0)
                confidence = max(0.0, min(1.0, DEFAULT_CONFIDENCE_BASE + penalty))
                parent = self._find_parent_concept(keyword) if self._config.enable_hierarchical else None
                concepts.append(ExtractedConcept(
                    name=keyword,
                    confidence=confidence,
                    domain=domain_name,
                    tier=tier,
                    parent_concept=parent,
                ))

        return concepts

    def _calculate_domain_score(
        self,
        primary_matches: set[str],
        domain_matches: set[str],
        score_adjustments: dict[str, Any],
    ) -> float:
        """Calculate overall domain confidence score.

        Args:
            primary_matches: Matched primary keywords
            domain_matches: Matched domain keywords
            score_adjustments: Score adjustment configuration

        Returns:
            Domain confidence score 0.0-1.0
        """
        base_score = 0.0

        if domain_matches:
            domain_boost = score_adjustments.get(SCORE_DOMAIN_KEYWORD_PRESENT, DEFAULT_CONFIDENCE_BOOST)
            base_score += len(domain_matches) * domain_boost

        if primary_matches and not domain_matches:
            penalty = score_adjustments.get(SCORE_PRIMARY_ONLY_NO_DOMAIN, 0)
            base_score += penalty
        elif primary_matches:
            base_score += len(primary_matches) * 0.05

        return min(1.0, max(0.0, base_score + DEFAULT_CONFIDENCE_BASE))

    def _find_parent_concept(self, keyword: str) -> str | None:
        """Find parent concept for hierarchical relationships (AC-2.3.4).

        Args:
            keyword: Keyword to find parent for

        Returns:
            Parent concept name or None
        """
        # Simple heuristic: map specific terms to broader concepts
        parent_map = {
            "attention": "transformer",
            "embedding": "vector",
            "langchain": "llm",
            "llamaindex": "llm",
            "fastapi": "python",
            "flask": "python",
            "django": "python",
            "kubernetes": "container",
            "docker": "container",
        }
        return parent_map.get(keyword.lower())

    def _get_primary_domain(self, domain_scores: dict[str, float]) -> str | None:
        """Get primary domain from scores.

        Args:
            domain_scores: Domain to score mapping

        Returns:
            Domain with highest score or None
        """
        if not domain_scores:
            return None
        return max(domain_scores.keys(), key=lambda d: domain_scores[d])

    def extract_concepts_from_keywords(
        self, keywords: list[str]
    ) -> ConceptExtractionResult:
        """Extract concepts from EEP-1 filtered keywords.

        Args:
            keywords: List of keywords from EEP-1 filtering

        Returns:
            ConceptExtractionResult with matched concepts
        """
        # Convert keywords to text for standard extraction
        text = " ".join(keywords)
        return self.extract_concepts(text)

    def get_domain_concepts(self, domain: str) -> list[str]:
        """Get all concepts for a domain.

        Args:
            domain: Domain name

        Returns:
            List of domain keywords/concepts
        """
        domain_config = self._domains.get(domain, {})
        primary = domain_config.get(KEY_PRIMARY_KEYWORDS, [])
        domain_kw = domain_config.get(KEY_DOMAIN_KEYWORDS, [])
        return list(set(primary + domain_kw))

    def get_tier_concepts(self, tier: str) -> list[str]:
        """Get all concepts for a tier.

        Args:
            tier: Tier name

        Returns:
            List of tier concepts
        """
        tier_config = self._tiers.get(tier, {})
        return tier_config.get(KEY_CONCEPTS, [])

    def get_tier_info(self, tier: str) -> dict[str, Any] | None:
        """Get tier metadata including priority.

        Args:
            tier: Tier name

        Returns:
            Tier configuration dict or None
        """
        return self._tiers.get(tier)

    def classify_domain(self, text: str) -> str | None:
        """Classify text into primary domain.

        Args:
            text: Text to classify

        Returns:
            Primary domain name or None
        """
        result = self.extract_concepts(text)
        return result.primary_domain

    def get_concept_hierarchy(self) -> dict[str, list[str]]:
        """Get concept hierarchy tree.

        Returns:
            Dict mapping parent concepts to children
        """
        hierarchy: dict[str, list[str]] = {}

        # Build from all extracted concepts if hierarchical enabled
        if not self._config.enable_hierarchical:
            return hierarchy

        # Simple parent-child mapping
        parent_map = {
            "transformer": ["attention", "encoder", "decoder"],
            "vector": ["embedding", "index"],
            "llm": ["langchain", "llamaindex", "openai", "anthropic"],
            "python": ["fastapi", "flask", "django", "pydantic"],
            "container": ["kubernetes", "docker"],
        }

        return parent_map


# =============================================================================
# FakeConceptExtractor for Testing
# =============================================================================


class FakeConceptExtractor:
    """Fake ConceptExtractor for testing without real taxonomy files.

    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md
    Implements ConceptExtractorProtocol for duck typing.
    """

    def __init__(
        self,
        mock_concepts: list[ExtractedConcept] | None = None,
        mock_domain_scores: dict[str, float] | None = None,
    ) -> None:
        """Initialize with optional mock data.

        Args:
            mock_concepts: Concepts to return from extract_concepts
            mock_domain_scores: Domain scores to return
        """
        self._mock_concepts = mock_concepts or []
        self._mock_domain_scores = mock_domain_scores or {}

    def extract_concepts(self, _text: str) -> ConceptExtractionResult:
        """Return mock concepts (text parameter unused in fake, matches Protocol)."""
        return ConceptExtractionResult(
            concepts=self._mock_concepts,
            domain_scores=self._mock_domain_scores,
        )

    def extract_concepts_from_keywords(
        self, keywords: list[str]
    ) -> ConceptExtractionResult:
        """Return mock concepts."""
        return self.extract_concepts(" ".join(keywords))

    def get_domain_concepts(self, domain: str) -> list[str]:
        """Return mock domain concepts."""
        return [c.name for c in self._mock_concepts if c.domain == domain]

    def get_tier_concepts(self, tier: str) -> list[str]:
        """Return mock tier concepts."""
        return [c.name for c in self._mock_concepts if c.tier == tier]

    def classify_domain(self, _text: str) -> str | None:
        """Return mock domain classification (text parameter unused, matches Protocol)."""
        if self._mock_domain_scores:
            return max(self._mock_domain_scores.keys(), key=lambda d: self._mock_domain_scores[d])
        return None
