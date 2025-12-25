"""
Hybrid Tiered Classifier Orchestrator (Tier Orchestration).

WBS: WBS-AC5 - Orchestrator Pipeline
AC Block: AC-5.1 through AC-5.7

This module orchestrates the 4-tier classification cascade:
1. Tier 1: Alias Lookup (O(1) hash, confidence=1.0)
2. Tier 2: Trained Classifier (SBERT + LogisticRegression, threshold=0.7)
3. Tier 3: Heuristic Filter (noise detection, reject noise terms)
4. Tier 4: LLM Fallback (ai-agents call for unknown terms)

Pattern: Pipeline / Chain of Responsibility
- Each tier has veto power to stop the cascade
- Dependency injection for all tier components

Reference: HYBRID_TIERED_CLASSIFIER_ARCHITECTURE.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.classifiers.alias_lookup import AliasLookup
    from src.classifiers.heuristic_filter import HeuristicFilterProtocol
    from src.classifiers.llm_fallback import LLMFallbackProtocol
    from src.classifiers.trained_classifier import ConceptClassifierProtocol

# =============================================================================
# Constants (AC-8.1: No duplicated strings)
# =============================================================================

CONFIDENCE_THRESHOLD: Final[float] = 0.7
CLASSIFICATION_REJECTED: Final[str] = "rejected"
CLASSIFICATION_UNKNOWN: Final[str] = "unknown"

# Tier identifiers
TIER_ALIAS_LOOKUP: Final[int] = 1
TIER_TRAINED_CLASSIFIER: Final[int] = 2
TIER_HEURISTIC_FILTER: Final[int] = 3
TIER_LLM_FALLBACK: Final[int] = 4


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClassificationResponse:
    """Unified response from the Hybrid Tiered Classifier.

    Attributes:
        term: The original term that was classified
        classification: Either "concept", "keyword", "rejected", or "unknown"
        confidence: Confidence score (0.0 to 1.0)
        canonical_term: Normalized/canonical form of the term
        tier_used: Which tier produced the result (1-4)
        rejection_reason: Reason for rejection if classification="rejected"
    """

    term: str
    classification: str
    confidence: float
    canonical_term: str
    tier_used: int
    rejection_reason: str | None = field(default=None)


# =============================================================================
# Protocol Definition (AC-8.4)
# =============================================================================


@runtime_checkable
class HybridTieredClassifierProtocol(Protocol):
    """Protocol for Hybrid Tiered Classifier implementations.

    Enables dependency injection and test doubles.
    """

    async def classify(self, term: str) -> ClassificationResponse:
        """Classify a single term through the tier cascade.

        Args:
            term: The term to classify

        Returns:
            ClassificationResponse with classification details
        """
        ...

    async def classify_batch(self, terms: list[str]) -> list[ClassificationResponse]:
        """Classify multiple terms.

        Args:
            terms: List of terms to classify

        Returns:
            List of ClassificationResponse objects in same order as input
        """
        ...


# =============================================================================
# Main Implementation
# =============================================================================


class HybridTieredClassifier:
    """Orchestrates the 4-tier classification cascade.

    The classifier checks each tier in order, stopping when a tier
    produces a definitive result:

    1. Tier 1 (Alias Lookup): Exact match → return immediately
    2. Tier 2 (Trained Classifier): Confident (>=0.7) → return
    3. Tier 3 (Heuristic Filter): Noise detected → reject
    4. Tier 4 (LLM Fallback): Unknown term → call LLM

    All tier components are injectable via constructor for testing.

    Example:
        classifier = HybridTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=trained_classifier,
            heuristic_filter=heuristic_filter,
            llm_fallback=llm_fallback,
        )
        result = await classifier.classify("microservice")
    """

    __slots__ = (
        "_alias_lookup",
        "_trained_classifier",
        "_heuristic_filter",
        "_llm_fallback",
    )

    def __init__(
        self,
        alias_lookup: AliasLookup,
        trained_classifier: ConceptClassifierProtocol,
        heuristic_filter: HeuristicFilterProtocol,
        llm_fallback: LLMFallbackProtocol,
    ) -> None:
        """Initialize with all 4 tier components.

        Args:
            alias_lookup: Tier 1 - O(1) hash lookup
            trained_classifier: Tier 2 - SBERT + LogisticRegression
            heuristic_filter: Tier 3 - Noise term detection
            llm_fallback: Tier 4 - LLM validation via ai-agents
        """
        self._alias_lookup = alias_lookup
        self._trained_classifier = trained_classifier
        self._heuristic_filter = heuristic_filter
        self._llm_fallback = llm_fallback

    async def classify(self, term: str) -> ClassificationResponse:
        """Classify a term through the 4-tier cascade.

        Tiers are checked in order 1→2→3→4. Each tier can:
        - Return a result (stops cascade)
        - Pass to next tier (cascade continues)
        - Reject the term (Tier 3 only, stops cascade)

        Args:
            term: The term to classify

        Returns:
            ClassificationResponse with tier_used indicating which tier
            produced the final result
        """
        # Tier 1: Alias Lookup (O(1) hash)
        tier1_result = self._check_tier1(term)
        if tier1_result is not None:
            return tier1_result

        # Tier 2: Trained Classifier
        tier2_result = self._check_tier2(term)
        if tier2_result is not None:
            return tier2_result

        # Tier 3: Heuristic Filter (noise detection)
        tier3_result = self._check_tier3(term)
        if tier3_result is not None:
            return tier3_result

        # Tier 4: LLM Fallback (final tier)
        return await self._check_tier4(term)

    async def classify_batch(self, terms: list[str]) -> list[ClassificationResponse]:
        """Classify multiple terms.

        Each term goes through the full cascade independently.
        Results are returned in the same order as input.

        Args:
            terms: List of terms to classify

        Returns:
            List of ClassificationResponse objects
        """
        if not terms:
            return []

        results: list[ClassificationResponse] = []
        for term in terms:
            result = await self.classify(term)
            results.append(result)

        return results

    def _check_tier1(self, term: str) -> ClassificationResponse | None:
        """Check Tier 1: Alias Lookup.

        Args:
            term: The term to look up

        Returns:
            ClassificationResponse if found, None to continue cascade
        """
        result = self._alias_lookup.get(term)
        if result is None:
            return None

        return ClassificationResponse(
            term=term,
            classification=result.classification,
            confidence=result.confidence,
            canonical_term=result.canonical_term,
            tier_used=TIER_ALIAS_LOOKUP,
        )

    def _check_tier2(self, term: str) -> ClassificationResponse | None:
        """Check Tier 2: Trained Classifier.

        Only returns a result if confidence >= CONFIDENCE_THRESHOLD.

        Args:
            term: The term to classify

        Returns:
            ClassificationResponse if confident, None to continue cascade
        """
        result = self._trained_classifier.predict(term)

        if result.confidence >= CONFIDENCE_THRESHOLD:
            return ClassificationResponse(
                term=term,
                classification=result.predicted_label,
                confidence=result.confidence,
                canonical_term=term,  # Tier 2 doesn't normalize
                tier_used=TIER_TRAINED_CLASSIFIER,
            )

        return None

    def _check_tier3(self, term: str) -> ClassificationResponse | None:
        """Check Tier 3: Heuristic Filter.

        Detects noise terms and rejects them.

        Args:
            term: The term to check

        Returns:
            ClassificationResponse with rejection if noise, None to continue
        """
        result = self._heuristic_filter.check(term)
        if result is None:
            return None

        return ClassificationResponse(
            term=term,
            classification=CLASSIFICATION_REJECTED,
            confidence=1.0,  # Noise rejection is definitive
            canonical_term=term,
            tier_used=TIER_HEURISTIC_FILTER,
            rejection_reason=result.rejection_reason,
        )

    async def _check_tier4(self, term: str) -> ClassificationResponse:
        """Check Tier 4: LLM Fallback.

        Final tier - always produces a result.

        Args:
            term: The term to classify

        Returns:
            ClassificationResponse from LLM validation
        """
        result = await self._llm_fallback.classify(term)

        return ClassificationResponse(
            term=term,
            classification=result.classification,
            confidence=result.confidence,
            canonical_term=result.canonical_term,
            tier_used=TIER_LLM_FALLBACK,
        )


# =============================================================================
# Fake Implementation (AC-8.4: Protocol-based fakes)
# =============================================================================


class SyncTieredClassifier:
    """Synchronous classifier using Tiers 1-3 only (no LLM).

    This classifier is useful for batch processing where:
    - Most terms are handled by Tiers 1-3 (>95%)
    - Remaining "unknown" terms can be processed separately via async classify

    The sync classifier NEVER calls Tier 4 (LLM). Terms that would
    reach Tier 4 are returned with classification="unknown".

    Usage:
        sync_classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=trained_classifier,
            heuristic_filter=heuristic_filter,
        )
        result = sync_classifier.classify("microservice")  # No await needed
    """

    __slots__ = (
        "_alias_lookup",
        "_trained_classifier",
        "_heuristic_filter",
    )

    def __init__(
        self,
        alias_lookup: AliasLookup,
        trained_classifier: ConceptClassifierProtocol,
        heuristic_filter: HeuristicFilterProtocol,
    ) -> None:
        """Initialize with Tiers 1-3 components.

        Args:
            alias_lookup: Tier 1 - O(1) hash lookup
            trained_classifier: Tier 2 - SBERT + LogisticRegression
            heuristic_filter: Tier 3 - Noise term detection
        """
        self._alias_lookup = alias_lookup
        self._trained_classifier = trained_classifier
        self._heuristic_filter = heuristic_filter

    def classify(self, term: str) -> ClassificationResponse:
        """Classify a term through Tiers 1-3 synchronously.

        Terms not resolved by Tiers 1-3 return classification="unknown".

        Args:
            term: The term to classify

        Returns:
            ClassificationResponse with tier_used 1-3, or unknown
        """
        # Tier 1: Alias Lookup
        result = self._alias_lookup.get(term)
        if result is not None:
            return ClassificationResponse(
                term=term,
                classification=result.classification,
                confidence=result.confidence,
                canonical_term=result.canonical_term,
                tier_used=TIER_ALIAS_LOOKUP,
            )

        # Tier 2: Trained Classifier
        tier2_result = self._trained_classifier.predict(term)
        if tier2_result.confidence >= CONFIDENCE_THRESHOLD:
            return ClassificationResponse(
                term=term,
                classification=tier2_result.predicted_label,
                confidence=tier2_result.confidence,
                canonical_term=term,
                tier_used=TIER_TRAINED_CLASSIFIER,
            )

        # Tier 3: Heuristic Filter
        tier3_result = self._heuristic_filter.check(term)
        if tier3_result is not None:
            return ClassificationResponse(
                term=term,
                classification=CLASSIFICATION_REJECTED,
                confidence=1.0,
                canonical_term=term,
                tier_used=TIER_HEURISTIC_FILTER,
                rejection_reason=tier3_result.rejection_reason,
            )

        # Would need Tier 4 - return unknown for sync processing
        return ClassificationResponse(
            term=term,
            classification=CLASSIFICATION_UNKNOWN,
            confidence=0.0,
            canonical_term=term,
            tier_used=TIER_HEURISTIC_FILTER,  # Last tier checked
        )

    def classify_batch(self, terms: list[str]) -> list[ClassificationResponse]:
        """Classify multiple terms synchronously.

        Args:
            terms: List of terms to classify

        Returns:
            List of ClassificationResponse objects
        """
        return [self.classify(term) for term in terms]


class FakeHybridTieredClassifier:
    """Fake implementation for testing.

    Returns pre-configured responses without running the actual cascade.

    Usage:
        fake = FakeHybridTieredClassifier(responses={"term": response})
        result = await fake.classify("term")
    """

    def __init__(
        self,
        responses: Mapping[str, ClassificationResponse] | None = None,
    ) -> None:
        """Initialize with pre-configured responses.

        Args:
            responses: Dict mapping terms to expected responses
        """
        self._responses: dict[str, ClassificationResponse] = (
            dict(responses) if responses else {}
        )

    async def classify(self, term: str) -> ClassificationResponse:
        """Return pre-configured response or default.

        Args:
            term: The term to classify

        Returns:
            Configured response or default unknown response
        """
        if term in self._responses:
            return self._responses[term]

        return ClassificationResponse(
            term=term,
            classification=CLASSIFICATION_UNKNOWN,
            confidence=0.5,
            canonical_term=term,
            tier_used=TIER_LLM_FALLBACK,
        )

    async def classify_batch(self, terms: list[str]) -> list[ClassificationResponse]:
        """Classify multiple terms using configured responses.

        Args:
            terms: List of terms to classify

        Returns:
            List of ClassificationResponse objects
        """
        return [await self.classify(term) for term in terms]
