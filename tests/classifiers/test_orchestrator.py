"""
Tests for HybridTieredClassifier (WBS-AC5).

TDD RED Phase: All tests written before implementation.

AC-5.1: Tier 1 Short-Circuit - Known term returns immediately
AC-5.2: Tier 2 Acceptance - Confident prediction stops cascade
AC-5.3: Tier 3 Rejection - Noise term rejected
AC-5.4: Tier 4 Fallback - Unknown term falls through to LLM
AC-5.5: Cascade Logic - Tiers checked in order 1→2→3→4
AC-5.6: Dependency Injection - All components injectable
AC-5.7: Batch Classification - Multiple terms processed
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.classifiers import (
    AliasLookupResult,
    ClassificationResult,
    FakeClassifier,
    FakeHeuristicFilter,
    FakeLLMFallback,
    HeuristicFilterResult,
    LLMFallbackResult,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Test: ClassificationResponse Dataclass (AC-5.1)
# =============================================================================


class TestClassificationResponse:
    """Test the unified ClassificationResponse dataclass."""

    def test_response_dataclass_exists(self) -> None:
        """ClassificationResponse should exist."""
        from src.classifiers.orchestrator import ClassificationResponse

        assert ClassificationResponse is not None

    def test_response_has_term_field(self) -> None:
        """ClassificationResponse should have term field."""
        from src.classifiers.orchestrator import ClassificationResponse

        response = ClassificationResponse(
            term="microservice",
            classification="concept",
            confidence=1.0,
            canonical_term="microservice",
            tier_used=1,
        )
        assert response.term == "microservice"

    def test_response_has_classification_field(self) -> None:
        """ClassificationResponse should have classification field."""
        from src.classifiers.orchestrator import ClassificationResponse

        response = ClassificationResponse(
            term="microservice",
            classification="concept",
            confidence=1.0,
            canonical_term="microservice",
            tier_used=1,
        )
        assert response.classification == "concept"

    def test_response_has_confidence_field(self) -> None:
        """ClassificationResponse should have confidence field."""
        from src.classifiers.orchestrator import ClassificationResponse

        response = ClassificationResponse(
            term="microservice",
            classification="concept",
            confidence=0.95,
            canonical_term="microservice",
            tier_used=2,
        )
        assert response.confidence == 0.95

    def test_response_has_canonical_term_field(self) -> None:
        """ClassificationResponse should have canonical_term field."""
        from src.classifiers.orchestrator import ClassificationResponse

        response = ClassificationResponse(
            term="api gateway",
            classification="concept",
            confidence=1.0,
            canonical_term="api_gateway",
            tier_used=1,
        )
        assert response.canonical_term == "api_gateway"

    def test_response_has_tier_used_field(self) -> None:
        """ClassificationResponse should have tier_used field."""
        from src.classifiers.orchestrator import ClassificationResponse

        response = ClassificationResponse(
            term="microservice",
            classification="concept",
            confidence=1.0,
            canonical_term="microservice",
            tier_used=1,
        )
        assert response.tier_used == 1

    def test_response_has_rejection_reason_optional(self) -> None:
        """ClassificationResponse should have optional rejection_reason field."""
        from src.classifiers.orchestrator import ClassificationResponse

        # Without rejection
        response1 = ClassificationResponse(
            term="microservice",
            classification="concept",
            confidence=1.0,
            canonical_term="microservice",
            tier_used=1,
        )
        assert response1.rejection_reason is None

        # With rejection
        response2 = ClassificationResponse(
            term="www",
            classification="rejected",
            confidence=1.0,
            canonical_term="www",
            tier_used=3,
            rejection_reason="noise_url_fragments",
        )
        assert response2.rejection_reason == "noise_url_fragments"

    def test_response_is_frozen(self) -> None:
        """ClassificationResponse should be immutable."""
        from src.classifiers.orchestrator import ClassificationResponse

        response = ClassificationResponse(
            term="microservice",
            classification="concept",
            confidence=1.0,
            canonical_term="microservice",
            tier_used=1,
        )
        with pytest.raises(AttributeError):
            response.term = "changed"  # type: ignore[misc]


# =============================================================================
# Test: HybridTieredClassifier Protocol (AC-5.6)
# =============================================================================


class TestHybridTieredClassifierProtocol:
    """Test the HybridTieredClassifierProtocol definition."""

    def test_protocol_exists(self) -> None:
        """HybridTieredClassifierProtocol should exist."""
        from src.classifiers.orchestrator import HybridTieredClassifierProtocol

        assert HybridTieredClassifierProtocol is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should be runtime_checkable for isinstance checks."""
        from typing import runtime_checkable

        from src.classifiers.orchestrator import HybridTieredClassifierProtocol

        assert hasattr(HybridTieredClassifierProtocol, "__protocol_attrs__") or hasattr(
            HybridTieredClassifierProtocol, "_is_runtime_protocol"
        )

    def test_classifier_passes_protocol(self) -> None:
        """HybridTieredClassifier should pass Protocol check."""
        from src.classifiers.orchestrator import (
            HybridTieredClassifier,
            HybridTieredClassifierProtocol,
        )

        # Create with mock dependencies
        mock_alias = MagicMock()
        mock_trained = FakeClassifier(responses={})
        mock_heuristic = FakeHeuristicFilter()
        mock_llm = FakeLLMFallback()

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        assert isinstance(classifier, HybridTieredClassifierProtocol)


# =============================================================================
# Test: Dependency Injection (AC-5.6)
# =============================================================================


class TestDependencyInjection:
    """Test that all 4 tier components are injectable via constructor."""

    def test_accepts_alias_lookup(self) -> None:
        """Should accept AliasLookup instance."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        assert classifier._alias_lookup is mock_alias

    def test_accepts_trained_classifier(self) -> None:
        """Should accept TrainedClassifier instance."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_trained = FakeClassifier(responses={})

        classifier = HybridTieredClassifier(
            alias_lookup=MagicMock(),
            trained_classifier=mock_trained,
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        assert classifier._trained_classifier is mock_trained

    def test_accepts_heuristic_filter(self) -> None:
        """Should accept HeuristicFilter instance."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_heuristic = FakeHeuristicFilter()

        classifier = HybridTieredClassifier(
            alias_lookup=MagicMock(),
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=mock_heuristic,
            llm_fallback=FakeLLMFallback(),
        )

        assert classifier._heuristic_filter is mock_heuristic

    def test_accepts_llm_fallback(self) -> None:
        """Should accept LLMFallback instance."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_llm = FakeLLMFallback()

        classifier = HybridTieredClassifier(
            alias_lookup=MagicMock(),
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=mock_llm,
        )

        assert classifier._llm_fallback is mock_llm

    def test_all_components_required(self) -> None:
        """All 4 components should be required (no defaults)."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        with pytest.raises(TypeError):
            HybridTieredClassifier()  # type: ignore[call-arg]


# =============================================================================
# Test: Tier 1 Short-Circuit (AC-5.1)
# =============================================================================


class TestTier1ShortCircuit:
    """Test that known terms in alias lookup short-circuit the cascade."""

    @pytest.mark.asyncio
    async def test_tier1_hit_returns_immediately(self) -> None:
        """When term found in Tier 1, should return without checking other tiers."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        # Mock Tier 1 to return a result
        mock_alias = MagicMock()
        mock_alias.get.return_value = AliasLookupResult(
            canonical_term="microservice",
            classification="concept",
            confidence=1.0,
            tier_used=1,
        )

        # Create mocks for other tiers (should NOT be called)
        mock_trained = MagicMock()
        mock_heuristic = MagicMock()
        mock_llm = MagicMock()

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("microservice")

        # Verify Tier 1 was called
        mock_alias.get.assert_called_once_with("microservice")

        # Verify other tiers were NOT called
        mock_trained.predict.assert_not_called()
        mock_heuristic.check.assert_not_called()
        mock_llm.classify.assert_not_called()

        # Verify result
        assert result.tier_used == 1
        assert result.classification == "concept"
        assert result.confidence == 1.0
        assert result.canonical_term == "microservice"

    @pytest.mark.asyncio
    async def test_tier1_returns_canonical_term(self) -> None:
        """Tier 1 hit should return canonical term from alias lookup."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = AliasLookupResult(
            canonical_term="api_gateway",  # Canonical form
            classification="concept",
            confidence=1.0,
            tier_used=1,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        result = await classifier.classify("API Gateway")  # Original form

        assert result.canonical_term == "api_gateway"
        assert result.term == "API Gateway"

    @pytest.mark.asyncio
    async def test_tier1_case_insensitive(self) -> None:
        """Tier 1 lookup should be case-insensitive (handled by AliasLookup)."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = AliasLookupResult(
            canonical_term="kubernetes",
            classification="concept",
            confidence=1.0,
            tier_used=1,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        result = await classifier.classify("KUBERNETES")

        assert result.tier_used == 1


# =============================================================================
# Test: Tier 2 Acceptance (AC-5.2)
# =============================================================================


class TestTier2Acceptance:
    """Test that confident Tier 2 predictions stop the cascade."""

    @pytest.mark.asyncio
    async def test_tier2_confident_returns(self) -> None:
        """When Tier 2 is confident (>=0.7), should return without Tier 3/4."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        # Tier 1 miss
        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        # Tier 2 confident prediction
        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="concept",
            confidence=0.85,
            tier_used=2,
        )

        # Tiers 3/4 should NOT be called
        mock_heuristic = MagicMock()
        mock_llm = MagicMock()

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("machine learning")

        # Verify cascade
        mock_alias.get.assert_called_once()
        mock_trained.predict.assert_called_once_with("machine learning")
        mock_heuristic.check.assert_not_called()
        mock_llm.classify.assert_not_called()

        # Verify result
        assert result.tier_used == 2
        assert result.classification == "concept"
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_tier2_exactly_threshold(self) -> None:
        """Confidence exactly at 0.7 should accept."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="keyword",
            confidence=0.7,  # Exactly at threshold
            tier_used=2,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        result = await classifier.classify("test term")

        assert result.tier_used == 2
        assert result.classification == "keyword"

    @pytest.mark.asyncio
    async def test_tier2_low_confidence_continues(self) -> None:
        """When Tier 2 confidence < 0.7, should continue to Tier 3."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown",
            confidence=0.5,  # Below threshold
            tier_used=2,
        )

        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = None  # Not noise

        mock_llm = AsyncMock()
        mock_llm.classify.return_value = LLMFallbackResult(
            classification="concept",
            confidence=0.9,
            canonical_term="ambiguous_term",
            tier_used=4,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("ambiguous term")

        # Should have checked Tier 3
        mock_heuristic.check.assert_called_once()
        # Should have reached Tier 4
        assert result.tier_used == 4


# =============================================================================
# Test: Tier 3 Rejection (AC-5.3)
# =============================================================================


class TestTier3Rejection:
    """Test that noise terms are rejected at Tier 3."""

    @pytest.mark.asyncio
    async def test_tier3_rejects_noise(self) -> None:
        """Noise term should be rejected with rejection_reason."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        # Tier 1 miss
        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        # Tier 2 low confidence
        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown",
            confidence=0.3,
            tier_used=2,
        )

        # Tier 3 detects noise
        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = HeuristicFilterResult(
            rejection_reason="noise_url_fragments",
            matched_term="www",
            category="url_fragments",
        )

        # Tier 4 should NOT be called
        mock_llm = MagicMock()

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("www")

        # Verify Tier 4 NOT called
        mock_llm.classify.assert_not_called()

        # Verify rejection result
        assert result.tier_used == 3
        assert result.classification == "rejected"
        assert result.rejection_reason == "noise_url_fragments"

    @pytest.mark.asyncio
    async def test_tier3_watermark_rejection(self) -> None:
        """Watermark terms should be rejected."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown", confidence=0.4, tier_used=2
        )

        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = HeuristicFilterResult(
            rejection_reason="noise_watermarks",
            matched_term="manning publications",
            category="watermarks",
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=FakeLLMFallback(),
        )

        result = await classifier.classify("manning publications")

        assert result.rejection_reason == "noise_watermarks"

    @pytest.mark.asyncio
    async def test_tier3_pass_continues_to_tier4(self) -> None:
        """When Tier 3 passes (no noise), should continue to Tier 4."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown", confidence=0.4, tier_used=2
        )

        # Tier 3 passes
        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = None

        mock_llm = AsyncMock()
        mock_llm.classify.return_value = LLMFallbackResult(
            classification="concept",
            confidence=0.88,
            canonical_term="novel_concept",
            tier_used=4,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("novel concept")

        # Tier 4 should be called
        mock_llm.classify.assert_called_once_with("novel concept")
        assert result.tier_used == 4


# =============================================================================
# Test: Tier 4 Fallback (AC-5.4)
# =============================================================================


class TestTier4Fallback:
    """Test that unknown terms fall through to Tier 4 LLM."""

    @pytest.mark.asyncio
    async def test_tier4_called_when_all_miss(self) -> None:
        """When Tiers 1-3 all miss/pass, Tier 4 should be called."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown", confidence=0.5, tier_used=2
        )

        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = None

        mock_llm = AsyncMock()
        mock_llm.classify.return_value = LLMFallbackResult(
            classification="concept",
            confidence=0.92,
            canonical_term="emerging_technology",
            tier_used=4,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("emerging technology")

        mock_llm.classify.assert_called_once_with("emerging technology")
        assert result.tier_used == 4
        assert result.classification == "concept"
        assert result.confidence == 0.92
        assert result.canonical_term == "emerging_technology"

    @pytest.mark.asyncio
    async def test_tier4_async_call(self) -> None:
        """Tier 4 classify should be awaited (async)."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown", confidence=0.4, tier_used=2
        )

        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = None

        # Use FakeLLMFallback (async implementation)
        fake_llm = FakeLLMFallback(
            responses={
                "async test": LLMFallbackResult(
                    classification="keyword",
                    confidence=0.75,
                    canonical_term="async_test",
                    tier_used=4,
                )
            }
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=fake_llm,
        )

        result = await classifier.classify("async test")

        assert result.tier_used == 4
        assert result.classification == "keyword"


# =============================================================================
# Test: Full Cascade Logic (AC-5.5)
# =============================================================================


class TestCascadeLogic:
    """Test the complete cascade 1→2→3→4."""

    @pytest.mark.asyncio
    async def test_full_cascade_order(self) -> None:
        """Verify tiers are checked in order 1→2→3→4."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        call_order: list[str] = []

        # Track call order
        mock_alias = MagicMock()

        def alias_side_effect(term: str) -> None:
            call_order.append("tier1")
            return None

        mock_alias.get.side_effect = alias_side_effect

        mock_trained = MagicMock()

        def trained_side_effect(term: str) -> ClassificationResult:
            call_order.append("tier2")
            return ClassificationResult(
                predicted_label="unknown", confidence=0.4, tier_used=2
            )

        mock_trained.predict.side_effect = trained_side_effect

        mock_heuristic = MagicMock()

        def heuristic_side_effect(term: str) -> None:
            call_order.append("tier3")
            return None

        mock_heuristic.check.side_effect = heuristic_side_effect

        mock_llm = AsyncMock()

        async def llm_side_effect(term: str) -> LLMFallbackResult:
            call_order.append("tier4")
            return LLMFallbackResult(
                classification="concept",
                confidence=0.9,
                canonical_term="test_term",
                tier_used=4,
            )

        mock_llm.classify.side_effect = llm_side_effect

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        await classifier.classify("test term")

        assert call_order == ["tier1", "tier2", "tier3", "tier4"]

    @pytest.mark.asyncio
    async def test_cascade_stops_at_tier1(self) -> None:
        """Cascade should stop at Tier 1 if term found."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        call_order: list[str] = []

        mock_alias = MagicMock()

        def alias_side_effect(term: str) -> AliasLookupResult:
            call_order.append("tier1")
            return AliasLookupResult(
                canonical_term="known_term",
                classification="concept",
                confidence=1.0,
                tier_used=1,
            )

        mock_alias.get.side_effect = alias_side_effect

        mock_trained = MagicMock()
        mock_trained.predict.side_effect = lambda t: call_order.append("tier2")

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        await classifier.classify("known term")

        assert call_order == ["tier1"]

    @pytest.mark.asyncio
    async def test_cascade_stops_at_tier2(self) -> None:
        """Cascade should stop at Tier 2 if confident."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        call_order: list[str] = []

        mock_alias = MagicMock()

        def alias_side_effect(term: str) -> None:
            call_order.append("tier1")
            return None

        mock_alias.get.side_effect = alias_side_effect

        mock_trained = MagicMock()

        def trained_side_effect(term: str) -> ClassificationResult:
            call_order.append("tier2")
            return ClassificationResult(
                predicted_label="concept", confidence=0.85, tier_used=2
            )

        mock_trained.predict.side_effect = trained_side_effect

        mock_heuristic = MagicMock()
        mock_heuristic.check.side_effect = lambda t: call_order.append("tier3")

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=FakeLLMFallback(),
        )

        await classifier.classify("confident term")

        assert call_order == ["tier1", "tier2"]

    @pytest.mark.asyncio
    async def test_cascade_stops_at_tier3_rejection(self) -> None:
        """Cascade should stop at Tier 3 if noise detected."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        call_order: list[str] = []

        mock_alias = MagicMock()
        mock_alias.get.side_effect = lambda t: (call_order.append("tier1"), None)[1]

        mock_trained = MagicMock()
        mock_trained.predict.side_effect = lambda t: (
            call_order.append("tier2"),
            ClassificationResult(
                predicted_label="unknown", confidence=0.3, tier_used=2
            ),
        )[1]

        mock_heuristic = MagicMock()

        def heuristic_side_effect(term: str) -> HeuristicFilterResult:
            call_order.append("tier3")
            return HeuristicFilterResult(
                rejection_reason="noise_watermarks",
                matched_term=term,
                category="watermarks",
            )

        mock_heuristic.check.side_effect = heuristic_side_effect

        mock_llm = AsyncMock()
        mock_llm.classify.side_effect = lambda t: call_order.append("tier4")

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        await classifier.classify("noise term")

        assert call_order == ["tier1", "tier2", "tier3"]


# =============================================================================
# Test: Batch Classification (AC-5.7)
# =============================================================================


class TestBatchClassification:
    """Test classify_batch processes multiple terms."""

    @pytest.mark.asyncio
    async def test_batch_returns_list(self) -> None:
        """classify_batch should return list of ClassificationResponse."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = AliasLookupResult(
            canonical_term="test",
            classification="concept",
            confidence=1.0,
            tier_used=1,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        results = await classifier.classify_batch(["term1", "term2", "term3"])

        assert isinstance(results, list)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_empty_list(self) -> None:
        """classify_batch with empty list should return empty list."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        classifier = HybridTieredClassifier(
            alias_lookup=MagicMock(),
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        results = await classifier.classify_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self) -> None:
        """classify_batch should return results in same order as input."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        # Return different results based on term
        def alias_lookup(term: str) -> AliasLookupResult | None:
            if term == "known":
                return AliasLookupResult(
                    canonical_term="known",
                    classification="concept",
                    confidence=1.0,
                    tier_used=1,
                )
            return None

        mock_alias = MagicMock()
        mock_alias.get.side_effect = alias_lookup

        mock_trained = FakeClassifier(
            responses={
                "unknown1": ("keyword", 0.8),
                "unknown2": ("concept", 0.75),
            }
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        terms = ["known", "unknown1", "unknown2"]
        results = await classifier.classify_batch(terms)

        assert results[0].term == "known"
        assert results[0].tier_used == 1

        assert results[1].term == "unknown1"
        assert results[1].classification == "keyword"

        assert results[2].term == "unknown2"
        assert results[2].classification == "concept"

    @pytest.mark.asyncio
    async def test_batch_mixed_tiers(self) -> None:
        """classify_batch should handle terms that resolve at different tiers."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        def alias_lookup(term: str) -> AliasLookupResult | None:
            if term == "tier1_term":
                return AliasLookupResult(
                    canonical_term="tier1_term",
                    classification="concept",
                    confidence=1.0,
                    tier_used=1,
                )
            return None

        mock_alias = MagicMock()
        mock_alias.get.side_effect = alias_lookup

        def trained_predict(term: str) -> ClassificationResult:
            if term == "tier2_term":
                return ClassificationResult(
                    predicted_label="keyword", confidence=0.9, tier_used=2
                )
            return ClassificationResult(
                predicted_label="unknown", confidence=0.3, tier_used=2
            )

        mock_trained = MagicMock()
        mock_trained.predict.side_effect = trained_predict

        def heuristic_check(term: str) -> HeuristicFilterResult | None:
            if term == "noise_term":
                return HeuristicFilterResult(
                    rejection_reason="noise_generic_filler",
                    matched_term=term,
                    category="generic_filler",
                )
            return None

        mock_heuristic = MagicMock()
        mock_heuristic.check.side_effect = heuristic_check

        mock_llm = FakeLLMFallback(
            responses={
                "tier4_term": LLMFallbackResult(
                    classification="concept",
                    confidence=0.88,
                    canonical_term="tier4_term",
                    tier_used=4,
                )
            }
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        terms = ["tier1_term", "tier2_term", "noise_term", "tier4_term"]
        results = await classifier.classify_batch(terms)

        assert results[0].tier_used == 1
        assert results[1].tier_used == 2
        assert results[2].tier_used == 3
        assert results[2].rejection_reason == "noise_generic_filler"
        assert results[3].tier_used == 4


# =============================================================================
# Test: Constants (AC-8.1)
# =============================================================================


class TestConstants:
    """Test module-level constants are defined."""

    def test_confidence_threshold_exists(self) -> None:
        """CONFIDENCE_THRESHOLD should be defined."""
        from src.classifiers.orchestrator import CONFIDENCE_THRESHOLD

        assert CONFIDENCE_THRESHOLD == 0.7

    def test_rejection_classification_exists(self) -> None:
        """CLASSIFICATION_REJECTED should be defined."""
        from src.classifiers.orchestrator import CLASSIFICATION_REJECTED

        assert CLASSIFICATION_REJECTED == "rejected"

    def test_tier_constants_exist(self) -> None:
        """Tier constants should be defined."""
        from src.classifiers.orchestrator import (
            TIER_ALIAS_LOOKUP,
            TIER_HEURISTIC_FILTER,
            TIER_LLM_FALLBACK,
            TIER_TRAINED_CLASSIFIER,
        )

        assert TIER_ALIAS_LOOKUP == 1
        assert TIER_TRAINED_CLASSIFIER == 2
        assert TIER_HEURISTIC_FILTER == 3
        assert TIER_LLM_FALLBACK == 4


# =============================================================================
# Test: FakeHybridTieredClassifier (AC-8.4)
# =============================================================================


class TestFakeHybridTieredClassifier:
    """Test the fake implementation for testing."""

    def test_fake_exists(self) -> None:
        """FakeHybridTieredClassifier should exist."""
        from src.classifiers.orchestrator import FakeHybridTieredClassifier

        assert FakeHybridTieredClassifier is not None

    def test_fake_passes_protocol(self) -> None:
        """Fake should pass Protocol check."""
        from src.classifiers.orchestrator import (
            FakeHybridTieredClassifier,
            HybridTieredClassifierProtocol,
        )

        fake = FakeHybridTieredClassifier()
        assert isinstance(fake, HybridTieredClassifierProtocol)

    @pytest.mark.asyncio
    async def test_fake_returns_configured_response(self) -> None:
        """Fake should return pre-configured responses."""
        from src.classifiers.orchestrator import (
            ClassificationResponse,
            FakeHybridTieredClassifier,
        )

        response = ClassificationResponse(
            term="test",
            classification="concept",
            confidence=0.95,
            canonical_term="test",
            tier_used=1,
        )

        fake = FakeHybridTieredClassifier(responses={"test": response})

        result = await fake.classify("test")

        assert result == response

    @pytest.mark.asyncio
    async def test_fake_default_response(self) -> None:
        """Fake should return default for unconfigured terms."""
        from src.classifiers.orchestrator import FakeHybridTieredClassifier

        fake = FakeHybridTieredClassifier()

        result = await fake.classify("unknown_term")

        assert result.term == "unknown_term"
        assert result.classification == "unknown"
        assert result.tier_used == 4

    @pytest.mark.asyncio
    async def test_fake_classify_batch(self) -> None:
        """Fake should support classify_batch."""
        from src.classifiers.orchestrator import (
            ClassificationResponse,
            FakeHybridTieredClassifier,
        )

        response1 = ClassificationResponse(
            term="term1",
            classification="concept",
            confidence=1.0,
            canonical_term="term1",
            tier_used=1,
        )

        fake = FakeHybridTieredClassifier(responses={"term1": response1})

        results = await fake.classify_batch(["term1", "term2"])

        assert len(results) == 2
        assert results[0] == response1
        assert results[1].term == "term2"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_term(self) -> None:
        """Empty term should be handled gracefully."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown", confidence=0.0, tier_used=2
        )

        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = None

        mock_llm = FakeLLMFallback()

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("")

        assert result.term == ""

    @pytest.mark.asyncio
    async def test_whitespace_term(self) -> None:
        """Whitespace-only term should be handled."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = None

        mock_trained = MagicMock()
        mock_trained.predict.return_value = ClassificationResult(
            predicted_label="unknown", confidence=0.0, tier_used=2
        )

        mock_heuristic = MagicMock()
        mock_heuristic.check.return_value = None

        mock_llm = FakeLLMFallback()

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=mock_trained,
            heuristic_filter=mock_heuristic,
            llm_fallback=mock_llm,
        )

        result = await classifier.classify("   ")

        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters(self) -> None:
        """Terms with special characters should be handled."""
        from src.classifiers.orchestrator import HybridTieredClassifier

        mock_alias = MagicMock()
        mock_alias.get.return_value = AliasLookupResult(
            canonical_term="c++",
            classification="keyword",
            confidence=1.0,
            tier_used=1,
        )

        classifier = HybridTieredClassifier(
            alias_lookup=mock_alias,
            trained_classifier=FakeClassifier(responses={}),
            heuristic_filter=FakeHeuristicFilter(),
            llm_fallback=FakeLLMFallback(),
        )

        result = await classifier.classify("C++")

        assert result.canonical_term == "c++"
