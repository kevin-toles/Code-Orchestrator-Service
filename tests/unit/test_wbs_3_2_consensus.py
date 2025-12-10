"""
WBS 3.2 Unit Tests - Consensus Algorithm

TDD RED Phase: Tests written BEFORE implementation.
All tests should FAIL initially until GREEN phase.

WBS 3.2.1: Implement voting - Terms need ≥2/3 model agreement
WBS 3.2.2: Weighted scoring - Final score = weighted avg across models
WBS 3.2.3: Excluded terms tracking - Track why terms were rejected

Patterns Applied:
- Protocol typing for FakeAgents (CODING_PATTERNS_ANALYSIS.md line 150)
- Dataclass models for term tracking
"""


# =============================================================================
# WBS 3.2.1: Consensus Voting Tests
# =============================================================================


class TestConsensusVoting:
    """Tests for ≥2/3 model agreement voting."""

    def test_consensus_module_exists(self) -> None:
        """Consensus module can be imported."""
        from src.orchestrator.consensus import ConsensusBuilder

        assert ConsensusBuilder is not None

    def test_consensus_builder_initializes(self) -> None:
        """ConsensusBuilder initializes with default config."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder()
        assert builder is not None

    def test_consensus_requires_two_thirds_agreement(self) -> None:
        """Terms need ≥2/3 (at least 2 of 3) model agreement."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder(min_agreement=2)

        # Term agreed by all 3 models
        term_all_agree = {
            "term": "chunking",
            "generator_score": 0.9,
            "validator_approved": True,
            "ranker_score": 0.85,
        }

        # Term agreed by only 1 model
        term_one_agrees = {
            "term": "data",
            "generator_score": 0.3,
            "validator_approved": False,
            "ranker_score": 0.2,
        }

        result_all = builder.check_agreement(term_all_agree)
        result_one = builder.check_agreement(term_one_agrees)

        assert result_all["agreed"] is True
        assert result_all["models_agreed"] == 3
        assert result_one["agreed"] is False
        assert result_one["models_agreed"] < 2

    def test_consensus_counts_model_agreement(self) -> None:
        """Consensus correctly counts how many models agree on a term."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder()

        # Only generator and ranker agree (validator rejected)
        term_two_agree = {
            "term": "embedding",
            "generator_score": 0.8,
            "validator_approved": False,
            "ranker_score": 0.75,
        }

        result = builder.check_agreement(term_two_agree)
        # Generator and ranker agree (scores > threshold)
        # Validator rejected
        assert result["models_agreed"] == 2

    def test_consensus_threshold_configurable(self) -> None:
        """Minimum agreement threshold is configurable."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder(min_agreement=3)  # Require all 3

        term = {
            "term": "RAG",
            "generator_score": 0.9,
            "validator_approved": True,
            "ranker_score": 0.1,  # Low ranker score
        }

        result = builder.check_agreement(term)
        # Ranker didn't agree (score < 0.5), so only 2 models agreed
        assert result["agreed"] is False


# =============================================================================
# WBS 3.2.2: Weighted Scoring Tests
# =============================================================================


class TestWeightedScoring:
    """Tests for weighted average scoring across models."""

    def test_weighted_score_calculation(self) -> None:
        """Final score is weighted average across models."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder()

        term = {
            "term": "chunking",
            "generator_score": 0.9,
            "validator_score": 0.8,  # Validation confidence
            "ranker_score": 0.85,
        }

        final_score = builder.calculate_weighted_score(term)

        # Should be weighted average (default equal weights)
        expected = (0.9 + 0.8 + 0.85) / 3
        assert abs(final_score - expected) < 0.01

    def test_weighted_score_with_custom_weights(self) -> None:
        """Weighted score respects custom model weights."""
        from src.orchestrator.consensus import ConsensusBuilder

        # Ranker has more weight (it's the final arbiter)
        weights = {"generator": 0.2, "validator": 0.3, "ranker": 0.5}
        builder = ConsensusBuilder(model_weights=weights)

        term = {
            "term": "RAG",
            "generator_score": 0.6,
            "validator_score": 0.7,
            "ranker_score": 0.9,
        }

        final_score = builder.calculate_weighted_score(term)

        # Weighted: 0.6*0.2 + 0.7*0.3 + 0.9*0.5 = 0.12 + 0.21 + 0.45 = 0.78
        expected = 0.78
        assert abs(final_score - expected) < 0.01

    def test_score_thresholds_configurable(self) -> None:
        """Score threshold for model agreement is configurable."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder(score_threshold=0.7)

        # Score below threshold should not count as agreement
        term = {
            "term": "test",
            "generator_score": 0.6,  # Below 0.7
            "validator_approved": True,
            "ranker_score": 0.8,
        }

        result = builder.check_agreement(term)
        # Generator didn't agree (score < 0.7)
        assert result["models_agreed"] == 2


# =============================================================================
# WBS 3.2.3: Excluded Terms Tracking Tests
# =============================================================================


class TestExcludedTermsTracking:
    """Tests for tracking why terms were rejected."""

    def test_excluded_term_model_exists(self) -> None:
        """ExcludedTerm dataclass exists."""
        from src.orchestrator.consensus import ExcludedTerm

        assert ExcludedTerm is not None

    def test_excluded_term_has_required_fields(self) -> None:
        """ExcludedTerm has term, reason, and stage fields."""
        from src.orchestrator.consensus import ExcludedTerm

        excluded = ExcludedTerm(
            term="data",
            reason="Too generic",
            stage="validate",
            scores={"generator": 0.3, "validator": 0.0, "ranker": 0.2},
        )

        assert excluded.term == "data"
        assert excluded.reason == "Too generic"
        assert excluded.stage == "validate"
        assert excluded.scores["generator"] == 0.3

    def test_consensus_tracks_excluded_terms(self) -> None:
        """ConsensusBuilder tracks all excluded terms with reasons."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder()

        terms = [
            {
                "term": "chunking",
                "generator_score": 0.9,
                "validator_approved": True,
                "ranker_score": 0.85,
            },
            {
                "term": "data",
                "generator_score": 0.3,
                "validator_approved": False,
                "ranker_score": 0.2,
            },
            {
                "term": "split",
                "generator_score": 0.4,
                "validator_approved": False,
                "ranker_score": 0.3,
            },
        ]

        result = builder.build_consensus(terms)

        # chunking should be in final terms
        assert len(result.final_terms) >= 1
        assert any(t.term == "chunking" for t in result.final_terms)

        # data and split should be excluded
        assert len(result.excluded_terms) >= 2
        excluded_term_names = [t.term for t in result.excluded_terms]
        assert "data" in excluded_term_names
        assert "split" in excluded_term_names

    def test_excluded_terms_have_rejection_reason(self) -> None:
        """Each excluded term has a specific rejection reason."""
        from src.orchestrator.consensus import ConsensusBuilder

        builder = ConsensusBuilder()

        terms = [
            {
                "term": "the",
                "generator_score": 0.1,
                "validator_approved": False,
                "ranker_score": 0.1,
            },
        ]

        result = builder.build_consensus(terms)

        assert len(result.excluded_terms) == 1
        excluded = result.excluded_terms[0]
        assert excluded.reason is not None
        assert len(excluded.reason) > 0


# =============================================================================
# Consensus Result Model Tests
# =============================================================================


class TestConsensusResult:
    """Tests for ConsensusResult model."""

    def test_consensus_result_model_exists(self) -> None:
        """ConsensusResult dataclass exists."""
        from src.orchestrator.consensus import ConsensusResult

        assert ConsensusResult is not None

    def test_consensus_result_has_required_fields(self) -> None:
        """ConsensusResult has final_terms and excluded_terms."""
        from src.orchestrator.consensus import ConsensusResult

        # Check annotations exist
        assert "final_terms" in ConsensusResult.__annotations__
        assert "excluded_terms" in ConsensusResult.__annotations__

    def test_consensus_term_model_exists(self) -> None:
        """ConsensusTerm dataclass exists."""
        from src.orchestrator.consensus import ConsensusTerm

        assert ConsensusTerm is not None

    def test_consensus_term_has_required_fields(self) -> None:
        """ConsensusTerm has term, score, and models_agreed."""
        from src.orchestrator.consensus import ConsensusTerm

        term = ConsensusTerm(
            term="RAG",
            score=0.85,
            models_agreed=3,
        )

        assert term.term == "RAG"
        assert term.score == 0.85
        assert term.models_agreed == 3


# =============================================================================
# Integration with Orchestrator Tests
# =============================================================================


class TestConsensusIntegration:
    """Tests for consensus integration with orchestrator."""

    def test_orchestrator_uses_consensus_builder(self) -> None:
        """Orchestrator.run() uses ConsensusBuilder for final terms."""
        from src.orchestrator.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        result = orchestrator.run({
            "query": "LLM document chunking",
            "domain": "ai-ml",
        })

        # All returned terms should have models_agreed field
        for term in result.search_terms:
            assert hasattr(term, "models_agreed")
            assert term.models_agreed >= 2  # ≥2/3 agreement

    def test_orchestrator_result_includes_excluded_terms(self) -> None:
        """OrchestratorResult includes excluded terms list."""
        from src.orchestrator.orchestrator import Orchestrator

        orchestrator = Orchestrator()
        result = orchestrator.run({
            "query": "data processing split",  # Generic terms
            "domain": "ai-ml",
        })

        # Should have excluded_terms attribute
        assert hasattr(result, "excluded_terms")
        # Generic terms may be excluded
        assert isinstance(result.excluded_terms, list)
