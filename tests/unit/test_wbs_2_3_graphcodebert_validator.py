"""
Code-Orchestrator-Service - GraphCodeBERT Validator Tests

WBS 2.3: GraphCodeBERT Validator (Model Wrapper)
TDD Phase: RED - Write failing tests first

Tests for GraphCodeBERTValidator that validates and filters technical terms
using GraphCodeBERT embeddings for semantic similarity.

NOTE: These are HuggingFace model wrappers, NOT autonomous agents.
Autonomous agents (LangGraph workflows) live in the ai-agents service.

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- FakeClient for Testing (no real HuggingFace model in unit tests)
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached model from registry)
- #12: Embedding cache to avoid recomputation
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _create_mock_model_with_embeddings() -> tuple[MagicMock, MagicMock]:
    """Create mock model and tokenizer that return proper embedding tensors.

    Returns embeddings that simulate GraphCodeBERT's 768-dim output.
    Uses deterministic embeddings based on input hash for reproducibility.
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock tokenizer to return proper tensor dict
    def mock_tokenize(*args, **kwargs):
        text = args[0] if args else kwargs.get("text", "")
        # Create mock input tensors
        seq_len = min(len(str(text).split()), 512)
        return {
            "input_ids": torch.zeros(1, max(seq_len, 10)),
            "attention_mask": torch.ones(1, max(seq_len, 10)),
        }

    mock_tokenizer.side_effect = mock_tokenize

    # Mock model to return embeddings based on input text
    def mock_forward(**kwargs):
        # Generate deterministic embeddings based on input hash
        input_ids = kwargs.get("input_ids", torch.zeros(1, 10))
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Use input hash to create reproducible but varied embeddings
        seed = int(input_ids.sum().item()) % 10000
        np.random.seed(seed)
        embeddings = np.random.randn(batch_size, seq_len, 768).astype(np.float32)

        result = MagicMock()
        result.last_hidden_state = torch.tensor(embeddings)
        return result

    mock_model.side_effect = mock_forward

    return mock_model, mock_tokenizer


# =============================================================================
# WBS 2.3.1: GraphCodeBERT Model Loading Tests
# =============================================================================


class TestGraphCodeBERTLoading:
    """Test GraphCodeBERT model loading."""

    def test_graphcodebert_validator_class_exists(self) -> None:
        """GraphCodeBERTValidator class should exist in src.models.graphcodebert_validator."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        assert GraphCodeBERTValidator is not None

    def test_graphcodebert_validator_initializes(self) -> None:
        """GraphCodeBERTValidator can be instantiated with registry."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        assert validator is not None

    def test_graphcodebert_validator_gets_model_from_registry(self) -> None:
        """Validator retrieves model from ModelRegistry, not direct load.

        Anti-Pattern #12 Prevention: Use cached model from registry.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        assert validator._model is mock_model
        assert validator._tokenizer is mock_tokenizer

    def test_graphcodebert_validator_raises_when_model_not_loaded(self) -> None:
        """Validator raises ModelNotReadyError if model not in registry."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry
        from src.core.exceptions import ModelNotReadyError

        fake_registry = FakeModelRegistry()
        # Don't register graphcodebert model

        with pytest.raises(ModelNotReadyError):
            GraphCodeBERTValidator(registry=fake_registry)


# =============================================================================
# WBS 2.3.2: Term Validation Tests
# =============================================================================


class TestGraphCodeBERTValidation:
    """Test term validation functionality."""

    def test_validation_result_model_exists(self) -> None:
        """ValidationResult Pydantic model exists."""
        from src.models.graphcodebert_validator import ValidationResult

        assert ValidationResult is not None

    def test_validation_result_has_required_fields(self) -> None:
        """ValidationResult has valid_terms, rejected_terms, and similarity_scores."""
        from src.models.graphcodebert_validator import ValidationResult

        result = ValidationResult(
            valid_terms=["chunking", "RAG"],
            rejected_terms=["split", "data"],
            rejection_reasons={"split": "too_generic", "data": "too_generic"},
            similarity_scores={"chunking": 0.85, "RAG": 0.92},
        )

        assert result.valid_terms == ["chunking", "RAG"]
        assert result.rejected_terms == ["split", "data"]
        assert "split" in result.rejection_reasons
        assert result.similarity_scores["chunking"] == 0.85

    def test_validate_terms_returns_validation_result(self) -> None:
        """validate_terms() returns ValidationResult with similarity scores."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator, ValidationResult
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)
        result = validator.validate_terms(
            terms=["chunking", "RAG", "split"],
            original_query="LLM document processing",
            domain="ai-ml",
        )

        assert isinstance(result, ValidationResult)
        # Valid terms should have similarity scores
        for term in result.valid_terms:
            assert term in result.similarity_scores

    def test_validate_terms_filters_generic_terms(self) -> None:
        """validate_terms() filters overly generic terms before model inference.

        WBS 2.3.2 Test: Filters 'split', 'data' as too generic (pre-filter).
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)
        result = validator.validate_terms(
            terms=["chunking", "RAG", "split", "data", "embedding"],
            original_query="LLM document processing",
            domain="ai-ml",
        )

        # Generic terms should be rejected (pre-filter)
        assert "split" in result.rejected_terms
        assert "data" in result.rejected_terms
        assert result.rejection_reasons["split"] == "too_generic"
        assert result.rejection_reasons["data"] == "too_generic"

    def test_validate_terms_provides_rejection_reasons(self) -> None:
        """validate_terms() provides reasons for rejections (generic or low similarity)."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)
        result = validator.validate_terms(
            terms=["chunking", "data"],
            original_query="LLM document processing",
            domain="ai-ml",
        )

        assert "data" in result.rejection_reasons
        # Reason should be "too_generic" for pre-filtered terms
        # or "low_similarity:X.XXX" for model-filtered terms
        reason = result.rejection_reasons["data"]
        assert reason == "too_generic" or reason.startswith("low_similarity:")

    def test_validate_terms_uses_semantic_similarity(self) -> None:
        """validate_terms() uses model embeddings for semantic filtering.

        Anti-Pattern #12: Model is actually invoked for embeddings.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)
        result = validator.validate_terms(
            terms=["transformer", "attention", "neural_network"],
            original_query="Deep learning transformer architecture",
            domain="ai-ml",
        )

        # Model should have been called for embedding generation
        assert mock_model.call_count > 0

        # Valid terms should have similarity scores
        for term in result.valid_terms:
            assert term in result.similarity_scores
            assert 0.0 <= result.similarity_scores[term] <= 1.0


# =============================================================================
# WBS 2.3.3: Domain Classification Tests
# =============================================================================


class TestGraphCodeBERTDomainClassification:
    """Test domain classification functionality using embeddings."""

    def test_classify_domain_method_exists(self) -> None:
        """classify_domain() method exists."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        assert hasattr(validator, "classify_domain")

    def test_classify_domain_returns_valid_domain(self) -> None:
        """classify_domain() returns a valid domain string."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        domain = validator.classify_domain("RAG pipeline with semantic search")

        # Should return one of the known domains or 'general'
        assert domain in ["ai-ml", "systems", "web", "data", "general"]

    def test_classify_domain_uses_model_embeddings(self) -> None:
        """classify_domain() uses model to compute embeddings.

        Anti-Pattern #12: Model is actually invoked, not just hardcoded lists.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        # Clear any cached embeddings
        validator._embedding_cache.clear()
        validator._domain_embeddings.clear()

        _ = validator.classify_domain("TCP socket connection pooling")

        # Model should have been called for embeddings
        assert mock_model.call_count > 0

    def test_classify_domain_caches_domain_embeddings(self) -> None:
        """classify_domain() caches domain reference embeddings.

        Anti-Pattern #12: Domain embeddings computed once and cached.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        # First call should compute domain embeddings
        _ = validator.classify_domain("Test query 1")
        first_call_count = mock_model.call_count

        # Second call should use cached domain embeddings
        _ = validator.classify_domain("Test query 2")
        second_call_count = mock_model.call_count

        # Only the new query embedding should be computed, not domain references again
        # (second call should add fewer model calls than first)
        new_calls = second_call_count - first_call_count
        assert new_calls <= 2  # At most query + 1 domain re-check


# =============================================================================
# WBS 2.3.4: Semantic Expansion Tests
# =============================================================================


class TestGraphCodeBERTSemanticExpansion:
    """Test semantic term expansion functionality using embeddings."""

    def test_expand_terms_method_exists(self) -> None:
        """expand_terms() method exists."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        assert hasattr(validator, "expand_terms")

    def test_expand_terms_returns_original_without_candidates(self) -> None:
        """expand_terms() returns original terms when no candidates provided.

        Without expansion candidates, no semantic search can be performed.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        expanded = validator.expand_terms(["RAG"], domain="ai-ml")

        # Should return just the original term
        assert expanded == ["RAG"]

    def test_expand_terms_uses_semantic_similarity(self) -> None:
        """expand_terms() uses embeddings to find similar terms from candidates.

        WBS 2.3.4: Semantic expansion via nearest neighbors.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        # Provide expansion candidates
        candidates = ["retrieval", "semantic_search", "augmented_generation", "tokenization"]

        expanded = validator.expand_terms(
            ["RAG"],
            domain="ai-ml",
            expansion_candidates=candidates,
        )

        # Should include original term
        assert "RAG" in expanded

        # Should have used model for embeddings
        assert mock_model.call_count > 0

    def test_expand_terms_respects_max_expansions(self) -> None:
        """expand_terms() respects max_expansions parameter."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        candidates = ["a", "b", "c", "d", "e", "f", "g", "h"]

        expanded = validator.expand_terms(
            ["test"],
            domain="ai-ml",
            max_expansions=2,
            expansion_candidates=candidates,
        )

        # Original + max 2 expansions = max 3 terms
        assert len(expanded) <= 3


# =============================================================================
# Embedding and Utility Method Tests
# =============================================================================


class TestGraphCodeBERTEmbeddings:
    """Test embedding generation and caching."""

    def test_get_embedding_caches_results(self) -> None:
        """_get_embedding() caches results to avoid recomputation.

        Anti-Pattern #12: Embeddings cached after first computation.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        # Clear cache
        validator._embedding_cache.clear()

        # First call
        _ = validator._get_embedding("test text")
        first_count = mock_model.call_count

        # Second call with same text (should use cache)
        _ = validator._get_embedding("test text")
        second_count = mock_model.call_count

        # Model should not be called again for cached text
        assert second_count == first_count

    def test_get_term_embeddings_utility(self) -> None:
        """get_term_embeddings() returns embeddings for multiple terms."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        embeddings = validator.get_term_embeddings(["term1", "term2", "term3"])

        assert len(embeddings) == 3
        assert "term1" in embeddings
        assert embeddings["term1"].shape == (768,)  # GraphCodeBERT dim

    def test_batch_similarity_utility(self) -> None:
        """batch_similarity() returns scores for multiple terms."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model, mock_tokenizer = _create_mock_model_with_embeddings()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        scores = validator.batch_similarity(
            terms=["RAG", "chunking", "embedding"],
            query="LLM document processing",
        )

        assert len(scores) == 3
        for term, score in scores.items():
            assert 0.0 <= score <= 1.0


# =============================================================================
# GraphCodeBERT Validator Protocol Tests
# =============================================================================


class TestGraphCodeBERTValidatorProtocol:
    """Test Protocol typing for model wrapper interfaces."""

    def test_validator_protocol_exists(self) -> None:
        """ValidatorProtocol exists for duck typing."""
        from src.models.protocols import ValidatorProtocol

        assert ValidatorProtocol is not None

    def test_graphcodebert_validator_implements_protocol(self) -> None:
        """GraphCodeBERTValidator implements ValidatorProtocol."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        # Duck typing verification
        assert hasattr(GraphCodeBERTValidator, "validate_terms")
        assert hasattr(GraphCodeBERTValidator, "classify_domain")
        assert hasattr(GraphCodeBERTValidator, "expand_terms")
