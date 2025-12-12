"""
Code-Orchestrator-Service - GraphCodeBERT Validator Tests

WBS 2.3: GraphCodeBERT Validator (Model Wrapper)
TDD Phase: RED - Write failing tests first

Tests for GraphCodeBERTValidator that validates and filters technical terms.

NOTE: These are HuggingFace model wrappers, NOT autonomous agents.
Autonomous agents (LangGraph workflows) live in the ai-agents service.

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- FakeClient for Testing (no real HuggingFace model in unit tests)
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached model from registry)
"""

from unittest.mock import MagicMock

import pytest

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
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
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
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
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
        """ValidationResult has valid_terms and rejected_terms."""
        from src.models.graphcodebert_validator import ValidationResult

        result = ValidationResult(
            valid_terms=["chunking", "RAG"],
            rejected_terms=["split", "data"],
            rejection_reasons={"split": "too_generic", "data": "too_generic"},
        )

        assert result.valid_terms == ["chunking", "RAG"]
        assert result.rejected_terms == ["split", "data"]
        assert "split" in result.rejection_reasons

    def test_validate_terms_returns_validation_result(self) -> None:
        """validate_terms() returns ValidationResult."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator, ValidationResult
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock model for encoding
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)
        result = validator.validate_terms(
            terms=["chunking", "RAG", "split"],
            original_query="LLM document processing",
            domain="ai-ml",
        )

        assert isinstance(result, ValidationResult)

    def test_validate_terms_filters_generic_terms(self) -> None:
        """validate_terms() filters overly generic terms.

        WBS 2.3.2 Test: Filters 'split', 'data' as too generic.
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)
        result = validator.validate_terms(
            terms=["chunking", "RAG", "split", "data", "embedding"],
            original_query="LLM document processing",
            domain="ai-ml",
        )

        # Generic terms should be rejected
        assert "split" in result.rejected_terms
        assert "data" in result.rejected_terms

        # Domain-specific terms should be valid
        assert "chunking" in result.valid_terms
        assert "RAG" in result.valid_terms
        assert "embedding" in result.valid_terms

    def test_validate_terms_provides_rejection_reasons(self) -> None:
        """validate_terms() provides reasons for rejections."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)
        result = validator.validate_terms(
            terms=["chunking", "data"],
            original_query="LLM document processing",
            domain="ai-ml",
        )

        assert "data" in result.rejection_reasons
        assert result.rejection_reasons["data"] in ["too_generic", "low_relevance", "out_of_domain"]


# =============================================================================
# WBS 2.3.3: Domain Classification Tests
# =============================================================================


class TestGraphCodeBERTDomainClassification:
    """Test domain classification functionality."""

    def test_classify_domain_method_exists(self) -> None:
        """classify_domain() method exists."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        assert hasattr(validator, "classify_domain")

    def test_classify_domain_returns_ai_ml(self) -> None:
        """classify_domain() identifies AI/ML domain terms."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        domain = validator.classify_domain("RAG pipeline with semantic search")

        assert domain in ["ai-ml", "llm", "nlp"]

    def test_classify_domain_returns_systems(self) -> None:
        """classify_domain() identifies systems/infrastructure domain."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        domain = validator.classify_domain("TCP socket connection pooling")

        assert domain in ["systems", "networking", "infrastructure"]

    def test_validate_terms_uses_domain_filter(self) -> None:
        """validate_terms() filters based on domain classification."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        # AI-ML query should filter out systems terms
        result = validator.validate_terms(
            terms=["chunking", "socket", "embedding", "TCP"],
            original_query="LLM document chunking",
            domain="ai-ml",
        )

        # Systems terms should be rejected for AI-ML domain
        assert "socket" in result.rejected_terms or "TCP" in result.rejected_terms


# =============================================================================
# WBS 2.3.4: Semantic Expansion Tests
# =============================================================================


class TestGraphCodeBERTSemanticExpansion:
    """Test semantic term expansion functionality."""

    def test_expand_terms_method_exists(self) -> None:
        """expand_terms() method exists."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        assert hasattr(validator, "expand_terms")

    def test_expand_terms_adds_related_terms(self) -> None:
        """expand_terms() adds semantically related terms.

        WBS 2.3.4: RAG → semantic_search, retrieval
        """
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        expanded = validator.expand_terms(["RAG"], domain="ai-ml")

        # Should include original term
        assert "RAG" in expanded

        # Should include related terms
        assert len(expanded) > 1  # At least added something

    def test_expand_terms_respects_max_expansions(self) -> None:
        """expand_terms() respects max_expansions parameter."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.return_value = MagicMock(last_hidden_state=MagicMock())
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("graphcodebert", (mock_model, mock_tokenizer))

        validator = GraphCodeBERTValidator(registry=fake_registry)

        expanded = validator.expand_terms(["RAG"], domain="ai-ml", max_expansions=2)

        # Original + max 2 expansions = max 3 terms per input
        assert len(expanded) <= 3


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
