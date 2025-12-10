"""
Code-Orchestrator-Service - CodeT5+ Generator Agent Tests

WBS 2.2: CodeT5+ Generator Agent
TDD Phase: RED - Write failing tests first

Tests for CodeT5Agent that extracts technical terms from text.

Patterns Applied:
- Agent Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- FakeClient for Testing (no real HuggingFace model in unit tests)
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached model from registry)
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

# =============================================================================
# WBS 2.2.1: CodeT5+ Model Loading Tests
# =============================================================================


class TestCodeT5Loading:
    """Test CodeT5+ model loading."""

    def test_codet5_agent_class_exists(self) -> None:
        """CodeT5Agent class should exist in src.agents.codet5_agent."""
        from src.agents.codet5_agent import CodeT5Agent

        assert CodeT5Agent is not None

    def test_codet5_agent_initializes(self) -> None:
        """CodeT5Agent can be instantiated with registry."""
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)

        assert agent is not None

    def test_codet5_agent_gets_model_from_registry(self) -> None:
        """Agent retrieves model from ModelRegistry, not direct load.

        Anti-Pattern #12 Prevention: Use cached model from registry.
        """
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)

        assert agent._model is mock_model
        assert agent._tokenizer is mock_tokenizer

    def test_codet5_agent_raises_when_model_not_loaded(self) -> None:
        """Agent raises ModelNotReadyError if model not in registry."""
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry
        from src.core.exceptions import ModelNotReadyError

        fake_registry = FakeModelRegistry()
        # Don't register codet5 model

        with pytest.raises(ModelNotReadyError):
            CodeT5Agent(registry=fake_registry)


# =============================================================================
# WBS 2.2.2: Term Extraction Tests
# =============================================================================


class TestCodeT5TermExtraction:
    """Test term extraction functionality."""

    def test_extraction_result_model_exists(self) -> None:
        """ExtractionResult Pydantic model exists."""
        from src.agents.codet5_agent import ExtractionResult

        assert ExtractionResult is not None

    def test_extraction_result_has_required_fields(self) -> None:
        """ExtractionResult has primary_terms and related_terms."""
        from src.agents.codet5_agent import ExtractionResult

        result = ExtractionResult(
            primary_terms=["chunking", "RAG"],
            related_terms=["overlap", "embedding"],
        )

        assert result.primary_terms == ["chunking", "RAG"]
        assert result.related_terms == ["overlap", "embedding"]

    def test_extract_terms_returns_extraction_result(self) -> None:
        """extract_terms() returns ExtractionResult."""
        from src.agents.codet5_agent import CodeT5Agent, ExtractionResult
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock model to return generated tokens
        mock_model.generate.return_value = MagicMock()
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.decode.return_value = "chunking, RAG, overlap"

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)
        result = agent.extract_terms("This chapter covers document chunking for RAG")

        assert isinstance(result, ExtractionResult)

    def test_extract_terms_finds_primary_terms(self) -> None:
        """extract_terms() identifies primary technical terms.

        WBS 2.2.2 Acceptance: primary_terms[] populated
        """
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock to return specific terms
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: chunking, RAG; related: overlap"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)
        result = agent.extract_terms(
            "This chapter covers multi-stage document chunking with overlap for RAG pipelines"
        )

        assert len(result.primary_terms) >= 2
        # Terms should be semantically relevant to the input

    def test_extract_terms_finds_related_terms(self) -> None:
        """extract_terms() identifies related/supporting terms.

        WBS 2.2.2 Acceptance: related_terms[] populated
        """
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: chunking; related: overlap, embedding, tokens"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)
        result = agent.extract_terms(
            "This chapter covers multi-stage document chunking with overlap for RAG pipelines"
        )

        assert len(result.related_terms) >= 1


# =============================================================================
# WBS 2.2.3: Batch Processing Tests
# =============================================================================


class TestCodeT5BatchProcessing:
    """Test batch processing for multiple chapters."""

    def test_extract_terms_batch_accepts_list(self) -> None:
        """extract_terms_batch() accepts list of texts."""
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: term1; related: term2"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)
        texts = [
            "Chapter 1: Document chunking",
            "Chapter 2: RAG pipelines",
            "Chapter 3: Embedding models",
        ]

        results = agent.extract_terms_batch(texts)

        assert len(results) == 3

    def test_extract_terms_batch_returns_list_of_results(self) -> None:
        """Batch processing returns list of ExtractionResult."""
        from src.agents.codet5_agent import CodeT5Agent, ExtractionResult
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: term1; related: term2"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)
        texts = ["Text 1", "Text 2"]

        results = agent.extract_terms_batch(texts)

        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_extract_terms_batch_handles_empty_list(self) -> None:
        """Batch processing handles empty input list."""
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)

        results = agent.extract_terms_batch([])

        assert results == []


# =============================================================================
# WBS 2.2.4: Inference Timeout Tests
# =============================================================================


class TestCodeT5Timeout:
    """Test inference timeout handling."""

    def test_extract_terms_has_timeout_parameter(self) -> None:
        """extract_terms() accepts timeout_seconds parameter."""
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: term1"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)

        # Should not raise
        result = agent.extract_terms("Test text", timeout_seconds=30)

        assert result is not None

    def test_extract_terms_default_timeout_is_30s(self) -> None:
        """Default timeout is 30 seconds per WBS spec."""
        from src.agents.codet5_agent import CodeT5Agent

        assert CodeT5Agent.DEFAULT_TIMEOUT_SECONDS == 30

    def test_extract_terms_raises_on_timeout(self) -> None:
        """extract_terms() raises TimeoutError when inference exceeds limit."""
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Simulate slow model
        def slow_generate(*args: Any, **kwargs: Any) -> list[list[int]]:  # noqa: ARG001
            import time
            time.sleep(2)  # 2 second delay
            return [[1, 2, 3]]

        mock_model.generate.side_effect = slow_generate
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        agent = CodeT5Agent(registry=fake_registry)

        with pytest.raises(TimeoutError):
            agent.extract_terms("Test text", timeout_seconds=0.1)


# =============================================================================
# CodeT5Agent Protocol Tests
# =============================================================================


class TestCodeT5AgentProtocol:
    """Test Protocol typing for agent interfaces."""

    def test_generator_agent_protocol_exists(self) -> None:
        """GeneratorAgentProtocol exists for duck typing."""
        from src.agents.protocols import GeneratorAgentProtocol

        assert GeneratorAgentProtocol is not None

    def test_codet5_agent_implements_protocol(self) -> None:
        """CodeT5Agent implements GeneratorAgentProtocol."""
        from src.agents.codet5_agent import CodeT5Agent

        # Duck typing verification
        assert hasattr(CodeT5Agent, "extract_terms")
        assert hasattr(CodeT5Agent, "extract_terms_batch")
