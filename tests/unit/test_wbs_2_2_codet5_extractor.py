"""
Code-Orchestrator-Service - CodeT5+ Keyword Extractor Tests

WBS 2.2: CodeT5+ Extractor (Model Wrapper)
TDD Phase: RED - Write failing tests first

Tests for CodeT5Extractor that extracts technical terms from text.

NOTE: These are HuggingFace model wrappers, NOT autonomous agents.
Autonomous agents (LangGraph workflows) live in the ai-agents service.

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
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

    def test_codet5_extractor_class_exists(self) -> None:
        """CodeT5Extractor class should exist in src.models.codet5_extractor."""
        from src.models.codet5_extractor import CodeT5Extractor

        assert CodeT5Extractor is not None

    def test_codet5_extractor_initializes(self) -> None:
        """CodeT5Extractor can be instantiated with registry."""
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)

        assert extractor is not None

    def test_codet5_extractor_gets_model_from_registry(self) -> None:
        """Extractor retrieves model from ModelRegistry, not direct load.

        Anti-Pattern #12 Prevention: Use cached model from registry.
        """
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)

        assert extractor._model is mock_model
        assert extractor._tokenizer is mock_tokenizer

    def test_codet5_extractor_raises_when_model_not_loaded(self) -> None:
        """Extractor raises ModelNotReadyError if model not in registry."""
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry
        from src.core.exceptions import ModelNotReadyError

        fake_registry = FakeModelRegistry()
        # Don't register codet5 model

        with pytest.raises(ModelNotReadyError):
            CodeT5Extractor(registry=fake_registry)


# =============================================================================
# WBS 2.2.2: Term Extraction Tests
# =============================================================================


class TestCodeT5TermExtraction:
    """Test term extraction functionality."""

    def test_extraction_result_model_exists(self) -> None:
        """ExtractionResult Pydantic model exists."""
        from src.models.codet5_extractor import ExtractionResult

        assert ExtractionResult is not None

    def test_extraction_result_has_required_fields(self) -> None:
        """ExtractionResult has primary_terms and related_terms."""
        from src.models.codet5_extractor import ExtractionResult

        result = ExtractionResult(
            primary_terms=["chunking", "RAG"],
            related_terms=["overlap", "embedding"],
        )

        assert result.primary_terms == ["chunking", "RAG"]
        assert result.related_terms == ["overlap", "embedding"]

    def test_extract_terms_returns_extraction_result(self) -> None:
        """extract_terms() returns ExtractionResult."""
        from src.models.codet5_extractor import CodeT5Extractor, ExtractionResult
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock model to return generated tokens
        mock_model.generate.return_value = MagicMock()
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.decode.return_value = "chunking, RAG, overlap"

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)
        result = extractor.extract_terms("This chapter covers document chunking for RAG")

        assert isinstance(result, ExtractionResult)

    def test_extract_terms_finds_primary_terms(self) -> None:
        """extract_terms() identifies primary technical terms.

        WBS 2.2.2 Acceptance: primary_terms[] populated
        """
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock to return specific terms
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: chunking, RAG; related: overlap"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)
        result = extractor.extract_terms(
            "This chapter covers multi-stage document chunking with overlap for RAG pipelines"
        )

        assert len(result.primary_terms) >= 2
        # Terms should be semantically relevant to the input

    def test_extract_terms_finds_related_terms(self) -> None:
        """extract_terms() identifies related/supporting terms.

        WBS 2.2.2 Acceptance: related_terms[] populated
        """
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: chunking; related: overlap, embedding, tokens"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)
        result = extractor.extract_terms(
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
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: term1; related: term2"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)
        texts = [
            "Chapter 1: Document chunking",
            "Chapter 2: RAG pipelines",
            "Chapter 3: Embedding models",
        ]

        results = extractor.extract_terms_batch(texts)

        assert len(results) == 3

    def test_extract_terms_batch_returns_list_of_results(self) -> None:
        """Batch processing returns list of ExtractionResult."""
        from src.models.codet5_extractor import CodeT5Extractor, ExtractionResult
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: term1; related: term2"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)
        texts = ["Text 1", "Text 2"]

        results = extractor.extract_terms_batch(texts)

        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_extract_terms_batch_handles_empty_list(self) -> None:
        """Batch processing handles empty input list."""
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)

        results = extractor.extract_terms_batch([])

        assert results == []


# =============================================================================
# WBS 2.2.4: Inference Timeout Tests
# =============================================================================


class TestCodeT5Timeout:
    """Test inference timeout handling."""

    def test_extract_terms_has_timeout_parameter(self) -> None:
        """extract_terms() accepts timeout_seconds parameter."""
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = "primary: term1"
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        fake_registry.register_model("codet5", (mock_model, mock_tokenizer))

        extractor = CodeT5Extractor(registry=fake_registry)

        # Should not raise
        result = extractor.extract_terms("Test text", timeout_seconds=30)

        assert result is not None

    def test_extract_terms_default_timeout_is_30s(self) -> None:
        """Default timeout is 30 seconds per WBS spec."""
        from src.models.codet5_extractor import CodeT5Extractor

        assert CodeT5Extractor.DEFAULT_TIMEOUT_SECONDS == 30

    def test_extract_terms_raises_on_timeout(self) -> None:
        """extract_terms() raises TimeoutError when inference exceeds limit."""
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.registry import FakeModelRegistry

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

        extractor = CodeT5Extractor(registry=fake_registry)

        with pytest.raises(TimeoutError):
            extractor.extract_terms("Test text", timeout_seconds=0.1)


# =============================================================================
# CodeT5Extractor Protocol Tests
# =============================================================================


class TestCodeT5ExtractorProtocol:
    """Test Protocol typing for model wrapper interfaces."""

    def test_extractor_protocol_exists(self) -> None:
        """ExtractorProtocol exists for duck typing."""
        from src.models.protocols import ExtractorProtocol

        assert ExtractorProtocol is not None

    def test_codet5_extractor_implements_protocol(self) -> None:
        """CodeT5Extractor implements ExtractorProtocol."""
        from src.models.codet5_extractor import CodeT5Extractor

        # Duck typing verification
        assert hasattr(CodeT5Extractor, "extract_terms")
        assert hasattr(CodeT5Extractor, "extract_terms_batch")
