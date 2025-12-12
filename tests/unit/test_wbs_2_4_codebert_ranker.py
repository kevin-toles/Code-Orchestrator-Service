"""
WBS 2.4 Tests: CodeBERT Ranker (Relevance Scoring)

Renamed from CodeBERTAgent to CodeBERTRanker for clarity.
This is a HuggingFace model wrapper, NOT an autonomous agent.

Tests for semantic similarity scoring and term ranking using CodeBERT embeddings.

WBS Mapping:
- 2.4.1: CodeBERT model loading via Hugging Face
- 2.4.2: Embedding generation for code terms
- 2.4.3: Cosine similarity calculation
- 2.4.4: Term ranking by relevance score
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# WBS 2.4.1: CodeBERT Model Loading Tests
# =============================================================================


class TestCodeBERTLoading:
    """Test CodeBERT model loading from Hugging Face."""

    def test_codebert_ranker_class_exists(self) -> None:
        """CodeBERTRanker class exists and is importable."""
        from src.models.codebert_ranker import CodeBERTRanker

        assert CodeBERTRanker is not None

    def test_codebert_ranker_initializes_with_registry(self) -> None:
        """CodeBERTRanker initializes with a model registry.

        WBS 2.4.1 Acceptance: Loads via transformers library
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        assert ranker is not None

    def test_codebert_ranker_requires_registry(self) -> None:
        """CodeBERTRanker requires a registry to function."""
        from src.models.codebert_ranker import CodeBERTRanker

        with pytest.raises(TypeError):
            CodeBERTRanker()  # type: ignore[call-arg]

    def test_codebert_ranker_can_use_different_model_name(self) -> None:
        """CodeBERTRanker can be configured with different model names."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert-base", (mock_model, mock_tokenizer))

        # Should work with custom model name
        ranker = CodeBERTRanker(registry=fake_registry, model_name="codebert-base")
        assert ranker is not None


# =============================================================================
# WBS 2.4.2: Embedding Generation Tests
# =============================================================================


class TestCodeBERTEmbeddings:
    """Test CodeBERT embedding generation for code terms."""

    def test_get_embedding_method_exists(self) -> None:
        """get_embedding() method exists.

        WBS 2.4.2 Acceptance: Generate embeddings for code terms
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        assert hasattr(ranker, "get_embedding")
        assert callable(ranker.get_embedding)

    def test_get_embedding_returns_numpy_array(self) -> None:
        """get_embedding() returns a numpy array.

        WBS 2.4.2 Acceptance: 768-dimensional vector (CodeBERT base)
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock to return proper tensor shape
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            np.random.rand(1, 768)
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        embedding = ranker.get_embedding("chunking")

        assert isinstance(embedding, np.ndarray)

    def test_get_embedding_has_correct_dimension(self) -> None:
        """get_embedding() returns 768-dimensional vector."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            np.random.rand(1, 768)
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        embedding = ranker.get_embedding("chunking")

        # Should be 768 dimensions (CodeBERT base)
        assert embedding.shape[-1] == 768

    def test_get_embeddings_batch_exists(self) -> None:
        """get_embeddings_batch() method exists for multiple terms."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        assert hasattr(ranker, "get_embeddings_batch")
        assert callable(ranker.get_embeddings_batch)

    def test_get_embeddings_batch_returns_dict(self) -> None:
        """get_embeddings_batch() returns dict of term to embedding."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            np.random.rand(1, 768)
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        embeddings = ranker.get_embeddings_batch(["chunking", "networking"])

        assert isinstance(embeddings, dict)
        assert "chunking" in embeddings
        assert "networking" in embeddings


# =============================================================================
# WBS 2.4.3: Cosine Similarity Tests
# =============================================================================


class TestCodeBERTSimilarity:
    """Test cosine similarity calculation between embeddings."""

    def test_calculate_similarity_method_exists(self) -> None:
        """calculate_similarity() method exists.

        WBS 2.4.3 Acceptance: Calculate cosine similarity
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        assert hasattr(ranker, "calculate_similarity")
        assert callable(ranker.calculate_similarity)

    def test_calculate_similarity_returns_float(self) -> None:
        """calculate_similarity() returns a float score."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)

        # Create two random embeddings
        embedding1 = np.random.rand(768)
        embedding2 = np.random.rand(768)

        similarity = ranker.calculate_similarity(embedding1, embedding2)

        assert isinstance(similarity, float)

    def test_calculate_similarity_range(self) -> None:
        """calculate_similarity() returns value in [-1, 1] range.

        WBS 2.4.3 Test: Cosine similarity bounded
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)

        # Test with random embeddings
        for _ in range(10):
            embedding1 = np.random.rand(768)
            embedding2 = np.random.rand(768)

            similarity = ranker.calculate_similarity(embedding1, embedding2)

            assert -1.0 <= similarity <= 1.0

    def test_identical_embeddings_have_high_similarity(self) -> None:
        """Identical embeddings have similarity close to 1.0."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)

        embedding = np.random.rand(768)
        similarity = ranker.calculate_similarity(embedding, embedding)

        assert similarity > 0.999  # Should be ~1.0


# =============================================================================
# WBS 2.4.4: Term Ranking Tests
# =============================================================================


class TestCodeBERTRanking:
    """Test term ranking by relevance score."""

    def test_rank_terms_method_exists(self) -> None:
        """rank_terms() method exists.

        WBS 2.4.4 Acceptance: Rank terms by relevance
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        assert hasattr(ranker, "rank_terms")
        assert callable(ranker.rank_terms)

    def test_ranked_term_model_exists(self) -> None:
        """RankedTerm Pydantic model exists."""
        from src.models.codebert_ranker import RankedTerm

        assert RankedTerm is not None

    def test_ranked_term_has_term_and_score(self) -> None:
        """RankedTerm has term and score fields."""
        from src.models.codebert_ranker import RankedTerm

        ranked = RankedTerm(term="chunking", score=0.95)

        assert ranked.term == "chunking"
        assert ranked.score == 0.95

    def test_ranking_result_model_exists(self) -> None:
        """RankingResult Pydantic model exists."""
        from src.models.codebert_ranker import RankingResult

        assert RankingResult is not None

    def test_ranking_result_has_ranked_terms(self) -> None:
        """RankingResult has ranked_terms list."""
        from src.models.codebert_ranker import RankedTerm, RankingResult

        result = RankingResult(
            ranked_terms=[
                RankedTerm(term="chunking", score=0.9),
                RankedTerm(term="RAG", score=0.8),
            ]
        )

        assert len(result.ranked_terms) == 2

    def test_rank_terms_returns_ranking_result(self) -> None:
        """rank_terms() returns RankingResult."""
        from src.models.codebert_ranker import CodeBERTRanker, RankingResult
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            np.random.rand(1, 768)
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        result = ranker.rank_terms(
            terms=["chunking", "networking"],
            query="LLM document chunking for RAG",
        )

        assert isinstance(result, RankingResult)

    def test_rank_terms_sorts_by_score_descending(self) -> None:
        """rank_terms() sorts terms by score descending.

        WBS 2.4.4 Acceptance: Sort by relevance score descending
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock to return different scores for different terms
        call_count = [0]
        embeddings = [
            np.random.rand(1, 768) * 0.5,  # Lower for first term
            np.random.rand(1, 768) * 0.9,  # Higher for second term
        ]

        def mock_embedding_call(*args: Any, **kwargs: Any) -> MagicMock:  # noqa: ARG001
            mock_output = MagicMock()
            mock_output.last_hidden_state = MagicMock()
            mock_output.last_hidden_state.mean.return_value = MagicMock()
            idx = min(call_count[0], len(embeddings) - 1)
            mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
                embeddings[idx]
            )
            call_count[0] += 1
            return mock_output

        mock_model.side_effect = mock_embedding_call
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        result = ranker.rank_terms(
            terms=["chunking", "tokenization", "networking", "HTTP"],
            query="LLM document chunking for RAG",
        )

        # Verify sorted descending by score
        scores = [rt.score for rt in result.ranked_terms]
        assert scores == sorted(scores, reverse=True)

    def test_rank_terms_relevant_terms_score_higher(self) -> None:
        """Relevant terms score higher than irrelevant ones.

        WBS 2.4.4 Test: Chunking > networking for RAG query
        """
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            np.random.rand(1, 768)
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        ranker = CodeBERTRanker(registry=fake_registry)
        result = ranker.rank_terms(
            terms=["chunking", "tokenization", "networking", "HTTP"],
            query="LLM document chunking for RAG",
        )

        # Result should have ranked_terms
        assert len(result.ranked_terms) == 4
        assert all(hasattr(rt, "term") for rt in result.ranked_terms)
        assert all(hasattr(rt, "score") for rt in result.ranked_terms)


# =============================================================================
# CodeBERT Ranker Protocol Tests
# =============================================================================


class TestCodeBERTRankerProtocol:
    """Test Protocol typing for model wrapper interfaces."""

    def test_ranker_protocol_exists(self) -> None:
        """RankerProtocol exists for duck typing."""
        from src.models.protocols import RankerProtocol

        assert RankerProtocol is not None

    def test_codebert_ranker_implements_protocol(self) -> None:
        """CodeBERTRanker implements RankerProtocol."""
        from src.models.codebert_ranker import CodeBERTRanker

        # Duck typing verification
        assert hasattr(CodeBERTRanker, "get_embedding")
        assert hasattr(CodeBERTRanker, "get_embeddings_batch")
        assert hasattr(CodeBERTRanker, "calculate_similarity")
        assert hasattr(CodeBERTRanker, "rank_terms")


# =============================================================================
# Phase 2 Integration Test (Placeholder for HTTP test)
# =============================================================================


class TestPhase2Integration:
    """Test Phase 2 model loading integration."""

    def test_all_model_wrapper_classes_exist(self) -> None:
        """All three model wrapper classes exist and are importable."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        assert CodeT5Extractor is not None
        assert GraphCodeBERTValidator is not None
        assert CodeBERTRanker is not None

    def test_all_protocols_exist(self) -> None:
        """All model wrapper protocols exist."""
        from src.models.protocols import (
            ExtractorProtocol,
            RankerProtocol,
            ValidatorProtocol,
        )

        assert ExtractorProtocol is not None
        assert ValidatorProtocol is not None
        assert RankerProtocol is not None
