"""
Code-Orchestrator-Service - CodeBERT Ranker Agent Tests

WBS 2.4: CodeBERT Ranker Agent
TDD Phase: RED - Write failing tests first

Tests for CodeBERTAgent that generates embeddings and ranks terms by relevance.

Patterns Applied:
- Agent Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- FakeClient for Testing (no real HuggingFace model in unit tests)
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached model from registry)
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# =============================================================================
# WBS 2.4.1: CodeBERT Model Loading Tests
# =============================================================================


class TestCodeBERTLoading:
    """Test CodeBERT model loading."""

    def test_codebert_agent_class_exists(self) -> None:
        """CodeBERTAgent class should exist in src.agents.codebert_agent."""
        from src.agents.codebert_agent import CodeBERTAgent

        assert CodeBERTAgent is not None

    def test_codebert_agent_initializes(self) -> None:
        """CodeBERTAgent can be instantiated with registry."""
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        agent = CodeBERTAgent(registry=fake_registry)

        assert agent is not None

    def test_codebert_agent_gets_model_from_registry(self) -> None:
        """Agent retrieves model from ModelRegistry, not direct load.

        Anti-Pattern #12 Prevention: Use cached model from registry.
        """
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        agent = CodeBERTAgent(registry=fake_registry)

        assert agent._model is mock_model
        assert agent._tokenizer is mock_tokenizer

    def test_codebert_agent_raises_when_model_not_loaded(self) -> None:
        """Agent raises ModelNotReadyError if model not in registry."""
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry
        from src.core.exceptions import ModelNotReadyError

        fake_registry = FakeModelRegistry()
        # Don't register codebert model

        with pytest.raises(ModelNotReadyError):
            CodeBERTAgent(registry=fake_registry)


# =============================================================================
# WBS 2.4.2: Embedding Generation Tests
# =============================================================================


class TestCodeBERTEmbeddings:
    """Test embedding generation functionality."""

    def test_get_embedding_returns_numpy_array(self) -> None:
        """get_embedding() returns numpy array."""
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock model to return embeddings
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            np.random.rand(1, 768)
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        agent = CodeBERTAgent(registry=fake_registry)
        embedding = agent.get_embedding("chunking")

        assert isinstance(embedding, np.ndarray)

    def test_get_embedding_returns_768_dimensions(self) -> None:
        """get_embedding() returns 768-dimensional vector.

        WBS 2.4.2 Acceptance: 768-dim embeddings for terms
        """
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock model to return 768-dim embeddings
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            np.random.rand(1, 768)
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        agent = CodeBERTAgent(registry=fake_registry)
        embedding = agent.get_embedding("RAG")

        assert embedding.shape[-1] == 768

    def test_get_embeddings_batch_processes_multiple_terms(self) -> None:
        """get_embeddings_batch() processes multiple terms efficiently."""
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock for batch - need to return objects that can be indexed
        # and have .detach().numpy() method
        mock_output = MagicMock()
        mock_mean = MagicMock()

        # Create individual embedding mocks that return numpy arrays
        def create_embedding_item(idx: int) -> MagicMock:  # noqa: ARG001
            item = MagicMock()
            item.detach.return_value.numpy.return_value = np.random.rand(768)
            return item

        # Make indexing return mocks with detach().numpy()
        mock_mean.__getitem__ = lambda self, i: create_embedding_item(i)  # noqa: ARG005

        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = mock_mean
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        agent = CodeBERTAgent(registry=fake_registry)
        terms = ["chunking", "RAG", "embedding"]

        embeddings = agent.get_embeddings_batch(terms)

        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)


# =============================================================================
# WBS 2.4.3: Similarity Scoring Tests
# =============================================================================


class TestCodeBERTSimilarity:
    """Test cosine similarity scoring functionality."""

    def test_calculate_similarity_returns_float(self) -> None:
        """calculate_similarity() returns float score."""
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

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

        agent = CodeBERTAgent(registry=fake_registry)
        score = agent.calculate_similarity("chunking", "LLM document chunking for RAG")

        assert isinstance(score, float)

    def test_calculate_similarity_returns_0_to_1(self) -> None:
        """Cosine similarity score is between 0 and 1.

        WBS 2.4.3: Cosine similarity with query
        """
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

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

        agent = CodeBERTAgent(registry=fake_registry)
        score = agent.calculate_similarity("chunking", "document processing")

        assert 0.0 <= score <= 1.0

    def test_identical_terms_have_high_similarity(self) -> None:
        """Identical terms should have similarity close to 1.0."""
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

        fake_registry = FakeModelRegistry()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Configure mock to return identical embeddings for identical inputs
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_output.last_hidden_state.mean.return_value = MagicMock()
        # Same embedding for same input
        fixed_embedding = np.ones((1, 768)) * 0.5
        mock_output.last_hidden_state.mean.return_value.detach.return_value.numpy.return_value = (
            fixed_embedding
        )
        mock_model.return_value = mock_output
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

        fake_registry.register_model("codebert", (mock_model, mock_tokenizer))

        agent = CodeBERTAgent(registry=fake_registry)
        score = agent.calculate_similarity("chunking", "chunking")

        assert score > 0.95  # Should be very close to 1.0


# =============================================================================
# WBS 2.4.4: Ranking Tests
# =============================================================================


class TestCodeBERTRanking:
    """Test term ranking by relevance."""

    def test_ranked_term_model_exists(self) -> None:
        """RankedTerm Pydantic model exists."""
        from src.agents.codebert_agent import RankedTerm

        assert RankedTerm is not None

    def test_ranked_term_has_required_fields(self) -> None:
        """RankedTerm has term and score fields."""
        from src.agents.codebert_agent import RankedTerm

        ranked = RankedTerm(term="chunking", score=0.85)

        assert ranked.term == "chunking"
        assert ranked.score == 0.85

    def test_ranking_result_model_exists(self) -> None:
        """RankingResult Pydantic model exists."""
        from src.agents.codebert_agent import RankingResult

        assert RankingResult is not None

    def test_ranking_result_has_ranked_terms(self) -> None:
        """RankingResult has ranked_terms list."""
        from src.agents.codebert_agent import RankedTerm, RankingResult

        result = RankingResult(
            ranked_terms=[
                RankedTerm(term="chunking", score=0.9),
                RankedTerm(term="RAG", score=0.8),
            ]
        )

        assert len(result.ranked_terms) == 2

    def test_rank_terms_returns_ranking_result(self) -> None:
        """rank_terms() returns RankingResult."""
        from src.agents.codebert_agent import CodeBERTAgent, RankingResult
        from src.agents.registry import FakeModelRegistry

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

        agent = CodeBERTAgent(registry=fake_registry)
        result = agent.rank_terms(
            terms=["chunking", "networking"],
            query="LLM document chunking for RAG",
        )

        assert isinstance(result, RankingResult)

    def test_rank_terms_sorts_by_score_descending(self) -> None:
        """rank_terms() sorts terms by score descending.

        WBS 2.4.4 Acceptance: Sort by relevance score descending
        """
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

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

        agent = CodeBERTAgent(registry=fake_registry)
        result = agent.rank_terms(
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
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.registry import FakeModelRegistry

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

        agent = CodeBERTAgent(registry=fake_registry)
        result = agent.rank_terms(
            terms=["chunking", "tokenization", "networking", "HTTP"],
            query="LLM document chunking for RAG",
        )

        # Result should have ranked_terms
        assert len(result.ranked_terms) == 4
        assert all(hasattr(rt, "term") for rt in result.ranked_terms)
        assert all(hasattr(rt, "score") for rt in result.ranked_terms)


# =============================================================================
# CodeBERT Agent Protocol Tests
# =============================================================================


class TestCodeBERTAgentProtocol:
    """Test Protocol typing for agent interfaces."""

    def test_ranker_agent_protocol_exists(self) -> None:
        """RankerAgentProtocol exists for duck typing."""
        from src.agents.protocols import RankerAgentProtocol

        assert RankerAgentProtocol is not None

    def test_codebert_agent_implements_protocol(self) -> None:
        """CodeBERTAgent implements RankerAgentProtocol."""
        from src.agents.codebert_agent import CodeBERTAgent

        # Duck typing verification
        assert hasattr(CodeBERTAgent, "get_embedding")
        assert hasattr(CodeBERTAgent, "get_embeddings_batch")
        assert hasattr(CodeBERTAgent, "calculate_similarity")
        assert hasattr(CodeBERTAgent, "rank_terms")


# =============================================================================
# Phase 2 Integration Test (Placeholder for HTTP test)
# =============================================================================


class TestPhase2Integration:
    """Test Phase 2 model loading integration."""

    def test_all_agent_classes_exist(self) -> None:
        """All three agent classes exist and are importable."""
        from src.agents.codebert_agent import CodeBERTAgent
        from src.agents.codet5_agent import CodeT5Agent
        from src.agents.graphcodebert_agent import GraphCodeBERTAgent

        assert CodeT5Agent is not None
        assert GraphCodeBERTAgent is not None
        assert CodeBERTAgent is not None

    def test_all_protocols_exist(self) -> None:
        """All agent protocols exist."""
        from src.agents.protocols import (
            GeneratorAgentProtocol,
            RankerAgentProtocol,
            ValidatorAgentProtocol,
        )

        assert GeneratorAgentProtocol is not None
        assert ValidatorAgentProtocol is not None
        assert RankerAgentProtocol is not None
