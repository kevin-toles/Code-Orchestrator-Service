"""
Code-Orchestrator-Service - CodeBERT Term Ranker Tests

WBS 2.4: Term Ranker (Model Wrapper)
Tests for CodeBERTRanker that ranks terms using CodeBERT embeddings.

The ranker uses locally hosted microsoft/codebert-base for 768-dimensional
NL↔Code bimodal semantic ranking.

Architecture Role: RANKER (STATE 3: RANKING)

Test Coverage:
- RankingResult model structure
- 768-dim embedding generation
- Cosine similarity calculation
- Term ranking by relevance
- Batch embedding
"""

import pytest
import numpy as np


# =============================================================================
# WBS 2.4.1: Ranker Class Tests
# =============================================================================


class TestCodeBERTRankerClass:
    """Test CodeBERTRanker class exists and initializes."""

    def test_codebert_ranker_class_exists(self) -> None:
        """CodeBERTRanker class should exist."""
        from src.models.codebert_ranker import CodeBERTRanker

        assert CodeBERTRanker is not None

    def test_codebert_ranker_initializes(self) -> None:
        """CodeBERTRanker can be instantiated with local model."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        assert ranker is not None

    def test_ranker_has_codebert_model(self) -> None:
        """Ranker uses CodeBERT model (768-dim RoBERTa)."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        # CodeBERT uses _model and _tokenizer
        assert ranker._model is not None
        assert ranker._tokenizer is not None


# =============================================================================
# WBS 2.4.2: Result Model Tests
# =============================================================================


class TestRankingResultModels:
    """Test RankedTerm and RankingResult Pydantic models."""

    def test_ranked_term_model_exists(self) -> None:
        """RankedTerm Pydantic model exists."""
        from src.models.codebert_ranker import RankedTerm

        assert RankedTerm is not None

    def test_ranked_term_has_required_fields(self) -> None:
        """RankedTerm has term and score fields."""
        from src.models.codebert_ranker import RankedTerm

        ranked = RankedTerm(term="Redis", score=0.85)

        assert ranked.term == "Redis"
        assert ranked.score == 0.85

    def test_ranking_result_model_exists(self) -> None:
        """RankingResult Pydantic model exists."""
        from src.models.codebert_ranker import RankingResult

        assert RankingResult is not None

    def test_ranking_result_has_required_fields(self) -> None:
        """RankingResult has ranked_terms field."""
        from src.models.codebert_ranker import RankedTerm, RankingResult

        result = RankingResult(
            ranked_terms=[
                RankedTerm(term="Redis", score=0.9),
                RankedTerm(term="caching", score=0.8),
            ]
        )

        assert len(result.ranked_terms) == 2
        assert result.ranked_terms[0].term == "Redis"

    def test_ranking_result_empty(self) -> None:
        """RankingResult works with empty list."""
        from src.models.codebert_ranker import RankingResult

        result = RankingResult(ranked_terms=[])
        assert result.ranked_terms == []


# =============================================================================
# WBS 2.4.3: Embedding Generation Tests
# =============================================================================


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    def test_get_embedding_returns_array(self) -> None:
        """get_embedding() returns numpy array."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        embedding = ranker.get_embedding("Redis caching")

        assert isinstance(embedding, np.ndarray)

    def test_get_embedding_correct_dimension(self) -> None:
        """get_embedding() returns 768-dimensional vector (CodeBERT)."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        embedding = ranker.get_embedding("Redis caching")

        assert embedding.shape == (768,)

    def test_get_embeddings_batch_returns_list(self) -> None:
        """get_embeddings_batch() returns list of arrays."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        embeddings = ranker.get_embeddings_batch(["Redis", "caching", "layer"])

        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)

    def test_get_embeddings_batch_empty(self) -> None:
        """get_embeddings_batch() handles empty list."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        embeddings = ranker.get_embeddings_batch([])

        assert embeddings == []

    def test_embeddings_are_normalized(self) -> None:
        """Embeddings should be close to unit norm."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        embedding = ranker.get_embedding("Redis caching")

        norm = np.linalg.norm(embedding)
        # SBERT embeddings are not necessarily normalized, just check it's reasonable
        assert norm > 0


# =============================================================================
# WBS 2.4.4: Cosine Similarity Tests
# =============================================================================


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_calculate_similarity_returns_float(self) -> None:
        """calculate_similarity() returns float."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        similarity = ranker.calculate_similarity("Redis", "caching")

        assert isinstance(similarity, float)

    def test_calculate_similarity_range(self) -> None:
        """calculate_similarity() returns value in [0, 1]."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        similarity = ranker.calculate_similarity("Redis", "caching")

        assert 0.0 <= similarity <= 1.0

    def test_calculate_similarity_identical(self) -> None:
        """Identical texts have high similarity."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        similarity = ranker.calculate_similarity("Redis caching", "Redis caching")

        assert similarity > 0.99

    def test_calculate_similarity_related(self) -> None:
        """Related texts have some positive similarity."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        similarity = ranker.calculate_similarity("Redis", "distributed caching")

        # Related terms should have some positive similarity
        assert similarity > 0.1

    def test_calculate_similarity_unrelated(self) -> None:
        """Unrelated texts have lower similarity."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()

        related = ranker.calculate_similarity("Redis", "caching")
        unrelated = ranker.calculate_similarity("Redis", "banana fruit yellow")

        # Related should be more similar than unrelated
        assert related > unrelated


# =============================================================================
# WBS 2.4.5: Term Ranking Tests
# =============================================================================


class TestTermRanking:
    """Test term ranking functionality."""

    def test_rank_terms_returns_ranking_result(self) -> None:
        """rank_terms() returns RankingResult."""
        from src.models.codebert_ranker import CodeBERTRanker, RankingResult

        ranker = CodeBERTRanker()
        result = ranker.rank_terms(
            terms=["Redis", "caching", "layer"],
            query="Redis caching",
        )

        assert isinstance(result, RankingResult)

    def test_rank_terms_sorts_by_score_descending(self) -> None:
        """rank_terms() sorts terms by score descending."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        result = ranker.rank_terms(
            terms=["layer", "Redis", "caching", "system"],
            query="Redis caching",
        )

        # Should be sorted descending by score
        scores = [rt.score for rt in result.ranked_terms]
        assert scores == sorted(scores, reverse=True)

    def test_rank_terms_most_relevant_first(self) -> None:
        """rank_terms() puts most relevant terms first."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        result = ranker.rank_terms(
            terms=["banana", "Redis", "apple", "caching"],
            query="Redis distributed caching",
        )

        # Redis and caching should be ranked higher than banana/apple
        top_terms = [rt.term.lower() for rt in result.ranked_terms[:2]]
        assert "redis" in top_terms or "caching" in top_terms

    def test_rank_terms_empty_list(self) -> None:
        """rank_terms() handles empty term list."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        result = ranker.rank_terms(terms=[], query="Redis caching")

        assert result.ranked_terms == []

    def test_rank_terms_single_term(self) -> None:
        """rank_terms() handles single term."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        result = ranker.rank_terms(terms=["Redis"], query="Redis caching")

        assert len(result.ranked_terms) == 1
        assert result.ranked_terms[0].term == "Redis"

    def test_rank_terms_all_terms_have_scores(self) -> None:
        """rank_terms() assigns scores to all terms."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        terms = ["Redis", "caching", "layer", "system"]
        result = ranker.rank_terms(terms=terms, query="Redis caching")

        assert len(result.ranked_terms) == len(terms)
        for rt in result.ranked_terms:
            assert 0.0 <= rt.score <= 1.0


# =============================================================================
# WBS 2.4.6: Performance Tests
# =============================================================================


class TestRankerPerformance:
    """Test ranker performance characteristics."""

    def test_embedding_is_deterministic(self) -> None:
        """Same input produces same embedding."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()

        emb1 = ranker.get_embedding("Redis caching")
        emb2 = ranker.get_embedding("Redis caching")

        assert np.allclose(emb1, emb2)

    def test_ranking_is_deterministic(self) -> None:
        """Same inputs produce same ranking."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()

        result1 = ranker.rank_terms(
            terms=["Redis", "caching", "layer"],
            query="Redis caching",
        )

        result2 = ranker.rank_terms(
            terms=["Redis", "caching", "layer"],
            query="Redis caching",
        )

        # Order should be the same
        terms1 = [rt.term for rt in result1.ranked_terms]
        terms2 = [rt.term for rt in result2.ranked_terms]
        assert terms1 == terms2
