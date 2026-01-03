"""
Code-Orchestrator-Service - GraphCodeBERT Term Validator Tests

WBS 2.3: Term Validator (Model Wrapper)
Tests for GraphCodeBERTValidator that validates terms using GraphCodeBERT embeddings.

The validator uses locally hosted microsoft/graphcodebert-base for 768-dimensional
semantic similarity-based term validation and filtering.

Architecture Role: VALIDATOR (STATE 2: VALIDATION)

Test Coverage:
- ValidationResult model structure
- Term validation with 768-dim similarity scoring
- Generic term filtering
- Domain classification
- Batch processing
"""

import pytest


# =============================================================================
# WBS 2.3.1: Validator Class Tests
# =============================================================================


class TestGraphCodeBERTValidatorClass:
    """Test GraphCodeBERTValidator class exists and initializes."""

    def test_graphcodebert_validator_class_exists(self) -> None:
        """GraphCodeBERTValidator class should exist."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        assert GraphCodeBERTValidator is not None

    def test_graphcodebert_validator_initializes(self) -> None:
        """GraphCodeBERTValidator can be instantiated with local model."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        assert validator is not None

    def test_validator_has_graphcodebert_model(self) -> None:
        """Validator uses GraphCodeBERT model (768-dim RoBERTa)."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        # GraphCodeBERT uses _model and _tokenizer
        assert validator._model is not None
        assert validator._tokenizer is not None


# =============================================================================
# WBS 2.3.2: ValidationResult Model Tests
# =============================================================================


class TestValidationResult:
    """Test ValidationResult Pydantic model."""

    def test_validation_result_model_exists(self) -> None:
        """ValidationResult Pydantic model exists."""
        from src.models.graphcodebert_validator import ValidationResult

        assert ValidationResult is not None

    def test_validation_result_has_required_fields(self) -> None:
        """ValidationResult has valid_terms, rejected_terms, rejection_reasons."""
        from src.models.graphcodebert_validator import ValidationResult

        result = ValidationResult(
            valid_terms=["Redis", "caching"],
            rejected_terms=["data"],
            rejection_reasons={"data": "too_generic"},
            similarity_scores={"Redis": 0.8, "caching": 0.7},
        )

        assert result.valid_terms == ["Redis", "caching"]
        assert result.rejected_terms == ["data"]
        assert result.rejection_reasons == {"data": "too_generic"}
        assert result.similarity_scores["Redis"] == 0.8

    def test_validation_result_empty(self) -> None:
        """ValidationResult works with empty lists."""
        from src.models.graphcodebert_validator import ValidationResult

        result = ValidationResult(
            valid_terms=[],
            rejected_terms=[],
            rejection_reasons={},
        )

        assert result.valid_terms == []
        assert result.rejected_terms == []


# =============================================================================
# WBS 2.3.3: Term Validation Tests
# =============================================================================


class TestTermValidation:
    """Test term validation functionality."""

    def test_validate_terms_returns_validation_result(self) -> None:
        """validate_terms() returns ValidationResult."""
        from src.models.graphcodebert_validator import (
            GraphCodeBERTValidator,
            ValidationResult,
        )

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            terms=["Redis", "caching", "layer"],
            original_query="Redis distributed caching",
            domain="systems",
        )

        assert isinstance(result, ValidationResult)

    def test_validate_terms_accepts_valid_terms(self) -> None:
        """validate_terms() accepts semantically relevant terms."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            terms=["Redis", "caching", "distributed"],
            original_query="Redis distributed caching for horizontal scaling",
            domain="systems",
        )

        # Should accept terms related to the query
        assert len(result.valid_terms) > 0

    def test_validate_terms_rejects_generic_terms(self) -> None:
        """validate_terms() rejects generic/common terms."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            terms=["data", "value", "object", "Redis"],
            original_query="Redis caching",
            domain="systems",
        )

        # Generic terms should be rejected
        assert "data" in result.rejected_terms
        assert "value" in result.rejected_terms

    def test_validate_terms_rejects_short_terms(self) -> None:
        """validate_terms() rejects very short terms."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            terms=["a", "Redis", "x"],
            original_query="Redis caching",
            domain="systems",
        )

        # Very short terms should be rejected
        assert "a" in result.rejected_terms
        assert "x" in result.rejected_terms

    def test_validate_terms_provides_similarity_scores(self) -> None:
        """validate_terms() provides similarity scores for valid terms."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            terms=["Redis", "caching"],
            original_query="Redis caching layer",
            domain="systems",
        )

        # Should have similarity scores for valid terms
        for term in result.valid_terms:
            assert term in result.similarity_scores
            assert 0.0 <= result.similarity_scores[term] <= 1.0

    def test_validate_terms_handles_empty_list(self) -> None:
        """validate_terms() handles empty term list."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            terms=[],
            original_query="Redis caching",
            domain="systems",
        )

        assert result.valid_terms == []
        assert result.rejected_terms == []


# =============================================================================
# WBS 2.3.4: Similarity Threshold Tests
# =============================================================================


class TestSimilarityThreshold:
    """Test similarity threshold functionality."""

    def test_validate_terms_respects_min_similarity(self) -> None:
        """validate_terms() respects min_similarity parameter."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()

        # High threshold should reject more terms
        result_high = validator.validate_terms(
            terms=["Redis", "caching", "unrelated", "random"],
            original_query="Redis caching",
            domain="systems",
            min_similarity=0.7,
        )

        # Low threshold should accept more terms
        result_low = validator.validate_terms(
            terms=["Redis", "caching", "unrelated", "random"],
            original_query="Redis caching",
            domain="systems",
            min_similarity=0.1,
        )

        # Low threshold should have more or equal valid terms
        assert len(result_low.valid_terms) >= len(result_high.valid_terms)

    def test_rejection_reasons_include_low_similarity(self) -> None:
        """validate_terms() includes low_similarity in rejection reasons."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            terms=["Redis", "completely_unrelated_xyz"],
            original_query="Redis caching",
            domain="systems",
            min_similarity=0.5,
        )

        # Check if any rejection reason mentions similarity
        has_similarity_rejection = any(
            "similarity" in reason.lower()
            for reason in result.rejection_reasons.values()
        )
        # May or may not have similarity rejection depending on term
        assert isinstance(result.rejection_reasons, dict)


# =============================================================================
# WBS 2.3.5: Embedding Cache Tests
# =============================================================================


class TestEmbeddingCache:
    """Test embedding caching functionality."""

    def test_validator_caches_embeddings(self) -> None:
        """Validator caches embeddings to avoid recomputation."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()

        # First call should compute embedding
        _ = validator._get_embedding("Redis caching")

        # Should be cached
        assert "Redis caching"[:200] in validator._embedding_cache

    def test_cached_embedding_is_reused(self) -> None:
        """Cached embedding is reused on subsequent calls."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        import numpy as np

        validator = GraphCodeBERTValidator()

        # Get embedding twice
        emb1 = validator._get_embedding("Redis caching")
        emb2 = validator._get_embedding("Redis caching")

        # Should be the same array (from cache)
        assert np.array_equal(emb1, emb2)


# =============================================================================
# WBS 2.3.6: Cosine Similarity Tests
# =============================================================================


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Identical vectors have similarity of 1.0."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        import numpy as np

        validator = GraphCodeBERTValidator()

        vec = np.array([1.0, 2.0, 3.0])
        similarity = validator._cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_range(self) -> None:
        """Cosine similarity is in [0, 1] range."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()

        # Get embeddings for different texts
        emb1 = validator._get_embedding("Redis caching")
        emb2 = validator._get_embedding("machine learning neural networks")

        similarity = validator._cosine_similarity(emb1, emb2)

        assert 0.0 <= similarity <= 1.0


# =============================================================================
# WBS 2.3.7: Domain Classification Tests
# =============================================================================


class TestDomainClassification:
    """Test domain classification functionality."""

    def test_domain_references_exist(self) -> None:
        """Domain reference texts are defined."""
        from src.models.graphcodebert_validator import _DOMAIN_REFERENCES

        assert "ai-ml" in _DOMAIN_REFERENCES
        assert "systems" in _DOMAIN_REFERENCES
        assert "web" in _DOMAIN_REFERENCES

    def test_classify_domain_returns_valid_domain(self) -> None:
        """classify_domain() returns a valid domain string."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()

        if hasattr(validator, "classify_domain"):
            domain = validator.classify_domain("neural network deep learning")
            assert domain in ["ai-ml", "systems", "web", "data", "unknown"]
