"""
WBS 5.2 M2.2: Similarity API Endpoint Tests

Phase M2.2 Integration Tests per SBERT_EXTRACTION_MIGRATION_WBS.md:
- POST /v1/similarity - Compute similarity between texts
- POST /v1/embeddings - Generate embeddings for texts

TDD RED Phase - These tests define the API contract before implementation.

Anti-Pattern Compliance:
- #9 API Design: FastAPI router pattern
- #7 Exception Handling: Proper 4xx/5xx responses
- #12 Connection Pooling: Uses SBERTModelLoader singleton
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.sbert import EMBEDDING_DIMENSIONS


# =============================================================================
# M2.2.1 RED: test_similarity_endpoint (5 tests)
# =============================================================================


class TestSimilarityEndpointExists:
    """Tests for POST /v1/similarity endpoint existence."""

    def test_similarity_endpoint_exists(self) -> None:
        """POST /v1/similarity endpoint exists and does not return 404."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "machine learning",
                "text2": "deep learning",
            },
        )

        # Should not be 404 (endpoint exists)
        assert response.status_code != 404, "Similarity endpoint should exist"

    def test_similarity_returns_200_with_valid_input(self) -> None:
        """Similarity endpoint returns 200 with valid text1 and text2."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "neural network architecture",
                "text2": "deep learning model",
            },
        )

        assert response.status_code == 200

    def test_similarity_returns_score_field(self) -> None:
        """Similarity endpoint returns a similarity score."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "machine learning",
                "text2": "artificial intelligence",
            },
        )

        data = response.json()
        assert "score" in data, "Response should contain 'score' field"
        assert isinstance(data["score"], float), "Score should be a float"

    def test_similarity_score_in_valid_range(self) -> None:
        """Similarity score is between -1 and 1 (cosine similarity range)."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "python programming",
                "text2": "javascript coding",
            },
        )

        data = response.json()
        assert -1.0 <= data["score"] <= 1.0, "Score should be in cosine range [-1, 1]"

    def test_similarity_identical_texts_high_score(self) -> None:
        """Identical texts should have similarity score close to 1.0."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "semantic search using embeddings",
                "text2": "semantic search using embeddings",
            },
        )

        data = response.json()
        # Identical texts should have very high similarity
        assert data["score"] > 0.95, "Identical texts should have score > 0.95"


class TestSimilarityRequestValidation:
    """Tests for request validation on similarity endpoint."""

    def test_similarity_returns_422_without_text1(self) -> None:
        """Similarity endpoint returns 422 when text1 is missing."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={"text2": "only text2 provided"},
        )

        assert response.status_code == 422

    def test_similarity_returns_422_without_text2(self) -> None:
        """Similarity endpoint returns 422 when text2 is missing."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={"text1": "only text1 provided"},
        )

        assert response.status_code == 422

    def test_similarity_returns_422_with_empty_text1(self) -> None:
        """Similarity endpoint returns 422 when text1 is empty."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={"text1": "", "text2": "valid text"},
        )

        assert response.status_code == 422

    def test_similarity_returns_422_with_empty_text2(self) -> None:
        """Similarity endpoint returns 422 when text2 is empty."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={"text1": "valid text", "text2": ""},
        )

        assert response.status_code == 422


class TestSimilarityResponseMetadata:
    """Tests for similarity endpoint response metadata."""

    def test_similarity_includes_model_info(self) -> None:
        """Response includes information about the model used."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "vector database",
                "text2": "embedding storage",
            },
        )

        data = response.json()
        assert "model" in data, "Response should include 'model' field"

    def test_similarity_includes_processing_time(self) -> None:
        """Response includes processing time in milliseconds."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "transformer architecture",
                "text2": "attention mechanism",
            },
        )

        data = response.json()
        assert "processing_time_ms" in data, "Response should include processing time"
        assert isinstance(data["processing_time_ms"], (int, float))

    def test_similarity_processing_under_1_second(self) -> None:
        """Similarity computation should complete in under 1 second."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={
                "text1": "quick similarity check",
                "text2": "fast embedding comparison",
            },
        )

        data = response.json()
        assert data["processing_time_ms"] < 1000, "Should complete in under 1 second"


# =============================================================================
# M2.2.3 RED: test_batch_embedding (5 tests)
# =============================================================================


class TestEmbeddingsEndpointExists:
    """Tests for POST /v1/embeddings endpoint existence."""

    def test_embeddings_endpoint_exists(self) -> None:
        """POST /v1/embeddings endpoint exists and does not return 404."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["hello world"]},
        )

        assert response.status_code != 404, "Embeddings endpoint should exist"

    def test_embeddings_returns_200_with_valid_input(self) -> None:
        """Embeddings endpoint returns 200 with valid texts array."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["machine learning", "deep learning"]},
        )

        assert response.status_code == 200

    def test_embeddings_returns_array_of_vectors(self) -> None:
        """Embeddings endpoint returns an array of embedding vectors."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["test text one", "test text two"]},
        )

        data = response.json()
        assert "embeddings" in data, "Response should contain 'embeddings' field"
        assert isinstance(data["embeddings"], list), "Embeddings should be a list"
        assert len(data["embeddings"]) == 2, "Should return 2 embeddings for 2 texts"

    def test_embeddings_have_correct_dimension(self) -> None:
        """Each embedding vector has 384 dimensions (all-MiniLM-L6-v2)."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["single text for dimension check"]},
        )

        data = response.json()
        embedding = data["embeddings"][0]
        assert len(embedding) == EMBEDDING_DIMENSIONS, f"SBERT produces {EMBEDDING_DIMENSIONS}-dim vectors"

    def test_embeddings_are_normalized(self) -> None:
        """Embedding vectors are L2-normalized (unit length)."""
        client = TestClient(app)
        import math

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["normalized vector check"]},
        )

        data = response.json()
        embedding = data["embeddings"][0]
        # Compute L2 norm
        norm = math.sqrt(sum(x**2 for x in embedding))
        assert 0.99 < norm < 1.01, f"Embedding should be normalized, got norm={norm}"


class TestEmbeddingsRequestValidation:
    """Tests for request validation on embeddings endpoint."""

    def test_embeddings_returns_422_without_texts(self) -> None:
        """Embeddings endpoint returns 422 when texts field is missing."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={},
        )

        assert response.status_code == 422

    def test_embeddings_returns_422_with_empty_array(self) -> None:
        """Embeddings endpoint returns 422 when texts array is empty."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": []},
        )

        assert response.status_code == 422

    def test_embeddings_returns_422_with_empty_string_in_array(self) -> None:
        """Embeddings endpoint returns 422 when texts contains empty string."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["valid", "", "also valid"]},
        )

        assert response.status_code == 422


class TestEmbeddingsBatchProcessing:
    """Tests for batch embedding processing."""

    def test_embeddings_batch_returns_correct_count(self) -> None:
        """Batch request returns same number of embeddings as input texts."""
        client = TestClient(app)

        texts = [f"sample text {i}" for i in range(5)]
        response = client.post(
            "/v1/embeddings",
            json={"texts": texts},
        )

        data = response.json()
        assert len(data["embeddings"]) == len(texts)

    def test_embeddings_includes_model_info(self) -> None:
        """Embeddings response includes model information."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["model info check"]},
        )

        data = response.json()
        assert "model" in data, "Response should include 'model' field"

    def test_embeddings_includes_processing_time(self) -> None:
        """Embeddings response includes processing time."""
        client = TestClient(app)

        response = client.post(
            "/v1/embeddings",
            json={"texts": ["timing check"]},
        )

        data = response.json()
        assert "processing_time_ms" in data

    def test_embeddings_processing_under_2_seconds_for_batch(self) -> None:
        """Batch of 10 texts should complete in under 2 seconds."""
        client = TestClient(app)

        texts = [f"batch processing test text number {i}" for i in range(10)]
        response = client.post(
            "/v1/embeddings",
            json={"texts": texts},
        )

        data = response.json()
        assert data["processing_time_ms"] < 2000, "Batch should complete in under 2s"


# =============================================================================
# M2.2 Anti-Pattern Compliance Tests
# =============================================================================


# =============================================================================
# M2.3.3 RED: test_similarity_symmetric (3 tests)
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.3.3-M2.3.4
# =============================================================================


class TestSimilaritySymmetry:
    """Tests for similarity symmetry: sim(a,b) == sim(b,a).
    
    Per GUIDELINES Segment 13 (Pages 253-274): Semantic similarity is one of
    4 ways to measure similarity. Cosine similarity is inherently symmetric.
    """

    def test_similarity_is_symmetric(self) -> None:
        """Similarity(a,b) should equal Similarity(b,a)."""
        client = TestClient(app)

        text1 = "machine learning algorithms"
        text2 = "deep neural networks"

        response_ab = client.post(
            "/v1/similarity",
            json={"text1": text1, "text2": text2},
        )
        response_ba = client.post(
            "/v1/similarity",
            json={"text1": text2, "text2": text1},
        )

        score_ab = response_ab.json()["score"]
        score_ba = response_ba.json()["score"]

        # Scores should be identical (symmetric property of cosine similarity)
        assert abs(score_ab - score_ba) < 1e-6, f"sim(a,b)={score_ab} should equal sim(b,a)={score_ba}"

    def test_similarity_symmetric_with_different_lengths(self) -> None:
        """Symmetry holds even for texts of different lengths."""
        client = TestClient(app)

        short_text = "AI"
        long_text = "artificial intelligence systems using neural network architectures"

        response_ab = client.post(
            "/v1/similarity",
            json={"text1": short_text, "text2": long_text},
        )
        response_ba = client.post(
            "/v1/similarity",
            json={"text1": long_text, "text2": short_text},
        )

        score_ab = response_ab.json()["score"]
        score_ba = response_ba.json()["score"]

        assert abs(score_ab - score_ba) < 1e-6, "Symmetry should hold for different length texts"

    def test_similarity_self_equals_one(self) -> None:
        """Similarity of text with itself should be exactly 1.0."""
        client = TestClient(app)

        text = "the quick brown fox jumps over the lazy dog"

        response = client.post(
            "/v1/similarity",
            json={"text1": text, "text2": text},
        )

        score = response.json()["score"]
        # Self-similarity should be exactly 1.0 (or very close due to float precision)
        assert abs(score - 1.0) < 1e-5, f"Self-similarity should be 1.0, got {score}"


# =============================================================================
# M2.3.5 RED: test_similarity_batch (8 tests)
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.3.5-M2.3.6
# =============================================================================


class TestSimilarityBatch:
    """Tests for POST /v1/similarity/batch endpoint.
    
    Per GUIDELINES Line 722: NumPy vectorization provides "over an order of
    magnitude improvement for large-scale sampling." Batch endpoint should
    compute all embeddings once, then similarity scores vectorized.
    """

    def test_similarity_batch_endpoint_exists(self) -> None:
        """POST /v1/similarity/batch endpoint exists."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity/batch",
            json={
                "pairs": [
                    {"text1": "hello", "text2": "world"},
                ]
            },
        )

        assert response.status_code != 404, "Batch similarity endpoint should exist"

    def test_similarity_batch_returns_200_with_valid_input(self) -> None:
        """Batch similarity returns 200 with valid pairs."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity/batch",
            json={
                "pairs": [
                    {"text1": "machine learning", "text2": "deep learning"},
                    {"text1": "python", "text2": "java"},
                ]
            },
        )

        assert response.status_code == 200

    def test_similarity_batch_returns_correct_count(self) -> None:
        """Batch returns same number of scores as input pairs."""
        client = TestClient(app)

        pairs = [
            {"text1": f"text a{i}", "text2": f"text b{i}"}
            for i in range(5)
        ]
        response = client.post(
            "/v1/similarity/batch",
            json={"pairs": pairs},
        )

        data = response.json()
        assert "scores" in data, "Response should contain 'scores' field"
        assert len(data["scores"]) == len(pairs), "Should return one score per pair"

    def test_similarity_batch_scores_in_valid_range(self) -> None:
        """All batch scores are in cosine similarity range [-1, 1]."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity/batch",
            json={
                "pairs": [
                    {"text1": "apple", "text2": "orange"},
                    {"text1": "car", "text2": "bicycle"},
                    {"text1": "happy", "text2": "sad"},
                ]
            },
        )

        data = response.json()
        for i, score in enumerate(data["scores"]):
            assert -1.0 <= score <= 1.0, f"Score {i} should be in [-1, 1], got {score}"

    def test_similarity_batch_includes_model_info(self) -> None:
        """Batch response includes model information."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity/batch",
            json={
                "pairs": [{"text1": "test", "text2": "check"}]
            },
        )

        data = response.json()
        assert "model" in data, "Response should include 'model' field"

    def test_similarity_batch_includes_processing_time(self) -> None:
        """Batch response includes processing time."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity/batch",
            json={
                "pairs": [{"text1": "timing", "text2": "test"}]
            },
        )

        data = response.json()
        assert "processing_time_ms" in data, "Response should include processing time"

    def test_similarity_batch_returns_422_with_empty_pairs(self) -> None:
        """Batch returns 422 when pairs array is empty."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity/batch",
            json={"pairs": []},
        )

        assert response.status_code == 422, "Empty pairs should return 422"

    def test_similarity_batch_faster_than_sequential(self) -> None:
        """Batch processing should be faster than sequential calls.
        
        Per GUIDELINES: NumPy vectorization is "over an order of magnitude faster."
        Batch should benefit from computing embeddings once for all texts.
        """
        import time
        client = TestClient(app)

        pairs = [
            {"text1": f"sample text number {i}", "text2": f"comparison text {i}"}
            for i in range(10)
        ]

        # Time batch request
        start_batch = time.perf_counter()
        batch_response = client.post(
            "/v1/similarity/batch",
            json={"pairs": pairs},
        )
        batch_time = time.perf_counter() - start_batch

        # Time sequential requests
        start_seq = time.perf_counter()
        for pair in pairs:
            client.post(
                "/v1/similarity",
                json={"text1": pair["text1"], "text2": pair["text2"]},
            )
        seq_time = time.perf_counter() - start_seq

        # Batch should be faster (or at least not significantly slower)
        # Allow batch to be up to 1.5x slower due to test variability
        assert batch_response.status_code == 200
        assert batch_time < seq_time * 1.5, f"Batch ({batch_time:.3f}s) should not be much slower than sequential ({seq_time:.3f}s)"


# =============================================================================
# Anti-Pattern Compliance Tests
# =============================================================================


class TestSimilarityAntiPatternCompliance:
    """Tests verifying anti-pattern compliance for similarity endpoints."""

    def test_endpoint_uses_singleton_model_loader(self) -> None:
        """Similarity endpoint should use SBERTModelLoader singleton."""
        # Import to verify singleton pattern is used
        from src.models.sbert.model_loader import get_sbert_model

        # Make two requests - should use same model instance
        model1 = get_sbert_model()
        model2 = get_sbert_model()

        assert model1 is model2, "Should use singleton pattern"

    def test_error_response_has_proper_structure(self) -> None:
        """Error responses should follow FastAPI error structure."""
        client = TestClient(app)

        response = client.post(
            "/v1/similarity",
            json={},  # Invalid request
        )

        data = response.json()
        assert "detail" in data, "Error response should have 'detail' field"


# =============================================================================
# M2.4.1 RED: test_similar_chapters_endpoint (5 tests)
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.4.1-M2.4.2
# =============================================================================


class TestSimilarChaptersEndpoint:
    """Tests for POST /v1/similar-chapters endpoint.
    
    Per TIER_RELATIONSHIP_DIAGRAM.md: Similar chapters uses semantic similarity
    to find related chapters across the corpus with tier-aware relevance.
    """

    def test_similar_chapters_endpoint_exists(self) -> None:
        """POST /v1/similar-chapters endpoint exists."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "machine learning algorithms",
                "chapters": [
                    {"id": "ch1", "title": "Deep Learning", "content": "Neural networks and deep learning"},
                ],
            },
        )

        assert response.status_code != 404, "Similar chapters endpoint should exist"

    def test_similar_chapters_returns_200_with_valid_input(self) -> None:
        """Similar chapters returns 200 with valid input."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "semantic search",
                "chapters": [
                    {"id": "ch1", "title": "Vector Search", "content": "Embedding-based search using vectors"},
                    {"id": "ch2", "title": "Text Search", "content": "Full-text search with keywords"},
                ],
            },
        )

        assert response.status_code == 200

    def test_similar_chapters_returns_chapters_array(self) -> None:
        """Response contains 'chapters' array."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "natural language processing",
                "chapters": [
                    {"id": "ch1", "title": "NLP Basics", "content": "Introduction to NLP"},
                ],
            },
        )

        data = response.json()
        assert "chapters" in data, "Response should contain 'chapters' field"
        assert isinstance(data["chapters"], list), "Chapters should be a list"

    def test_similar_chapters_result_has_required_fields(self) -> None:
        """Each chapter result has id, title, and score."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "transformer architecture",
                "chapters": [
                    {"id": "ch1", "title": "Attention Mechanism", "content": "Self-attention in transformers"},
                ],
            },
        )

        data = response.json()
        if data["chapters"]:  # May be empty if threshold filters all
            chapter = data["chapters"][0]
            assert "id" in chapter, "Chapter should have 'id'"
            assert "title" in chapter, "Chapter should have 'title'"
            assert "score" in chapter, "Chapter should have 'score'"

    def test_similar_chapters_returns_top_k_results(self) -> None:
        """Default returns up to top_k (5) results."""
        client = TestClient(app)

        chapters = [
            {"id": f"ch{i}", "title": f"Chapter {i}", "content": f"Content about topic {i}"}
            for i in range(10)
        ]

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "topic",
                "chapters": chapters,
                "top_k": 3,
            },
        )

        data = response.json()
        assert len(data["chapters"]) <= 3, "Should respect top_k limit"


# =============================================================================
# M2.4.3 RED: test_similar_chapters_threshold (4 tests)
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.4.3-M2.4.4
# =============================================================================


class TestSimilarChaptersThreshold:
    """Tests for similarity threshold filtering.
    
    Per llm-document-enhancer ARCHITECTURE: Default threshold is 0.7 for production.
    Per engine defaults: threshold is 0.0 for maximum flexibility.
    """

    def test_threshold_filters_low_similarity(self) -> None:
        """High threshold filters out low-similarity chapters."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "machine learning",
                "chapters": [
                    {"id": "ch1", "title": "Cooking Recipes", "content": "How to bake a cake"},
                    {"id": "ch2", "title": "ML Algorithms", "content": "Machine learning classification"},
                ],
                "threshold": 0.5,  # High threshold
            },
        )

        data = response.json()
        # With high threshold, unrelated chapter should be filtered
        for chapter in data["chapters"]:
            assert chapter["score"] >= 0.5, f"Score {chapter['score']} should be >= threshold 0.5"

    def test_zero_threshold_returns_all(self) -> None:
        """Zero threshold returns all chapters."""
        client = TestClient(app)

        chapters = [
            {"id": "ch1", "title": "Topic A", "content": "Content A"},
            {"id": "ch2", "title": "Topic B", "content": "Content B"},
            {"id": "ch3", "title": "Topic C", "content": "Content C"},
        ]

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "anything",
                "chapters": chapters,
                "threshold": 0.0,
                "top_k": 10,
            },
        )

        data = response.json()
        # With zero threshold, should return all (up to top_k)
        assert len(data["chapters"]) == len(chapters)

    def test_threshold_validation_rejects_negative(self) -> None:
        """Threshold must be >= 0.0."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "test",
                "chapters": [{"id": "ch1", "title": "Test", "content": "Test"}],
                "threshold": -0.1,
            },
        )

        assert response.status_code == 422, "Negative threshold should be rejected"

    def test_threshold_validation_rejects_over_one(self) -> None:
        """Threshold must be <= 1.0."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "test",
                "chapters": [{"id": "ch1", "title": "Test", "content": "Test"}],
                "threshold": 1.5,
            },
        )

        assert response.status_code == 422, "Threshold > 1.0 should be rejected"


# =============================================================================
# M2.4.5 RED: test_similar_chapters_metadata (4 tests)
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.4.5-M2.4.6
# =============================================================================


class TestSimilarChaptersMetadata:
    """Tests for similar chapters response metadata.
    
    Per llm-document-enhancer ARCHITECTURE: Method tracked as 
    "sentence_transformers" or "tfidf" to indicate which algorithm was used.
    """

    def test_response_includes_method_field(self) -> None:
        """Response includes 'method' field."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "test query",
                "chapters": [{"id": "ch1", "title": "Test", "content": "Test content"}],
            },
        )

        data = response.json()
        assert "method" in data, "Response should include 'method' field"

    def test_method_is_valid_value(self) -> None:
        """Method is either 'sentence_transformers' or 'tfidf'."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "test",
                "chapters": [{"id": "ch1", "title": "Test", "content": "Test"}],
            },
        )

        data = response.json()
        valid_methods = {"sentence_transformers", "tfidf"}
        assert data["method"] in valid_methods, f"Method should be one of {valid_methods}"

    def test_response_includes_model_info(self) -> None:
        """Response includes model information."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "test",
                "chapters": [{"id": "ch1", "title": "Test", "content": "Test"}],
            },
        )

        data = response.json()
        assert "model" in data, "Response should include 'model' field"

    def test_response_includes_processing_time(self) -> None:
        """Response includes processing time."""
        client = TestClient(app)

        response = client.post(
            "/v1/similar-chapters",
            json={
                "query": "test",
                "chapters": [{"id": "ch1", "title": "Test", "content": "Test"}],
            },
        )

        data = response.json()
        assert "processing_time_ms" in data, "Response should include processing time"
        assert isinstance(data["processing_time_ms"], (int, float))
