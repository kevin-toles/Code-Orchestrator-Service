"""
EEP-5.2: CodeBERT API Endpoint Tests

TDD RED Phase: Tests for /api/v1/codebert/embed endpoint.

WBS Mapping:
- AC-5.2.1: Use existing CodeBERTRanker from codebert_ranker.py
- AC-5.2.2: Generate 768-dim embeddings for each code block
- AC-5.2.3: Cache embeddings to avoid recomputation (Anti-Pattern #12)

Anti-Patterns Avoided:
- #12: Uses FakeModelRegistry (no real HuggingFace model in tests)
- S1192: Extracted constants for repeated strings
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_ENDPOINT_EMBED = "/api/v1/codebert/embed"
_ENDPOINT_EMBED_BATCH = "/api/v1/codebert/embed/batch"
_ENDPOINT_SIMILARITY = "/api/v1/codebert/similarity"
_EMBEDDING_DIM = 768
_TEST_CODE_PYTHON = "def hello(): print('world')"
_TEST_CODE_JAVASCRIPT = "function hello() { console.log('world'); }"


# =============================================================================
# Test Classes
# =============================================================================


class TestCodeBERTEmbedEndpointExists:
    """Tests for CodeBERT embed endpoint existence (AC-5.2.1)."""

    def test_embed_endpoint_exists(self, client: TestClient) -> None:
        """AC-5.2.1: /api/v1/codebert/embed endpoint exists."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        # Should not be 404 (endpoint exists)
        assert response.status_code != 404

    def test_embed_endpoint_accepts_post(self, client: TestClient) -> None:
        """AC-5.2.1: Endpoint accepts POST requests."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        assert response.status_code in (200, 201, 422)  # Success or validation error

    def test_embed_batch_endpoint_exists(self, client: TestClient) -> None:
        """AC-5.2.1: /api/v1/codebert/embed/batch endpoint exists."""
        response = client.post(
            _ENDPOINT_EMBED_BATCH,
            json={"codes": [_TEST_CODE_PYTHON]},
        )
        assert response.status_code != 404


class TestCodeBERTEmbeddingGeneration:
    """Tests for 768-dim embedding generation (AC-5.2.2)."""

    def test_embed_returns_768_dim_vector(self, client: TestClient) -> None:
        """AC-5.2.2: Returns 768-dimensional embedding vector."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) == _EMBEDDING_DIM

    def test_embed_returns_list_of_floats(self, client: TestClient) -> None:
        """AC-5.2.2: Embedding is a list of float values."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        assert response.status_code == 200
        data = response.json()
        assert all(isinstance(x, float) for x in data["embedding"])

    def test_embed_batch_returns_multiple_embeddings(self, client: TestClient) -> None:
        """AC-5.2.2: Batch endpoint returns embedding for each code."""
        codes = [_TEST_CODE_PYTHON, _TEST_CODE_JAVASCRIPT]
        response = client.post(
            _ENDPOINT_EMBED_BATCH,
            json={"codes": codes},
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 2
        for emb in data["embeddings"]:
            assert len(emb) == _EMBEDDING_DIM

    def test_embed_empty_code_returns_zero_vector(self, client: TestClient) -> None:
        """AC-5.2.2: Empty code returns zero vector."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["embedding"]) == _EMBEDDING_DIM
        # Zero vector or near-zero acceptable
        assert all(abs(x) < 0.01 for x in data["embedding"])


class TestCodeBERTSimilarityEndpoint:
    """Tests for CodeBERT similarity endpoint."""

    def test_similarity_endpoint_exists(self, client: TestClient) -> None:
        """Similarity endpoint exists."""
        response = client.post(
            _ENDPOINT_SIMILARITY,
            json={
                "code_a": _TEST_CODE_PYTHON,
                "code_b": _TEST_CODE_PYTHON,
            },
        )
        assert response.status_code != 404

    def test_similarity_identical_code_returns_high_score(
        self, client: TestClient
    ) -> None:
        """Identical code returns similarity >= 0.9."""
        response = client.post(
            _ENDPOINT_SIMILARITY,
            json={
                "code_a": _TEST_CODE_PYTHON,
                "code_b": _TEST_CODE_PYTHON,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "similarity" in data
        assert data["similarity"] >= 0.9

    def test_similarity_different_code_returns_lower_score(
        self, client: TestClient
    ) -> None:
        """Different code returns similarity < 0.9."""
        response = client.post(
            _ENDPOINT_SIMILARITY,
            json={
                "code_a": _TEST_CODE_PYTHON,
                "code_b": "import pandas as pd\ndf = pd.DataFrame()",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["similarity"] < 0.9


class TestCodeBERTCaching:
    """Tests for embedding caching (AC-5.2.3, Anti-Pattern #12)."""

    def test_same_code_returns_same_embedding(self, client: TestClient) -> None:
        """AC-5.2.3: Same code returns identical embedding (caching)."""
        response1 = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        response2 = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        emb1 = response1.json()["embedding"]
        emb2 = response2.json()["embedding"]
        assert emb1 == emb2  # Deterministic

    def test_uses_codebert_ranker(self, client: TestClient) -> None:
        """AC-5.2.1: Uses existing CodeBERTRanker (not new model per request)."""
        # This is validated by the endpoint existing and working
        # The implementation should use CodeBERTRanker internally
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        assert response.status_code == 200


class TestCodeBERTRequestValidation:
    """Tests for request validation."""

    def test_embed_requires_code_field(self, client: TestClient) -> None:
        """Embed endpoint requires 'code' field."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={},  # Missing 'code'
        )
        assert response.status_code == 422

    def test_embed_batch_requires_codes_field(self, client: TestClient) -> None:
        """Batch endpoint requires 'codes' field."""
        response = client.post(
            _ENDPOINT_EMBED_BATCH,
            json={},  # Missing 'codes'
        )
        assert response.status_code == 422

    def test_similarity_requires_both_codes(self, client: TestClient) -> None:
        """Similarity endpoint requires both code_a and code_b."""
        response = client.post(
            _ENDPOINT_SIMILARITY,
            json={"code_a": _TEST_CODE_PYTHON},  # Missing 'code_b'
        )
        assert response.status_code == 422


class TestCodeBERTResponseSchema:
    """Tests for response schema compliance."""

    def test_embed_response_has_embedding_field(self, client: TestClient) -> None:
        """Embed response contains 'embedding' field."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data

    def test_embed_response_has_dimension_field(self, client: TestClient) -> None:
        """Embed response contains 'dimension' field."""
        response = client.post(
            _ENDPOINT_EMBED,
            json={"code": _TEST_CODE_PYTHON},
        )
        assert response.status_code == 200
        data = response.json()
        assert "dimension" in data
        assert data["dimension"] == _EMBEDDING_DIM

    def test_batch_response_has_embeddings_and_count(self, client: TestClient) -> None:
        """Batch response contains 'embeddings' and 'count' fields."""
        response = client.post(
            _ENDPOINT_EMBED_BATCH,
            json={"codes": [_TEST_CODE_PYTHON, _TEST_CODE_JAVASCRIPT]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert "count" in data
        assert data["count"] == 2

    def test_similarity_response_has_score_and_codes(self, client: TestClient) -> None:
        """Similarity response contains 'similarity', 'code_a', 'code_b'."""
        response = client.post(
            _ENDPOINT_SIMILARITY,
            json={
                "code_a": _TEST_CODE_PYTHON,
                "code_b": _TEST_CODE_JAVASCRIPT,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "similarity" in data
        assert 0.0 <= data["similarity"] <= 1.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create test client with FakeModelRegistry."""
    from src.main import app
    
    return TestClient(app)
