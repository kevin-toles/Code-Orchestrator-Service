"""
Integration Tests for Keywords API Endpoint

WBS: MSE-1.6 - Integration Tests for /api/v1/keywords Endpoint
TDD Phase: RED (tests written BEFORE implementation)

Tests for:
- POST /api/v1/keywords endpoint
- Request/Response validation
- Error handling
- Processing time reporting

Acceptance Criteria (from MSEP WBS):
- AC-1.2.1: POST /api/v1/keywords accepts corpus and top_k
- AC-1.2.2: Returns {"keywords": [[...], ...], "processing_time_ms": ...}
- AC-1.2.3: Handles empty corpus gracefully
- AC-1.2.4: Returns 400 for invalid input
- AC-1.2.5: 0 SonarQube issues

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: No duplicated string literals
- #2.2: Full type annotations
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Module constants per S1192 (no duplicated literals)
KEYWORDS_ENDPOINT: str = "/api/v1/keywords"
CONTENT_TYPE_JSON: str = "application/json"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create test client for Code-Orchestrator-Service."""
    from src.main import app

    return TestClient(app)


@pytest.fixture
def sample_corpus_request() -> dict:
    """Sample request body with valid corpus."""
    return {
        "corpus": [
            "Machine learning and deep learning are subfields of artificial intelligence. "
            "Neural networks power many modern AI systems.",
            "Python programming language is widely used for data science and machine learning. "
            "Libraries like scikit-learn and TensorFlow are popular.",
        ],
        "top_k": 5,
    }


@pytest.fixture
def empty_corpus_request() -> dict:
    """Request with empty corpus."""
    return {"corpus": [], "top_k": 5}


@pytest.fixture
def single_doc_request() -> dict:
    """Request with single document."""
    return {
        "corpus": ["Python is great for machine learning and data science applications."],
        "top_k": 3,
    }


# =============================================================================
# AC-1.2.1: POST /api/v1/keywords accepts corpus and top_k
# =============================================================================


class TestKeywordsEndpointAcceptsRequest:
    """Test that endpoint accepts valid requests."""

    def test_endpoint_exists(self, client: TestClient) -> None:
        """AC-1.2.1: POST /api/v1/keywords should exist."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": ["test document"], "top_k": 5},
        )
        # Should not return 404
        assert response.status_code != 404, "Endpoint not found"

    def test_accepts_corpus_list(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """AC-1.2.1: Should accept corpus as list of strings."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        assert response.status_code == 200

    def test_accepts_top_k_parameter(self, client: TestClient) -> None:
        """AC-1.2.1: Should accept top_k parameter."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": ["test document"], "top_k": 3},
        )
        assert response.status_code == 200

    def test_top_k_is_optional(self, client: TestClient) -> None:
        """AC-1.2.1: top_k should be optional with default value."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": ["test document for keyword extraction"]},
        )
        assert response.status_code == 200


# =============================================================================
# AC-1.2.2: Returns {"keywords": [[...], ...], "processing_time_ms": ...}
# =============================================================================


class TestKeywordsEndpointResponseFormat:
    """Test response format matches specification."""

    def test_returns_keywords_field(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """AC-1.2.2: Response should contain 'keywords' field."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        assert "keywords" in data, "Response missing 'keywords' field"

    def test_keywords_is_list_of_lists(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """AC-1.2.2: 'keywords' should be list of keyword lists."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        assert isinstance(data["keywords"], list)
        assert len(data["keywords"]) == len(sample_corpus_request["corpus"])
        for doc_keywords in data["keywords"]:
            assert isinstance(doc_keywords, list)

    def test_keywords_are_strings(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """AC-1.2.2: Keywords should be strings."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        for doc_keywords in data["keywords"]:
            for keyword in doc_keywords:
                assert isinstance(keyword, str)

    def test_returns_processing_time_ms(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """AC-1.2.2: Response should contain 'processing_time_ms' field."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        assert "processing_time_ms" in data, "Response missing 'processing_time_ms' field"
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] >= 0

    def test_respects_top_k_in_results(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """AC-1.2.2: Should return at most top_k keywords per document."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        top_k = sample_corpus_request["top_k"]
        for doc_keywords in data["keywords"]:
            assert len(doc_keywords) <= top_k


# =============================================================================
# AC-1.2.3: Handles empty corpus gracefully
# =============================================================================


class TestKeywordsEndpointEmptyCorpus:
    """Test handling of empty corpus."""

    def test_empty_corpus_returns_200(
        self, client: TestClient, empty_corpus_request: dict
    ) -> None:
        """AC-1.2.3: Empty corpus should return 200 OK."""
        response = client.post(KEYWORDS_ENDPOINT, json=empty_corpus_request)
        assert response.status_code == 200

    def test_empty_corpus_returns_empty_keywords(
        self, client: TestClient, empty_corpus_request: dict
    ) -> None:
        """AC-1.2.3: Empty corpus should return empty keywords list."""
        response = client.post(KEYWORDS_ENDPOINT, json=empty_corpus_request)
        data = response.json()
        
        assert data["keywords"] == []

    def test_corpus_with_empty_strings(self, client: TestClient) -> None:
        """AC-1.2.3: Corpus with empty strings should be handled."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": ["valid document", "", "another valid doc"], "top_k": 3},
        )
        data = response.json()
        
        assert response.status_code == 200
        assert len(data["keywords"]) == 3
        assert data["keywords"][1] == []  # Empty document should give empty keywords


# =============================================================================
# AC-1.2.4: Returns 400 for invalid input
# =============================================================================


class TestKeywordsEndpointValidation:
    """Test input validation and error responses."""

    def test_missing_corpus_returns_422(self, client: TestClient) -> None:
        """AC-1.2.4: Missing corpus should return 422 Unprocessable Entity."""
        response = client.post(KEYWORDS_ENDPOINT, json={"top_k": 5})
        assert response.status_code == 422

    def test_corpus_not_list_returns_422(self, client: TestClient) -> None:
        """AC-1.2.4: corpus as string instead of list should return 422."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": "single string not list", "top_k": 5},
        )
        assert response.status_code == 422

    def test_corpus_with_non_strings_returns_422(self, client: TestClient) -> None:
        """AC-1.2.4: corpus with non-string items should return 422."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": ["valid", 123, "also valid"], "top_k": 5},
        )
        assert response.status_code == 422

    def test_negative_top_k_returns_422(self, client: TestClient) -> None:
        """AC-1.2.4: Negative top_k should return 422."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": ["test document"], "top_k": -1},
        )
        assert response.status_code == 422

    def test_top_k_zero_is_valid(self, client: TestClient) -> None:
        """top_k=0 should be valid (returns empty keywords)."""
        response = client.post(
            KEYWORDS_ENDPOINT,
            json={"corpus": ["test document"], "top_k": 0},
        )
        # Should be 200 with empty keywords
        assert response.status_code == 200
        data = response.json()
        assert data["keywords"] == [[]]


# =============================================================================
# Additional Functional Tests
# =============================================================================


class TestKeywordsEndpointFunctionality:
    """Test keyword extraction functionality through the API."""

    def test_single_document_extraction(
        self, client: TestClient, single_doc_request: dict
    ) -> None:
        """Should extract keywords from single document."""
        response = client.post(KEYWORDS_ENDPOINT, json=single_doc_request)
        data = response.json()
        
        assert response.status_code == 200
        assert len(data["keywords"]) == 1
        assert len(data["keywords"][0]) > 0

    def test_keywords_contain_meaningful_terms(
        self, client: TestClient, single_doc_request: dict
    ) -> None:
        """Extracted keywords should include meaningful domain terms."""
        response = client.post(KEYWORDS_ENDPOINT, json=single_doc_request)
        data = response.json()
        
        all_keywords = [kw.lower() for kw in data["keywords"][0]]
        
        # At least one domain-relevant term should appear
        expected_terms = {"python", "machine", "learning", "data", "science"}
        found_terms = expected_terms.intersection(set(all_keywords))
        
        assert len(found_terms) >= 1, f"Expected domain terms, got: {all_keywords}"

    def test_stop_words_filtered(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """Stop words should be filtered from results."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        stop_words = {"the", "and", "is", "are", "a", "an", "of", "to", "in", "for"}
        
        for doc_keywords in data["keywords"]:
            for keyword in doc_keywords:
                # Single word keywords should not be stop words
                if " " not in keyword:
                    assert keyword.lower() not in stop_words


# =============================================================================
# Response Schema Tests
# =============================================================================


class TestKeywordsResponseSchema:
    """Test response adheres to Pydantic schema."""

    def test_response_has_all_fields(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """Response should have all required fields."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        required_fields = {"keywords", "processing_time_ms"}
        assert required_fields.issubset(data.keys())

    def test_processing_time_is_positive(
        self, client: TestClient, sample_corpus_request: dict
    ) -> None:
        """Processing time should be a positive number."""
        response = client.post(KEYWORDS_ENDPOINT, json=sample_corpus_request)
        data = response.json()
        
        assert data["processing_time_ms"] >= 0


# =============================================================================
# Keywords with Scores Endpoint (Optional)
# =============================================================================


class TestKeywordsWithScoresEndpoint:
    """Test optional endpoint that returns keywords with scores."""

    def test_with_scores_endpoint_exists(self, client: TestClient) -> None:
        """Optional: POST /api/v1/keywords/scores endpoint should exist."""
        response = client.post(
            "/api/v1/keywords/scores",
            json={"corpus": ["test document"], "top_k": 5},
        )
        # Should not return 404 if implemented
        # This is optional functionality - skip if not implemented
        if response.status_code == 404:
            pytest.skip("Keywords with scores endpoint not implemented")

    def test_with_scores_returns_score_field(self, client: TestClient) -> None:
        """Keywords with scores should include score for each keyword."""
        response = client.post(
            "/api/v1/keywords/scores",
            json={"corpus": ["Machine learning is a branch of artificial intelligence."], "top_k": 3},
        )
        
        if response.status_code == 404:
            pytest.skip("Keywords with scores endpoint not implemented")
        
        data = response.json()
        assert "keywords" in data
        
        for doc_keywords in data["keywords"]:
            for item in doc_keywords:
                assert "keyword" in item
                assert "score" in item
                assert isinstance(item["score"], (int, float))
                assert 0.0 <= item["score"] <= 1.0
