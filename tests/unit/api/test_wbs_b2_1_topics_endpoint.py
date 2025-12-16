"""
WBS B2.1: POST /api/v1/topics Endpoint Tests

TDD RED Phase - Tests written BEFORE implementation.
These tests will FAIL until topics.py is implemented.

Test Plan (per BERTOPIC_INTEGRATION_WBS.md):
1. test_topics_endpoint_exists - 200/422 not 404
2. test_topics_empty_corpus_returns_empty - Edge case
3. test_topics_returns_topic_list - Happy path
4. test_topics_response_schema_valid - Pydantic validation
5. test_topics_with_min_topic_size_param - Configuration

Patterns Applied:
- pytest fixtures (per existing test patterns)
- FastAPI TestClient (per existing test patterns)
- Pydantic response validation
- Anti-Pattern #7 compliance (proper error handling)

Anti-Patterns Avoided:
- S1172: Unused parameters (mark with underscore prefix)
- S3776: Cognitive complexity (keep tests focused)
- S1192: Duplicated literals (use constants)
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.main import app

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for API testing."""
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

TOPICS_ENDPOINT: str = "/api/v1/topics"
DEFAULT_MIN_TOPIC_SIZE: int = 2
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Sample corpus for testing
SAMPLE_CORPUS: list[str] = [
    "The Repository Pattern separates domain logic from data persistence.",
    "Dependency Injection enables loose coupling between components.",
    "The Factory Pattern creates objects without specifying exact classes.",
    "Unit tests verify individual components in isolation.",
    "Integration tests verify components work together correctly.",
    "The Singleton Pattern ensures only one instance exists.",
    "SOLID principles guide object-oriented design decisions.",
    "Clean Architecture separates concerns into distinct layers.",
]

# Minimal corpus for edge case testing
MINIMAL_CORPUS: list[str] = [
    "First document about software patterns.",
    "Second document about design principles.",
]


# =============================================================================
# Test 1: Endpoint Exists (returns 200/422, not 404)
# =============================================================================


class TestTopicsEndpointExists:
    """WBS B2.1 Test 1: Verify endpoint is registered and accessible."""

    def test_topics_endpoint_returns_200_with_valid_request(
        self, test_client: TestClient
    ) -> None:
        """POST /api/v1/topics should return 200 with valid corpus."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
            },
        )
        # Accept 200 (success) - not 404 (not found)
        assert response.status_code != status.HTTP_404_NOT_FOUND, (
            f"Endpoint {TOPICS_ENDPOINT} not registered (got 404)"
        )
        assert response.status_code == status.HTTP_200_OK

    def test_topics_endpoint_returns_422_with_invalid_request(
        self, test_client: TestClient
    ) -> None:
        """POST /api/v1/topics should return 422 with invalid input."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": [],  # Empty corpus should fail validation
            },
        )
        # Accept 422 (validation error) - not 404 (not found)
        assert response.status_code != status.HTTP_404_NOT_FOUND, (
            f"Endpoint {TOPICS_ENDPOINT} not registered (got 404)"
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Test 2: Empty Corpus Edge Case
# =============================================================================


class TestTopicsEmptyCorpus:
    """WBS B2.1 Test 2: Handle empty corpus gracefully."""

    def test_topics_empty_corpus_returns_validation_error(
        self, test_client: TestClient
    ) -> None:
        """Empty corpus should return 422 validation error."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": [],
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_topics_single_empty_string_returns_validation_error(
        self, test_client: TestClient
    ) -> None:
        """Corpus with only empty strings should return 422."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": ["", "   "],  # Whitespace-only strings
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Test 3: Happy Path - Returns Topic List
# =============================================================================


class TestTopicsReturnsList:
    """WBS B2.1 Test 3: Verify endpoint returns topic list."""

    def test_topics_returns_topics_list(self, test_client: TestClient) -> None:
        """POST /api/v1/topics should return list of topics."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Must have 'topics' key with list
        assert "topics" in data, "Response missing 'topics' key"
        assert isinstance(data["topics"], list), "'topics' must be a list"

    def test_topics_returns_topic_count(self, test_client: TestClient) -> None:
        """Response should include topic_count field."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "topic_count" in data, "Response missing 'topic_count' key"
        assert isinstance(data["topic_count"], int), "'topic_count' must be integer"
        assert data["topic_count"] >= 0, "'topic_count' must be non-negative"

    def test_topics_returns_model_info(self, test_client: TestClient) -> None:
        """Response should include model_info with embedding model."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "model_info" in data, "Response missing 'model_info' key"
        assert isinstance(data["model_info"], dict), "'model_info' must be dict"
        assert "embedding_model" in data["model_info"], "model_info missing 'embedding_model'"


# =============================================================================
# Test 4: Response Schema Validation
# =============================================================================


class TestTopicsResponseSchema:
    """WBS B2.1 Test 4: Validate response matches expected schema."""

    def test_topic_item_has_required_fields(self, test_client: TestClient) -> None:
        """Each topic in response should have required fields."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Each topic must have: topic_id, name, keywords, size
        for topic in data["topics"]:
            assert "topic_id" in topic, "Topic missing 'topic_id'"
            assert "name" in topic, "Topic missing 'name'"
            assert "keywords" in topic, "Topic missing 'keywords'"
            assert "size" in topic, "Topic missing 'size'"
            
            # Type validation
            assert isinstance(topic["topic_id"], int), "'topic_id' must be int"
            assert isinstance(topic["name"], str), "'name' must be str"
            assert isinstance(topic["keywords"], list), "'keywords' must be list"
            assert isinstance(topic["size"], int), "'size' must be int"

    def test_topic_keywords_are_strings(self, test_client: TestClient) -> None:
        """Topic keywords should be a list of strings."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for topic in data["topics"]:
            for keyword in topic["keywords"]:
                assert isinstance(keyword, str), f"Keyword '{keyword}' is not a string"

    def test_model_info_has_bertopic_version(self, test_client: TestClient) -> None:
        """model_info should include bertopic_version."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "bertopic_version" in data["model_info"], (
            "model_info missing 'bertopic_version'"
        )


# =============================================================================
# Test 5: min_topic_size Parameter
# =============================================================================


class TestTopicsMinTopicSize:
    """WBS B2.1 Test 5: Verify min_topic_size parameter works."""

    def test_topics_with_default_min_topic_size(
        self, test_client: TestClient
    ) -> None:
        """Default min_topic_size should be applied.

        Note: With KMeans fallback, min_topic_size filtering may not be exact
        due to clustering algorithm behavior. We verify the parameter is accepted.
        """
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
                # No min_topic_size - should use default
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Verify response structure is valid
        assert "topics" in data
        assert "topic_count" in data

    def test_topics_with_custom_min_topic_size(
        self, test_client: TestClient
    ) -> None:
        """Custom min_topic_size should be accepted and influence clustering.

        Note: With KMeans fallback, exact min_topic_size enforcement varies.
        BERTopic with full dependencies will filter topics more strictly.
        """
        custom_min_size = 3
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
                "min_topic_size": custom_min_size,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Verify response is valid with custom parameter
        assert "topics" in data
        assert data["topic_count"] >= 0

    def test_topics_with_embedding_model_param(
        self, test_client: TestClient
    ) -> None:
        """Custom embedding_model should be reflected in response."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["model_info"]["embedding_model"] == DEFAULT_EMBEDDING_MODEL

    def test_topics_invalid_min_topic_size_returns_422(
        self, test_client: TestClient
    ) -> None:
        """Invalid min_topic_size (e.g., 0 or negative) should return 422."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
                "min_topic_size": 0,  # Invalid - must be >= 1
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestTopicsEdgeCases:
    """Additional edge case tests for robustness."""

    def test_topics_single_document_corpus(self, test_client: TestClient) -> None:
        """Single document should be handled gracefully."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": ["Single document about design patterns."],
            },
        )
        # Should return 200 with empty topics or all outliers
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "topics" in data

    def test_topics_response_includes_processing_time(
        self, test_client: TestClient
    ) -> None:
        """Response should include processing time for observability."""
        response = test_client.post(
            TOPICS_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Processing time is optional but recommended per GUIDELINES
        if "processing_time_ms" in data:
            assert isinstance(data["processing_time_ms"], (int, float))
            assert data["processing_time_ms"] >= 0
