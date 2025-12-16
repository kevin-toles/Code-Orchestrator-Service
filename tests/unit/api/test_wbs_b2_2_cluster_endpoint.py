"""
WBS B2.2: POST /api/v1/cluster Endpoint Tests

TDD RED Phase - Tests written BEFORE implementation.
These tests will FAIL until cluster endpoint is implemented.

Test Plan (per BERTOPIC_INTEGRATION_WBS.md):
1. test_cluster_endpoint_exists - 200/422 not 404
2. test_cluster_empty_corpus_returns_empty - Edge case
3. test_cluster_returns_assignments - Happy path
4. test_cluster_assignment_per_document - One assignment per doc
5. test_cluster_with_chapter_index - Returns chapter metadata
6. test_cluster_with_precomputed_embeddings - Optimization

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

CLUSTER_ENDPOINT: str = "/api/v1/cluster"
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

# Sample chapter index for testing
SAMPLE_CHAPTER_INDEX: list[dict[str, str | int]] = [
    {"book": "patterns.json", "chapter": 1, "title": "Repository Pattern"},
    {"book": "patterns.json", "chapter": 2, "title": "Dependency Injection"},
    {"book": "patterns.json", "chapter": 3, "title": "Factory Pattern"},
    {"book": "testing.json", "chapter": 1, "title": "Unit Testing"},
    {"book": "testing.json", "chapter": 2, "title": "Integration Testing"},
    {"book": "patterns.json", "chapter": 4, "title": "Singleton Pattern"},
    {"book": "principles.json", "chapter": 1, "title": "SOLID Principles"},
    {"book": "architecture.json", "chapter": 1, "title": "Clean Architecture"},
]

# Minimal corpus for edge case testing
MINIMAL_CORPUS: list[str] = [
    "First document about software patterns.",
    "Second document about design principles.",
]

MINIMAL_CHAPTER_INDEX: list[dict[str, str | int]] = [
    {"book": "test.json", "chapter": 1, "title": "Patterns"},
    {"book": "test.json", "chapter": 2, "title": "Principles"},
]


# =============================================================================
# Test 1: Endpoint Exists (returns 200/422, not 404)
# =============================================================================


class TestClusterEndpointExists:
    """WBS B2.2 Test 1: Verify endpoint is registered and accessible."""

    def test_cluster_endpoint_returns_200_with_valid_request(
        self, test_client: TestClient
    ) -> None:
        """POST /api/v1/cluster should return 200 with valid corpus."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        # Accept 200 (success) - not 404 (not found)
        assert response.status_code != status.HTTP_404_NOT_FOUND, (
            f"Endpoint {CLUSTER_ENDPOINT} not registered (got 404)"
        )
        assert response.status_code == status.HTTP_200_OK

    def test_cluster_endpoint_returns_422_with_invalid_request(
        self, test_client: TestClient
    ) -> None:
        """POST /api/v1/cluster should return 422 with invalid input."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": [],  # Empty corpus should fail validation
                "chapter_index": [],
            },
        )
        # Accept 422 (validation error) - not 404 (not found)
        assert response.status_code != status.HTTP_404_NOT_FOUND, (
            f"Endpoint {CLUSTER_ENDPOINT} not registered (got 404)"
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Test 2: Empty Corpus Edge Case
# =============================================================================


class TestClusterEmptyCorpus:
    """WBS B2.2 Test 2: Handle empty corpus gracefully."""

    def test_cluster_empty_corpus_returns_validation_error(
        self, test_client: TestClient
    ) -> None:
        """Empty corpus should return 422 validation error."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": [],
                "chapter_index": [],
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_cluster_mismatched_lengths_returns_422(
        self, test_client: TestClient
    ) -> None:
        """Corpus and chapter_index with different lengths should return 422."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,  # 2 items
                "chapter_index": [MINIMAL_CHAPTER_INDEX[0]],  # 1 item
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Test 3: Happy Path - Returns Assignments
# =============================================================================


class TestClusterReturnsAssignments:
    """WBS B2.2 Test 3: Verify endpoint returns topic assignments."""

    def test_cluster_returns_assignments_list(self, test_client: TestClient) -> None:
        """POST /api/v1/cluster should return list of assignments."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
                "chapter_index": SAMPLE_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Must have 'assignments' key with list
        assert "assignments" in data, "Response missing 'assignments' key"
        assert isinstance(data["assignments"], list), "'assignments' must be a list"

    def test_cluster_returns_topics_list(self, test_client: TestClient) -> None:
        """Response should include topics list."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
                "chapter_index": SAMPLE_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "topics" in data, "Response missing 'topics' key"
        assert isinstance(data["topics"], list), "'topics' must be a list"

    def test_cluster_returns_topic_count(self, test_client: TestClient) -> None:
        """Response should include topic_count field."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
                "chapter_index": SAMPLE_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "topic_count" in data, "Response missing 'topic_count' key"
        assert isinstance(data["topic_count"], int), "'topic_count' must be integer"


# =============================================================================
# Test 4: One Assignment Per Document
# =============================================================================


class TestClusterAssignmentPerDocument:
    """WBS B2.2 Test 4: Verify one assignment per document."""

    def test_cluster_assignment_count_matches_corpus(
        self, test_client: TestClient
    ) -> None:
        """Number of assignments should match corpus length."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": SAMPLE_CORPUS,
                "chapter_index": SAMPLE_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert len(data["assignments"]) == len(SAMPLE_CORPUS), (
            f"Expected {len(SAMPLE_CORPUS)} assignments, got {len(data['assignments'])}"
        )

    def test_cluster_minimal_corpus_returns_correct_count(
        self, test_client: TestClient
    ) -> None:
        """Minimal corpus should return matching assignment count."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert len(data["assignments"]) == len(MINIMAL_CORPUS)


# =============================================================================
# Test 5: Chapter Index Metadata
# =============================================================================


class TestClusterWithChapterIndex:
    """WBS B2.2 Test 5: Verify chapter metadata is included in assignments."""

    def test_assignment_includes_book_field(self, test_client: TestClient) -> None:
        """Each assignment should include 'book' from chapter_index."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for assignment in data["assignments"]:
            assert "book" in assignment, "Assignment missing 'book' field"

    def test_assignment_includes_chapter_field(self, test_client: TestClient) -> None:
        """Each assignment should include 'chapter' from chapter_index."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for assignment in data["assignments"]:
            assert "chapter" in assignment, "Assignment missing 'chapter' field"

    def test_assignment_includes_title_field(self, test_client: TestClient) -> None:
        """Each assignment should include 'title' from chapter_index."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for assignment in data["assignments"]:
            assert "title" in assignment, "Assignment missing 'title' field"

    def test_assignment_includes_topic_fields(self, test_client: TestClient) -> None:
        """Each assignment should include topic_id, topic_name, confidence."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for assignment in data["assignments"]:
            assert "topic_id" in assignment, "Assignment missing 'topic_id'"
            assert "topic_name" in assignment, "Assignment missing 'topic_name'"
            assert "confidence" in assignment, "Assignment missing 'confidence'"

    def test_assignment_preserves_chapter_metadata(
        self, test_client: TestClient
    ) -> None:
        """Assignment metadata should match input chapter_index."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # First assignment should match first chapter_index entry
        first_assignment = data["assignments"][0]
        first_chapter = MINIMAL_CHAPTER_INDEX[0]
        
        assert first_assignment["book"] == first_chapter["book"]
        assert first_assignment["chapter"] == first_chapter["chapter"]
        assert first_assignment["title"] == first_chapter["title"]


# =============================================================================
# Test 6: Precomputed Embeddings (Optional)
# =============================================================================


class TestClusterWithPrecomputedEmbeddings:
    """WBS B2.2 Test 6: Verify precomputed embeddings are accepted."""

    def test_cluster_accepts_null_embeddings(self, test_client: TestClient) -> None:
        """Endpoint should accept null embeddings (compute internally)."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
                "embeddings": None,
            },
        )
        assert response.status_code == status.HTTP_200_OK

    def test_cluster_accepts_embedding_model_param(
        self, test_client: TestClient
    ) -> None:
        """Endpoint should accept embedding_model parameter."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
            },
        )
        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestClusterEdgeCases:
    """Additional edge case tests for robustness."""

    def test_cluster_single_document_handled(self, test_client: TestClient) -> None:
        """Single document should be handled gracefully."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": ["Single document about patterns."],
                "chapter_index": [{"book": "test.json", "chapter": 1, "title": "Ch1"}],
            },
        )
        # Should return 200 with one assignment
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["assignments"]) == 1

    def test_cluster_response_includes_processing_time(
        self, test_client: TestClient
    ) -> None:
        """Response should include processing time for observability."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Processing time is recommended per GUIDELINES
        if "processing_time_ms" in data:
            assert isinstance(data["processing_time_ms"], (int, float))
            assert data["processing_time_ms"] >= 0

    def test_cluster_confidence_is_valid_range(self, test_client: TestClient) -> None:
        """Confidence scores should be between 0.0 and 1.0."""
        response = test_client.post(
            CLUSTER_ENDPOINT,
            json={
                "corpus": MINIMAL_CORPUS,
                "chapter_index": MINIMAL_CHAPTER_INDEX,
            },
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for assignment in data["assignments"]:
            confidence = assignment["confidence"]
            assert 0.0 <= confidence <= 1.0, (
                f"Confidence {confidence} outside valid range [0.0, 1.0]"
            )
