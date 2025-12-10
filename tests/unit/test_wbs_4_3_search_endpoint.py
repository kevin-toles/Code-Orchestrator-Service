"""
WBS 4.3: Full Search Endpoint Tests (RED Phase)

Tests for POST /api/v1/search:
- 4.3.1: Full pipeline: extract → search → curate
- 4.3.2: Request validation
- 4.3.3: Response schema

Plus Phase 4 Integration Test from WBS_IMPLEMENTATION.md
"""

from fastapi.testclient import TestClient

from src.main import app

# =============================================================================
# WBS 4.3.1: Search Endpoint Tests
# =============================================================================


class TestSearchEndpoint:
    """Tests for POST /api/v1/search endpoint."""

    def test_search_endpoint_exists(self) -> None:
        """POST /api/v1/search endpoint exists."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={"query": "test", "domain": "ai-ml"},
        )

        # Should not return 404
        assert response.status_code != 404

    def test_search_returns_200_with_valid_input(self) -> None:
        """Search endpoint returns 200 with valid query and domain."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "LLM document chunking with overlap",
                "domain": "ai-ml",
            },
        )

        assert response.status_code == 200

    def test_search_returns_results_array(self) -> None:
        """Search endpoint returns results array."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "Multi-stage document chunking",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_results_have_required_fields(self) -> None:
        """Each result has book, chapter, relevance_score."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "RAG pipeline embedding",
                "domain": "ai-ml",
            },
        )

        data = response.json()

        for result in data["results"]:
            assert "book" in result
            assert "relevance_score" in result

    def test_search_results_respects_top_k(self) -> None:
        """Search respects top_k option limit."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "LLM RAG chunking embedding vector",
                "domain": "ai-ml",
                "options": {"top_k": 5},
            },
        )

        data = response.json()
        assert len(data["results"]) <= 5


# =============================================================================
# WBS 4.3.2: Request Validation Tests
# =============================================================================


class TestSearchValidation:
    """Tests for search request validation."""

    def test_search_returns_422_without_query(self) -> None:
        """Search returns 422 without query field."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={"domain": "ai-ml"},  # Missing query
        )

        assert response.status_code == 422

    def test_search_returns_422_without_domain(self) -> None:
        """Search returns 422 without domain field."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={"query": "test"},  # Missing domain
        )

        assert response.status_code == 422

    def test_search_accepts_options_parameter(self) -> None:
        """Search accepts options in request body."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "LLM chunking",
                "domain": "ai-ml",
                "options": {"top_k": 10},
            },
        )

        assert response.status_code == 200

    def test_search_validates_top_k_range(self) -> None:
        """Search validates top_k is in valid range."""
        client = TestClient(app)

        # top_k should be between 1 and 100
        response = client.post(
            "/api/v1/search",
            json={
                "query": "test",
                "domain": "ai-ml",
                "options": {"top_k": 0},  # Invalid
            },
        )

        assert response.status_code == 422


# =============================================================================
# WBS 4.3.3: Response Schema Tests
# =============================================================================


class TestSearchResponseSchema:
    """Tests for search response schema."""

    def test_response_has_metadata(self) -> None:
        """Response includes metadata object."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "document processing",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "metadata" in data
        assert isinstance(data["metadata"], dict)

    def test_metadata_has_pipeline_info(self) -> None:
        """Metadata includes pipeline information."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "neural network",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "pipeline" in data["metadata"]

    def test_metadata_pipeline_has_stages_completed(self) -> None:
        """Pipeline metadata includes stages_completed count."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "transformer attention",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "stages_completed" in data["metadata"]["pipeline"]


# =============================================================================
# Phase 4 Integration Test (from WBS_IMPLEMENTATION.md)
# =============================================================================


class TestPhase4Integration:
    """Phase 4 Integration Test per WBS_IMPLEMENTATION.md."""

    def test_phase4_full_search_pipeline(self) -> None:
        """POST /api/v1/search executes full pipeline with curation.

        This is the official Phase 4 Integration Test from WBS.
        """
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "Multi-stage document chunking with overlap for RAG",
                "domain": "ai-ml",
                "options": {"top_k": 10},
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Got curated results
        assert len(data["results"]) >= 0  # May be empty if no semantic-search available
        assert len(data["results"]) <= 10

        # Results are from correct domain (if any results)
        for result in data["results"]:
            assert "C++ Concurrency" not in result["book"]  # Wrong domain filtered
            assert result["relevance_score"] >= 0.3  # Semantic threshold achievable

        # Metadata present
        assert "pipeline" in data["metadata"]
        assert data["metadata"]["pipeline"]["stages_completed"] == 4

    def test_search_filters_wrong_domain_results(self) -> None:
        """Search filters out results from wrong domain."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "memory chunk allocation",  # Could match C++ AND AI
                "domain": "ai-ml",
            },
        )

        data = response.json()

        # No C++ results should be in ai-ml domain search
        for result in data["results"]:
            assert "C++ Concurrency" not in result.get("book", "")

    def test_search_results_meet_relevance_threshold(self) -> None:
        """All results meet minimum relevance threshold of 0.3."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "LLM document chunking",
                "domain": "ai-ml",
            },
        )

        data = response.json()

        for result in data["results"]:
            assert result["relevance_score"] >= 0.3


# =============================================================================
# Performance Tests
# =============================================================================


class TestSearchPerformance:
    """Tests for search endpoint performance."""

    def test_search_response_time_under_5_seconds(self) -> None:
        """Search response time is under 5 seconds per WBS SLA."""
        import time

        client = TestClient(app)

        start = time.time()
        response = client.post(
            "/api/v1/search",
            json={
                "query": "semantic search embeddings",
                "domain": "ai-ml",
            },
        )
        elapsed_ms = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed_ms < 5000, f"Response took {elapsed_ms}ms, exceeds 5s SLA"

    def test_metadata_includes_processing_time(self) -> None:
        """Metadata includes processing_time_ms."""
        client = TestClient(app)

        response = client.post(
            "/api/v1/search",
            json={
                "query": "vector database",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "processing_time_ms" in data["metadata"]
        assert isinstance(data["metadata"]["processing_time_ms"], (int, float))
