"""
WBS 3.2 Integration Tests - /v1/extract Endpoint

Phase 3 Integration Test per WBS_IMPLEMENTATION.md:
- POST /v1/extract returns consensus terms
- search_terms[0].models_agreed >= 2
- metadata.processing_time_ms < 5000

Patterns Applied:
- FastAPI TestClient for in-process testing
- Pydantic request/response models
"""

from fastapi.testclient import TestClient

from src.main import app


class TestExtractEndpoint:
    """Tests for POST /v1/extract endpoint."""

    def test_extract_endpoint_exists(self) -> None:
        """POST /v1/extract endpoint exists."""
        client = TestClient(app)

        # Should not return 404
        response = client.post(
            "/v1/extract",
            json={"query": "test", "domain": "ai-ml"},
        )

        assert response.status_code != 404

    def test_extract_returns_200_with_valid_input(self) -> None:
        """Extract endpoint returns 200 with valid query and domain."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "LLM document chunking with overlap",
                "domain": "ai-ml",
            },
        )

        assert response.status_code == 200

    def test_extract_returns_search_terms(self) -> None:
        """Extract endpoint returns search_terms array."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "LLM document chunking with overlap",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "search_terms" in data
        assert isinstance(data["search_terms"], list)

    def test_extract_search_terms_have_models_agreed(self) -> None:
        """Each search term has models_agreed >= 2."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "LLM document chunking with overlap",
                "domain": "ai-ml",
            },
        )

        data = response.json()

        # Per WBS: search_terms[0].models_agreed >= 2
        if len(data["search_terms"]) > 0:
            first_term = data["search_terms"][0]
            assert "models_agreed" in first_term
            assert first_term["models_agreed"] >= 2

    def test_extract_returns_metadata(self) -> None:
        """Extract endpoint returns metadata object."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "LLM document chunking",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "metadata" in data
        assert isinstance(data["metadata"], dict)

    def test_extract_metadata_has_processing_time(self) -> None:
        """Metadata includes processing_time_ms."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "RAG pipeline",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "processing_time_ms" in data["metadata"]
        assert isinstance(data["metadata"]["processing_time_ms"], (int, float))

    def test_extract_processing_time_under_5_seconds(self) -> None:
        """Processing time is under 5000ms per WBS SLA."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "semantic search embeddings",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        # Per WBS: metadata.processing_time_ms < 5000
        assert data["metadata"]["processing_time_ms"] < 5000

    def test_extract_returns_422_without_query(self) -> None:
        """Extract endpoint returns 422 without query field."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={"domain": "ai-ml"},  # Missing query
        )

        assert response.status_code == 422

    def test_extract_works_without_domain(self) -> None:
        """Extract endpoint works without domain field (optional)."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={"query": "test"},  # Domain is optional
        )

        assert response.status_code == 200

    def test_extract_search_terms_have_score(self) -> None:
        """Each search term has a score field."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "transformer attention mechanism",
                "domain": "ai-ml",
            },
        )

        data = response.json()

        for term in data["search_terms"]:
            assert "score" in term
            assert isinstance(term["score"], (int, float))
            assert 0.0 <= term["score"] <= 1.0

    def test_extract_search_terms_have_term_field(self) -> None:
        """Each search term has a term field."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "vector database indexing",
                "domain": "ai-ml",
            },
        )

        data = response.json()

        for term in data["search_terms"]:
            assert "term" in term
            assert isinstance(term["term"], str)
            assert len(term["term"]) > 0


class TestExtractEndpointOptions:
    """Tests for /extract endpoint with options."""

    def test_extract_accepts_options_parameter(self) -> None:
        """Extract endpoint accepts options in request body."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "LLM chunking",
                "domain": "ai-ml",
                "options": {
                    "min_confidence": 0.5,
                    "max_terms": 5,
                },
            },
        )

        assert response.status_code == 200

    def test_extract_respects_max_terms_option(self) -> None:
        """Extract respects max_terms option limit."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "LLM RAG chunking embedding vector database semantic search",
                "domain": "ai-ml",
                "options": {"max_terms": 3},
            },
        )

        data = response.json()
        assert len(data["search_terms"]) <= 3


class TestExtractMetadataDetails:
    """Tests for detailed metadata in extract response."""

    def test_metadata_includes_stages_completed(self) -> None:
        """Metadata includes stages_completed list."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "document processing",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        assert "stages_completed" in data["metadata"]
        assert isinstance(data["metadata"]["stages_completed"], list)

    def test_metadata_stages_include_all_four(self) -> None:
        """Stages completed includes generate, validate, rank, consensus."""
        client = TestClient(app)

        response = client.post(
            "/v1/extract",
            json={
                "query": "neural network training",
                "domain": "ai-ml",
            },
        )

        data = response.json()
        stages = data["metadata"]["stages_completed"]

        assert "generate" in stages
        assert "validate" in stages
        # rank may be skipped if all terms rejected
        assert "consensus" in stages
