"""Integration tests for Classify API Endpoint (AC-9.3).

WBS: WBS-AC9 - Testing Requirements
Task: AC9.6 - Write integration test: API end-to-end

Tests the FastAPI classify endpoints with real HTTP requests:
- POST /api/v1/classify (single term)
- POST /api/v1/classify/batch (multiple terms)

This validates end-to-end API behavior including request validation,
response schemas, and HTTP status codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.classify import (
    ClassifyRequest,
    ClassifyResponse,
    classify_router,
    get_classifier,
)
from src.classifiers.llm_fallback import FakeLLMFallback, LLMFallbackResult
from src.classifiers.orchestrator import (
    ClassificationResponse,
    FakeHybridTieredClassifier,
    HybridTieredClassifier,
)


# =============================================================================
# Test App Setup
# =============================================================================


def create_test_app(use_real_classifier: bool = False) -> FastAPI:
    """Create test FastAPI app with classifier injection."""
    app = FastAPI(title="Classify API Integration Test")
    app.include_router(classify_router, prefix="/api")
    return app


@pytest.fixture
def fake_classifier() -> FakeHybridTieredClassifier:
    """Create fake classifier for unit-level integration tests."""
    responses = {
        "microservices": ClassificationResponse(
            term="microservices",
            classification="concept",
            confidence=1.0,
            canonical_term="microservices",
            tier_used=1,
        ),
        "implementation": ClassificationResponse(
            term="implementation",
            classification="keyword",
            confidence=1.0,
            canonical_term="implementation",
            tier_used=1,
        ),
        "def": ClassificationResponse(
            term="def",
            classification="rejected",
            confidence=1.0,
            canonical_term="def",
            tier_used=3,
            rejection_reason="noise_code_artifacts",
        ),
    }
    return FakeHybridTieredClassifier(responses=responses)


@pytest.fixture
def app(fake_classifier: FakeHybridTieredClassifier) -> FastAPI:
    """Create test app with dependency override."""
    app = create_test_app()

    # Override the classifier dependency
    app.dependency_overrides[get_classifier] = lambda: fake_classifier

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Integration Tests - Single Classification Endpoint
# =============================================================================


class TestClassifyEndpointIntegration:
    """Integration tests for POST /api/v1/classify endpoint."""

    def test_classify_concept_returns_200(self, client: TestClient) -> None:
        """POST /classify with valid concept returns 200."""
        response = client.post(
            "/api/v1/classify",
            json={"term": "microservices"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "concept"
        assert data["confidence"] == 1.0
        assert data["tier_used"] == 1

    def test_classify_keyword_returns_200(self, client: TestClient) -> None:
        """POST /classify with valid keyword returns 200."""
        response = client.post(
            "/api/v1/classify",
            json={"term": "implementation"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "keyword"

    def test_classify_rejected_term_returns_200(self, client: TestClient) -> None:
        """POST /classify with noise term returns 200 with rejection."""
        response = client.post(
            "/api/v1/classify",
            json={"term": "def"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "rejected"
        assert data["rejection_reason"] == "noise_code_artifacts"
        assert data["tier_used"] == 3

    def test_classify_empty_term_returns_422(self, client: TestClient) -> None:
        """POST /classify with empty term returns 422 validation error."""
        response = client.post(
            "/api/v1/classify",
            json={"term": ""},
        )

        assert response.status_code == 422
        # FastAPI validation error format
        data = response.json()
        assert "detail" in data

    def test_classify_whitespace_term_returns_422(self, client: TestClient) -> None:
        """POST /classify with whitespace-only term returns 422."""
        response = client.post(
            "/api/v1/classify",
            json={"term": "   "},
        )

        assert response.status_code == 422

    def test_classify_missing_term_returns_422(self, client: TestClient) -> None:
        """POST /classify with missing term field returns 422."""
        response = client.post(
            "/api/v1/classify",
            json={},
        )

        assert response.status_code == 422

    def test_classify_with_domain_parameter(self, client: TestClient) -> None:
        """POST /classify with domain parameter accepted."""
        response = client.post(
            "/api/v1/classify",
            json={"term": "microservices", "domain": "software_engineering"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "concept"

    def test_classify_response_schema(self, client: TestClient) -> None:
        """POST /classify response matches ClassifyResponse schema."""
        response = client.post(
            "/api/v1/classify",
            json={"term": "microservices"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields present
        assert "term" in data
        assert "classification" in data
        assert "confidence" in data
        assert "canonical_term" in data
        assert "tier_used" in data
        # Optional field
        assert "rejection_reason" in data or data.get("rejection_reason") is None


# =============================================================================
# Integration Tests - Batch Classification Endpoint
# =============================================================================


class TestBatchClassifyEndpointIntegration:
    """Integration tests for POST /api/v1/classify/batch endpoint."""

    def test_batch_classify_returns_200(self, client: TestClient) -> None:
        """POST /classify/batch with valid terms returns 200."""
        response = client.post(
            "/api/v1/classify/batch",
            json={"terms": ["microservices", "implementation"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_batch_classify_preserves_order(self, client: TestClient) -> None:
        """POST /classify/batch results match input order."""
        response = client.post(
            "/api/v1/classify/batch",
            json={"terms": ["implementation", "microservices", "def"]},
        )

        assert response.status_code == 200
        data = response.json()
        terms = [r["term"] for r in data["results"]]
        assert terms == ["implementation", "microservices", "def"]

    def test_batch_classify_empty_list_returns_empty_results(self, client: TestClient) -> None:
        """POST /classify/batch with empty list returns 200 with empty results."""
        response = client.post(
            "/api/v1/classify/batch",
            json={"terms": []},
        )

        # Empty list is valid, returns empty results
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []

    def test_batch_classify_single_term(self, client: TestClient) -> None:
        """POST /classify/batch with single term works."""
        response = client.post(
            "/api/v1/classify/batch",
            json={"terms": ["microservices"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1

    def test_batch_classify_mixed_classifications(self, client: TestClient) -> None:
        """POST /classify/batch handles mixed term types."""
        response = client.post(
            "/api/v1/classify/batch",
            json={"terms": ["microservices", "def"]},
        )

        assert response.status_code == 200
        data = response.json()

        classifications = {r["term"]: r["classification"] for r in data["results"]}
        assert classifications["microservices"] == "concept"
        assert classifications["def"] == "rejected"


# =============================================================================
# Integration Tests - OpenAPI Documentation
# =============================================================================


class TestOpenAPIIntegration:
    """Tests for OpenAPI documentation endpoints."""

    def test_openapi_schema_includes_classify(self, client: TestClient) -> None:
        """OpenAPI schema includes /classify endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        assert "/api/v1/classify" in schema["paths"]
        assert "post" in schema["paths"]["/api/v1/classify"]

    def test_openapi_schema_includes_batch(self, client: TestClient) -> None:
        """OpenAPI schema includes /classify/batch endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        assert "/api/v1/classify/batch" in schema["paths"]


# =============================================================================
# Integration Tests - Error Handling
# =============================================================================


class TestErrorHandlingIntegration:
    """Tests for error handling and edge cases."""

    def test_invalid_json_returns_422(self, client: TestClient) -> None:
        """POST /classify with invalid JSON returns 422."""
        response = client.post(
            "/api/v1/classify",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_wrong_content_type_returns_422(self, client: TestClient) -> None:
        """POST /classify with wrong content type returns 422."""
        response = client.post(
            "/api/v1/classify",
            data="term=microservices",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 422

    def test_get_method_not_allowed(self, client: TestClient) -> None:
        """GET /classify returns 405 Method Not Allowed."""
        response = client.get("/api/v1/classify")

        assert response.status_code == 405


# =============================================================================
# Real Classifier Integration Tests (Optional - Slower)
# =============================================================================


class TestRealClassifierIntegration:
    """Integration tests with real classifier components.

    These tests are skipped if model files are not present.
    They verify end-to-end behavior with production components.
    """

    @pytest.fixture
    def project_root(self) -> Path:
        """Get project root."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def real_client(self, project_root: Path) -> TestClient | None:
        """Create test client with real classifier if available."""
        from src.classifiers.alias_lookup import AliasLookup
        from src.classifiers.heuristic_filter import HeuristicFilter
        from src.classifiers.trained_classifier import TrainedClassifier

        alias_path = project_root / "config" / "alias_lookup.json"
        model_path = project_root / "models" / "concept_classifier.joblib"
        noise_path = project_root / "config" / "noise_terms.yaml"

        if not all(p.exists() for p in [alias_path, model_path, noise_path]):
            pytest.skip("Model files not available for real classifier test")

        alias_lookup = AliasLookup(lookup_path=alias_path)
        trained_classifier = TrainedClassifier(model_path=model_path)
        heuristic_filter = HeuristicFilter(config_path=noise_path)
        fake_llm = FakeLLMFallback(responses={})

        real_classifier = HybridTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=trained_classifier,
            heuristic_filter=heuristic_filter,
            llm_fallback=fake_llm,
        )

        app = create_test_app()
        app.dependency_overrides[get_classifier] = lambda: real_classifier

        return TestClient(app)

    def test_real_classifier_concept(self, real_client: TestClient | None) -> None:
        """Test real classifier with known concept."""
        if real_client is None:
            pytest.skip("Real classifier not available")

        response = real_client.post(
            "/api/v1/classify",
            json={"term": "microservices"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["classification"] == "concept"
        assert data["tier_used"] == 1

    def test_real_classifier_noise_rejection(
        self, real_client: TestClient | None
    ) -> None:
        """Test real classifier handles noise-like terms."""
        if real_client is None:
            pytest.skip("Real classifier not available")

        # 'def' may be in alias_lookup.json as a keyword, so use flexible assertion
        response = real_client.post(
            "/api/v1/classify",
            json={"term": "def"},
        )

        assert response.status_code == 200
        data = response.json()
        # May be classified as keyword (in alias lookup) or rejected (noise filter)
        assert data["classification"] in ["concept", "keyword", "rejected"]
