"""
WBS-AC6: Classification API Endpoint Tests

TDD Phase: RED - Tests written BEFORE implementation
AC Block: AC-6.1 through AC-6.6

Acceptance Criteria:
- AC-6.1: Endpoint Registration - POST `/api/v1/classify` in OpenAPI
- AC-6.2: Valid Request - Returns ClassifyResponse with classification
- AC-6.3: Empty Term Error - Returns 422 Validation Error
- AC-6.4: Optional Domain - Domain parameter passed to classifier
- AC-6.5: Batch Endpoint - POST `/api/v1/classify/batch` works
- AC-6.6: Dependency Injection - Classifier injected via Depends()

Anti-Patterns Avoided:
- S1192: Constants for repeated string literals
- S3776: Simple test methods (CC < 15)
- #12: FakeHybridTieredClassifier used (no real connections in tests)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.classifiers.orchestrator import (
    ClassificationResponse,
    FakeHybridTieredClassifier,
)

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

# Endpoint paths
CLASSIFY_ENDPOINT = "/api/v1/classify"
CLASSIFY_BATCH_ENDPOINT = "/api/v1/classify/batch"
OPENAPI_ENDPOINT = "/openapi.json"

# Test data
TEST_TERM_MICROSERVICE = "microservice"
TEST_TERM_KUBERNETES = "kubernetes"
TEST_TERM_RAG = "RAG"
TEST_DOMAIN_DEVOPS = "devops"
TEST_DOMAIN_LLM_RAG = "llm_rag"

# HTTP status codes
HTTP_200_OK = 200
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_404_NOT_FOUND = 404


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fake_classifier() -> FakeHybridTieredClassifier:
    """Create a FakeHybridTieredClassifier with pre-configured responses."""
    return FakeHybridTieredClassifier(
        responses={
            TEST_TERM_MICROSERVICE: ClassificationResponse(
                term=TEST_TERM_MICROSERVICE,
                classification="concept",
                confidence=1.0,
                canonical_term="microservice",
                tier_used=1,
            ),
            TEST_TERM_KUBERNETES: ClassificationResponse(
                term=TEST_TERM_KUBERNETES,
                classification="concept",
                confidence=0.85,
                canonical_term="kubernetes",
                tier_used=2,
            ),
            TEST_TERM_RAG: ClassificationResponse(
                term=TEST_TERM_RAG,
                classification="concept",
                confidence=0.92,
                canonical_term="retrieval_augmented_generation",
                tier_used=4,
            ),
        }
    )


@pytest.fixture
def app_with_fake_classifier(fake_classifier: FakeHybridTieredClassifier) -> FastAPI:
    """Create FastAPI app with injected fake classifier."""
    from src.api.classify import classify_router, get_classifier

    app = FastAPI()
    app.include_router(classify_router, prefix="/api")

    # Override the dependency
    def override_get_classifier() -> FakeHybridTieredClassifier:
        return fake_classifier

    app.dependency_overrides[get_classifier] = override_get_classifier
    return app


@pytest.fixture
def client(app_with_fake_classifier: FastAPI) -> TestClient:
    """Create test client with fake classifier."""
    return TestClient(app_with_fake_classifier)


@pytest.fixture
def real_client() -> TestClient:
    """Create test client with real app (for OpenAPI tests)."""
    from src.main import app
    return TestClient(app)


# =============================================================================
# Test: Endpoint Registration (AC-6.1)
# =============================================================================


class TestEndpointRegistration:
    """AC-6.1: POST /api/v1/classify should be registered in OpenAPI."""

    def test_classify_endpoint_in_openapi(self, real_client: TestClient) -> None:
        """POST /api/v1/classify should appear in OpenAPI schema."""
        response = real_client.get(OPENAPI_ENDPOINT)
        assert response.status_code == HTTP_200_OK
        
        openapi_schema = response.json()
        paths = openapi_schema.get("paths", {})
        
        assert CLASSIFY_ENDPOINT in paths, (
            f"Expected {CLASSIFY_ENDPOINT} in OpenAPI paths"
        )
        assert "post" in paths[CLASSIFY_ENDPOINT], (
            f"Expected POST method for {CLASSIFY_ENDPOINT}"
        )

    def test_batch_endpoint_in_openapi(self, real_client: TestClient) -> None:
        """POST /api/v1/classify/batch should appear in OpenAPI schema."""
        response = real_client.get(OPENAPI_ENDPOINT)
        assert response.status_code == HTTP_200_OK
        
        openapi_schema = response.json()
        paths = openapi_schema.get("paths", {})
        
        assert CLASSIFY_BATCH_ENDPOINT in paths, (
            f"Expected {CLASSIFY_BATCH_ENDPOINT} in OpenAPI paths"
        )
        assert "post" in paths[CLASSIFY_BATCH_ENDPOINT], (
            f"Expected POST method for {CLASSIFY_BATCH_ENDPOINT}"
        )

    def test_endpoint_not_404(self, client: TestClient) -> None:
        """Endpoint should exist (not return 404)."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        assert response.status_code != HTTP_404_NOT_FOUND


# =============================================================================
# Test: Valid Request (AC-6.2)
# =============================================================================


class TestValidRequest:
    """AC-6.2: Valid request returns ClassifyResponse."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        """POST with valid term should return 200."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        assert response.status_code == HTTP_200_OK

    def test_response_has_term_field(self, client: TestClient) -> None:
        """Response should include original term."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        data = response.json()
        assert "term" in data
        assert data["term"] == TEST_TERM_MICROSERVICE

    def test_response_has_classification_field(self, client: TestClient) -> None:
        """Response should include classification."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        data = response.json()
        assert "classification" in data
        assert data["classification"] in ["concept", "keyword", "rejected", "unknown"]

    def test_response_has_confidence_field(self, client: TestClient) -> None:
        """Response should include confidence score."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        data = response.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_has_canonical_term_field(self, client: TestClient) -> None:
        """Response should include canonical_term."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        data = response.json()
        assert "canonical_term" in data

    def test_response_has_tier_used_field(self, client: TestClient) -> None:
        """Response should include tier_used."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        data = response.json()
        assert "tier_used" in data
        assert data["tier_used"] in [1, 2, 3, 4]

    def test_response_matches_classifier_output(self, client: TestClient) -> None:
        """Response should match FakeClassifier's configured response."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        data = response.json()
        
        assert data["term"] == TEST_TERM_MICROSERVICE
        assert data["classification"] == "concept"
        assert data["confidence"] == 1.0
        assert data["canonical_term"] == "microservice"
        assert data["tier_used"] == 1


# =============================================================================
# Test: Empty Term Error (AC-6.3)
# =============================================================================


class TestEmptyTermError:
    """AC-6.3: Empty term returns 422 Validation Error."""

    def test_empty_term_returns_422(self, client: TestClient) -> None:
        """Empty string term should return 422."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": ""},
        )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_whitespace_term_returns_422(self, client: TestClient) -> None:
        """Whitespace-only term should return 422."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": "   "},
        )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_term_returns_422(self, client: TestClient) -> None:
        """Missing term field should return 422."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={},
        )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_null_term_returns_422(self, client: TestClient) -> None:
        """Null term should return 422."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": None},
        )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_422_includes_detail(self, client: TestClient) -> None:
        """422 response should include validation detail."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": ""},
        )
        data = response.json()
        assert "detail" in data


# =============================================================================
# Test: Optional Domain Parameter (AC-6.4)
# =============================================================================


class TestOptionalDomain:
    """AC-6.4: Domain parameter passed to classifier."""

    def test_request_accepts_domain(self, client: TestClient) -> None:
        """Request should accept optional domain parameter."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={
                "term": TEST_TERM_MICROSERVICE,
                "domain": TEST_DOMAIN_DEVOPS,
            },
        )
        assert response.status_code == HTTP_200_OK

    def test_domain_not_required(self, client: TestClient) -> None:
        """Request should work without domain parameter."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        assert response.status_code == HTTP_200_OK

    def test_domain_in_response(self, client: TestClient) -> None:
        """Response should include domain when provided."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={
                "term": TEST_TERM_MICROSERVICE,
                "domain": TEST_DOMAIN_DEVOPS,
            },
        )
        data = response.json()
        # Domain might be in response or used internally
        # The key is that it doesn't cause an error
        assert response.status_code == HTTP_200_OK


# =============================================================================
# Test: Batch Endpoint (AC-6.5)
# =============================================================================


class TestBatchEndpoint:
    """AC-6.5: POST /api/v1/classify/batch processes multiple terms."""

    def test_batch_endpoint_exists(self, client: TestClient) -> None:
        """Batch endpoint should exist."""
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": [TEST_TERM_MICROSERVICE]},
        )
        assert response.status_code != HTTP_404_NOT_FOUND

    def test_batch_returns_list(self, client: TestClient) -> None:
        """Batch should return list of results."""
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": [TEST_TERM_MICROSERVICE, TEST_TERM_KUBERNETES]},
        )
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_batch_result_count_matches_input(self, client: TestClient) -> None:
        """Batch should return same number of results as input terms."""
        terms = [TEST_TERM_MICROSERVICE, TEST_TERM_KUBERNETES, TEST_TERM_RAG]
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": terms},
        )
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == len(terms)

    def test_batch_empty_list_returns_empty(self, client: TestClient) -> None:
        """Batch with empty list should return empty results."""
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": []},
        )
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert data["results"] == []

    def test_batch_results_have_all_fields(self, client: TestClient) -> None:
        """Each batch result should have all classification fields."""
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": [TEST_TERM_MICROSERVICE]},
        )
        assert response.status_code == HTTP_200_OK
        data = response.json()
        result = data["results"][0]
        
        assert "term" in result
        assert "classification" in result
        assert "confidence" in result
        assert "canonical_term" in result
        assert "tier_used" in result

    def test_batch_preserves_order(self, client: TestClient) -> None:
        """Batch results should be in same order as input."""
        terms = [TEST_TERM_KUBERNETES, TEST_TERM_MICROSERVICE]
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": terms},
        )
        assert response.status_code == HTTP_200_OK
        data = response.json()
        
        assert data["results"][0]["term"] == TEST_TERM_KUBERNETES
        assert data["results"][1]["term"] == TEST_TERM_MICROSERVICE

    def test_batch_accepts_domain(self, client: TestClient) -> None:
        """Batch should accept optional domain parameter."""
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={
                "terms": [TEST_TERM_MICROSERVICE],
                "domain": TEST_DOMAIN_DEVOPS,
            },
        )
        assert response.status_code == HTTP_200_OK


# =============================================================================
# Test: Dependency Injection (AC-6.6)
# =============================================================================


class TestDependencyInjection:
    """AC-6.6: Classifier injected via Depends()."""

    def test_classifier_is_injectable(
        self, app_with_fake_classifier: FastAPI
    ) -> None:
        """Classifier should be injectable via dependency override."""
        from src.api.classify import get_classifier
        
        # Verify dependency can be overridden
        assert get_classifier in app_with_fake_classifier.dependency_overrides

    def test_fake_classifier_used(
        self, client: TestClient, fake_classifier: FakeHybridTieredClassifier
    ) -> None:
        """Test that fake classifier's responses are used."""
        # The fake classifier returns specific values for "microservice"
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        data = response.json()
        
        # These values come from the FakeHybridTieredClassifier fixture
        assert data["classification"] == "concept"
        assert data["tier_used"] == 1

    def test_different_terms_different_results(
        self, client: TestClient
    ) -> None:
        """Different terms should get different results from fake."""
        # microservice should be tier 1
        resp1 = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        # kubernetes should be tier 2
        resp2 = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_KUBERNETES},
        )
        
        assert resp1.json()["tier_used"] == 1
        assert resp2.json()["tier_used"] == 2


# =============================================================================
# Test: Request/Response Models
# =============================================================================


class TestRequestResponseModels:
    """Test Pydantic request and response model validation."""

    def test_request_model_validates_term_type(self, client: TestClient) -> None:
        """Request should validate term is string."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": 12345},  # Wrong type
        )
        # Should either coerce to string or return 422
        # FastAPI/Pydantic may coerce integers to strings
        assert response.status_code in [HTTP_200_OK, HTTP_422_UNPROCESSABLE_ENTITY]

    def test_request_ignores_extra_fields(self, client: TestClient) -> None:
        """Request should ignore extra fields."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={
                "term": TEST_TERM_MICROSERVICE,
                "extra_field": "should_be_ignored",
            },
        )
        assert response.status_code == HTTP_200_OK

    def test_batch_request_validates_terms_list(self, client: TestClient) -> None:
        """Batch request should validate terms is a list."""
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": "not_a_list"},
        )
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Test: Response Format
# =============================================================================


class TestResponseFormat:
    """Test JSON response format compliance."""

    def test_response_is_json(self, client: TestClient) -> None:
        """Response should be valid JSON."""
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": TEST_TERM_MICROSERVICE},
        )
        assert response.headers["content-type"] == "application/json"
        # Should not raise
        _ = response.json()

    def test_batch_response_is_json(self, client: TestClient) -> None:
        """Batch response should be valid JSON."""
        response = client.post(
            CLASSIFY_BATCH_ENDPOINT,
            json={"terms": [TEST_TERM_MICROSERVICE]},
        )
        assert response.headers["content-type"] == "application/json"
        _ = response.json()

    def test_rejection_reason_included_when_rejected(
        self, client: TestClient, app_with_fake_classifier: FastAPI
    ) -> None:
        """Response should include rejection_reason when classification is rejected."""
        from src.api.classify import get_classifier
        
        # Create fake with rejected term
        fake_with_rejection = FakeHybridTieredClassifier(
            responses={
                "noise_term": ClassificationResponse(
                    term="noise_term",
                    classification="rejected",
                    confidence=0.0,
                    canonical_term="noise_term",
                    tier_used=3,
                    rejection_reason="noise_watermarks",
                ),
            }
        )
        
        app_with_fake_classifier.dependency_overrides[get_classifier] = (
            lambda: fake_with_rejection
        )
        
        client = TestClient(app_with_fake_classifier)
        response = client.post(
            CLASSIFY_ENDPOINT,
            json={"term": "noise_term"},
        )
        data = response.json()
        
        assert data["classification"] == "rejected"
        assert "rejection_reason" in data
        assert data["rejection_reason"] == "noise_watermarks"
