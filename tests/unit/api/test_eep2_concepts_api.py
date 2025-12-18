"""
Unit Tests for EEP-2.4: Concept Extraction API Endpoint

WBS: EEP-2.4 - Concept Extraction Endpoint
TDD Phase: Tests for POST /api/v1/concepts endpoint

Acceptance Criteria:
- AC-2.4.1: Add POST /api/v1/concepts endpoint

Anti-Patterns Avoided:
- S1192: Constants for repeated string literals
- S3776: Simple test methods
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

CONCEPTS_ENDPOINT = "/api/v1/concepts"
CONCEPTS_DOMAINS_ENDPOINT = "/api/v1/concepts/domains"

TEST_TEXT_RAG = "RAG retrieval vector embedding transformer attention langchain"
TEST_TEXT_MICROSERVICES = "kubernetes docker microservices API gateway circuit breaker"
TEST_KEYWORDS = ["retrieval", "embedding", "vector", "RAG", "langchain"]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create test client for FastAPI app."""
    from src.main import app
    return TestClient(app)


@pytest.fixture
def domain_taxonomy_path() -> Path:
    """Path to real domain taxonomy."""
    return Path("/Users/kevintoles/POC/semantic-search-service/config/domain_taxonomy.json")


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestConceptExtractionEndpoint:
    """Test POST /api/v1/concepts endpoint (AC-2.4.1)."""

    def test_endpoint_exists(self, client: TestClient) -> None:
        """AC-2.4.1: Endpoint should exist and accept POST."""
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_RAG},
        )
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    def test_extract_concepts_from_text(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """AC-2.4.1: Should extract concepts from text."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")

        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_RAG},
        )
        assert response.status_code == 200
        data = response.json()
        assert "concepts" in data
        assert "domain_score" in data  # AC-2.4.3
        assert "domain_scores" in data
        assert "primary_domain" in data

    def test_request_with_domain_field(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """AC-2.4.2: Request should accept top-level domain field."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")

        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_RAG, "domain": "llm_rag"},
        )
        assert response.status_code == 200
        data = response.json()
        # All concepts should be from llm_rag domain
        for concept in data["concepts"]:
            assert concept["domain"] == "llm_rag"

    def test_response_domain_score_field(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """AC-2.4.3: Response should include domain_score (float)."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")

        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_RAG},
        )
        assert response.status_code == 200
        data = response.json()
        assert "domain_score" in data
        assert isinstance(data["domain_score"], float)
        assert 0.0 <= data["domain_score"] <= 1.0

    def test_extract_concepts_from_keywords(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """AC-2.4.1: Should extract concepts from keywords."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"keywords": TEST_KEYWORDS},
        )
        assert response.status_code == 200
        data = response.json()
        assert "concepts" in data

    def test_requires_text_or_keywords(self, client: TestClient) -> None:
        """Should return 400 if neither text nor keywords provided."""
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={},
        )
        assert response.status_code == 400

    def test_response_has_metadata(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Response should include metadata."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_RAG},
        )
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "processing_time_ms" in data["metadata"]
        assert "total_concepts" in data["metadata"]

    def test_concepts_have_required_fields(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Concepts should have name, confidence, domain, tier fields."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_RAG},
        )
        assert response.status_code == 200
        data = response.json()
        
        if data["concepts"]:
            concept = data["concepts"][0]
            assert "name" in concept
            assert "confidence" in concept
            assert "domain" in concept
            assert "tier" in concept

    def test_options_min_confidence(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Options should filter by min_confidence."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        # Request with high confidence threshold
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={
                "text": TEST_TEXT_RAG,
                "options": {"min_confidence": 0.9},
            },
        )
        assert response.status_code == 200
        data = response.json()
        
        # All concepts should meet threshold
        for concept in data["concepts"]:
            assert concept["confidence"] >= 0.9

    def test_options_domain_filter(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Options should filter by domain."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={
                "text": TEST_TEXT_RAG,
                "options": {"domain_filter": "llm_rag"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        
        # All concepts should be from llm_rag domain
        for concept in data["concepts"]:
            assert concept["domain"] == "llm_rag"


class TestGetDomainsEndpoint:
    """Test GET /api/v1/concepts/domains endpoint."""

    def test_get_all_domains(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Should return all domains with their concepts."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.get(CONCEPTS_DOMAINS_ENDPOINT)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        # Should have at least one domain
        assert len(data) > 0

    def test_get_specific_domain(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Should return concepts for specific domain."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.get(f"{CONCEPTS_DOMAINS_ENDPOINT}/llm_rag")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_unknown_domain_returns_404(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Should return 404 for unknown domain."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.get(f"{CONCEPTS_DOMAINS_ENDPOINT}/unknown_domain_xyz")
        assert response.status_code == 404


class TestConceptExtractionIntegration:
    """Integration tests for concept extraction API."""

    def test_rag_text_classifies_as_llm_domain(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """RAG text should classify as llm_rag domain."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_RAG},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["primary_domain"] == "llm_rag"

    def test_microservices_text_classifies_correctly(
        self, client: TestClient, domain_taxonomy_path: Path
    ) -> None:
        """Microservices text should classify as microservices_architecture."""
        if not domain_taxonomy_path.exists():
            pytest.skip("Domain taxonomy not available")
        
        response = client.post(
            CONCEPTS_ENDPOINT,
            json={"text": TEST_TEXT_MICROSERVICES},
        )
        assert response.status_code == 200
        data = response.json()
        # Should have microservices_architecture in domain_scores
        assert "microservices_architecture" in data["domain_scores"]
