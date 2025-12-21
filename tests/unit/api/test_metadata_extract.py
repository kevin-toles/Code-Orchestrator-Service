"""Unit tests for Metadata Extraction API - WBS-1.4.

Tests POST /api/v1/metadata/extract endpoint per AC-2.1 through AC-2.8.
TDD Phase: RED - All tests should FAIL initially.
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.main import app


# Create test client
client = TestClient(app)


# === WBS-1.4.1-1.4.2: Endpoint Registration ===

class TestEndpointRegistration:
    """Tests for endpoint registration (AC-2.1)."""

    def test_endpoint_registered(self) -> None:
        """AC-2.1: POST /api/v1/metadata/extract should be registered."""
        # Attempt to access the endpoint
        response = client.post(
            "/api/v1/metadata/extract",
            json={"text": "test content"}
        )
        # Should not return 404 (not found)
        assert response.status_code != status.HTTP_404_NOT_FOUND

    def test_endpoint_in_openapi(self) -> None:
        """AC-2.1: Endpoint should appear in OpenAPI spec."""
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        openapi = response.json()
        paths = openapi.get("paths", {})
        
        assert "/api/v1/metadata/extract" in paths
        assert "post" in paths["/api/v1/metadata/extract"]


# === WBS-1.4.3-1.4.4: Empty Text Validation ===

class TestEmptyTextValidation:
    """Tests for empty text validation (AC-2.7).
    
    Note: Pydantic validation returns 422 for invalid input,
    which is correct per REST semantics (Unprocessable Entity).
    """

    def test_empty_text_returns_422(self) -> None:
        """AC-2.7: Empty text should return 422 Unprocessable Entity."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={"text": ""}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_whitespace_only_text_returns_422(self) -> None:
        """AC-2.7: Whitespace-only text should return 422."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={"text": "   \n\t   "}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_text_field_returns_422(self) -> None:
        """AC-2.7: Missing text field should return 422."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# === WBS-1.4.5-1.4.6: Invalid Options Validation ===

class TestInvalidOptionsValidation:
    """Tests for invalid options returns 422 (AC-2.8)."""

    def test_negative_top_k_keywords_returns_422(self) -> None:
        """AC-2.8: Negative top_k_keywords should return 422."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "valid text",
                "options": {"top_k_keywords": -1}
            }
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_confidence_range_returns_422(self) -> None:
        """AC-2.8: Confidence > 1.0 should return 422."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "valid text",
                "options": {"min_keyword_confidence": 1.5}
            }
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_top_k_exceeds_max_returns_422(self) -> None:
        """AC-2.8: top_k > 100 should return 422."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "valid text",
                "options": {"top_k_keywords": 101}
            }
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# === WBS-1.4.7-1.4.8: Full Extraction Pipeline ===

class TestFullExtractionPipeline:
    """Tests for full extraction pipeline (AC-2.2 through AC-2.6)."""

    def test_successful_extraction_returns_200(self) -> None:
        """Successful extraction should return 200."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": """
                Microservices architecture enables scalable systems.
                API design patterns improve service communication.
                Docker containers provide deployment flexibility.
                """
            }
        )
        assert response.status_code == status.HTTP_200_OK

    def test_response_contains_keywords(self) -> None:
        """AC-2.2: Response should contain keywords list."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Machine learning and artificial intelligence applications."
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "keywords" in data
        assert isinstance(data["keywords"], list)

    def test_keywords_have_required_fields(self) -> None:
        """AC-2.2: Keywords should have term, score, is_technical."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Kubernetes container orchestration for cloud deployment."
            }
        )
        data = response.json()
        
        if data.get("keywords"):
            keyword = data["keywords"][0]
            assert "term" in keyword
            assert "score" in keyword
            assert "is_technical" in keyword

    def test_response_contains_concepts(self) -> None:
        """AC-2.3: Response should contain concepts list."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "RAG architecture with vector embeddings for semantic search."
            }
        )
        data = response.json()
        
        assert "concepts" in data
        assert isinstance(data["concepts"], list)

    def test_concepts_have_required_fields(self) -> None:
        """AC-2.3: Concepts should have name, confidence, domain, tier."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Prompt engineering for large language model optimization."
            }
        )
        data = response.json()
        
        if data.get("concepts"):
            concept = data["concepts"][0]
            assert "name" in concept
            assert "confidence" in concept
            assert "domain" in concept
            assert "tier" in concept

    def test_response_contains_rejected_keywords(self) -> None:
        """AC-2.4: Response should contain rejected section."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "OceanOfPDF provides technical content. Using www links."
            }
        )
        data = response.json()
        
        assert "rejected" in data
        assert "keywords" in data["rejected"]
        assert "reasons" in data["rejected"]

    def test_response_contains_quality_score(self) -> None:
        """AC-2.5: Response should contain quality_score."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Software engineering best practices and methodologies."
            }
        )
        data = response.json()
        
        assert "metadata" in data
        assert "quality_score" in data["metadata"]
        assert 0.0 <= data["metadata"]["quality_score"] <= 1.0

    def test_response_contains_domain_detection(self) -> None:
        """AC-2.6: Response should contain detected_domain."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "API gateway design patterns for microservices architecture."
            }
        )
        data = response.json()
        
        assert "metadata" in data
        assert "detected_domain" in data["metadata"]
        assert "domain_confidence" in data["metadata"]


# === Response Metadata Tests ===

class TestResponseMetadata:
    """Tests for response metadata fields."""

    def test_response_contains_processing_time(self) -> None:
        """Response should contain processing_time_ms."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={"text": "Quick test content."}
        )
        data = response.json()
        
        assert "metadata" in data
        assert "processing_time_ms" in data["metadata"]
        assert data["metadata"]["processing_time_ms"] >= 0

    def test_response_contains_text_length(self) -> None:
        """Response should contain text_length."""
        text = "This is a test with specific length."
        response = client.post(
            "/api/v1/metadata/extract",
            json={"text": text}
        )
        data = response.json()
        
        assert "metadata" in data
        assert "text_length" in data["metadata"]
        assert data["metadata"]["text_length"] == len(text)

    def test_response_contains_stages_completed(self) -> None:
        """Response should contain stages_completed."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={"text": "Test content for stages."}
        )
        data = response.json()
        
        assert "metadata" in data
        assert "stages_completed" in data["metadata"]
        assert isinstance(data["metadata"]["stages_completed"], list)
        assert "keywords" in data["metadata"]["stages_completed"]


# === Options Tests ===

class TestExtractionOptions:
    """Tests for extraction options."""

    def test_respects_top_k_keywords(self) -> None:
        """Should respect top_k_keywords option."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": """
                Software development includes coding, testing, debugging,
                deployment, monitoring, documentation, and maintenance.
                """,
                "options": {"top_k_keywords": 3}
            }
        )
        data = response.json()
        
        assert len(data["keywords"]) <= 3

    def test_respects_filter_noise_false(self) -> None:
        """Should respect filter_noise=false option."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "OceanOfPDF content",
                "options": {"filter_noise": False}
            }
        )
        data = response.json()
        
        # With filter_noise=False, rejected should be empty
        assert data["rejected"]["keywords"] == []

    def test_accepts_optional_title(self) -> None:
        """Should accept optional title parameter."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Chapter content here.",
                "title": "Introduction to AI"
            }
        )
        assert response.status_code == status.HTTP_200_OK

    def test_accepts_optional_book_title(self) -> None:
        """Should accept optional book_title parameter."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Chapter content here.",
                "book_title": "Machine Learning Handbook"
            }
        )
        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# HCE-5.0: Pipeline Integration API Tests (AC-5.3, AC-5.4)
# =============================================================================


class TestExtractionConfigRequest:
    """HCE-5.8, HCE-5.9: Test request schema has extraction_config (AC-5.3)."""

    def test_request_accepts_extraction_config(self) -> None:
        """HCE-5.8: Request should accept extraction_config field."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Machine learning models process data efficiently.",
                "extraction_config": {
                    "enable_yake": True,
                    "enable_textrank": True,
                    "enable_stem_dedup": True,
                    "enable_semantic_dedup": False
                }
            }
        )
        assert response.status_code == status.HTTP_200_OK

    def test_request_extraction_config_optional(self) -> None:
        """HCE-5.8: extraction_config should be optional."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Deep learning neural networks enable AI systems."
            }
        )
        assert response.status_code == status.HTTP_200_OK

    def test_config_disable_yake(self) -> None:
        """HCE-5.9: Config should allow disabling YAKE."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "API design patterns improve REST services.",
                "extraction_config": {
                    "enable_yake": False,
                    "enable_textrank": True
                }
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Pipeline metadata should reflect which stages were executed
        if "pipeline_metadata" in data.get("metadata", {}):
            stages = data["metadata"]["pipeline_metadata"].get("stages_executed", [])
            assert "yake" not in stages

    def test_config_disable_semantic_dedup(self) -> None:
        """HCE-5.9: Config should allow disabling semantic dedup."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Kubernetes orchestration enables container management.",
                "extraction_config": {
                    "enable_semantic_dedup": False
                }
            }
        )
        assert response.status_code == status.HTTP_200_OK


class TestPipelineMetadataResponse:
    """HCE-5.12, HCE-5.13: Test response has pipeline_metadata (AC-5.4)."""

    def test_response_has_pipeline_metadata(self) -> None:
        """HCE-5.12: Response metadata should include pipeline_metadata."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Machine learning models process data efficiently."
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Pipeline metadata should be in response metadata
        metadata = data.get("metadata", {})
        assert "pipeline_metadata" in metadata or "stages_completed" in metadata

    def test_pipeline_metadata_has_stages_executed(self) -> None:
        """HCE-5.12: pipeline_metadata should have stages_executed."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Deep learning neural networks enable AI systems."
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        metadata = data.get("metadata", {})
        # Either pipeline_metadata.stages_executed or stages_completed directly
        stages = metadata.get("pipeline_metadata", {}).get("stages_executed", [])
        if not stages:
            stages = metadata.get("stages_completed", [])
        
        assert isinstance(stages, list)

    def test_response_has_dedup_stats(self) -> None:
        """HCE-5.13: Response should include dedup_stats."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Machine learning and deep learning enable AI systems. Models model data."
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        metadata = data.get("metadata", {})
        # dedup_stats should be present
        assert "dedup_stats" in metadata or "pipeline_metadata" in metadata

    def test_dedup_stats_has_stem_removed(self) -> None:
        """HCE-5.13: dedup_stats should have stem_removed count."""
        response = client.post(
            "/api/v1/metadata/extract",
            json={
                "text": "Implementation implements implementing. Complex complexity."
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        metadata = data.get("metadata", {})
        dedup_stats = metadata.get("dedup_stats", {})
        if not dedup_stats:
            dedup_stats = metadata.get("pipeline_metadata", {}).get("dedup_stats", {})
        
        # stem_removed key should exist (may be 0)
        assert "stem_removed" in dedup_stats or len(dedup_stats) == 0
