"""
Phase 2 Integration Tests - Model Loading Infrastructure

WBS 2.x Integration Test:
- All three model wrappers can be initialized
- SBERT singleton loads correctly
- /ready endpoint returns model status

This test can be run:
1. Against a running service: pytest tests/integration/ --service-url=http://localhost:8083
2. With TestClient (in-process): pytest tests/integration/

Pattern: Integration tests per CODING_PATTERNS_ANALYSIS.md
"""

import pytest
from fastapi.testclient import TestClient

from src.api.health import get_health_service
from src.main import app


class TestPhase2ModelsLoad:
    """Phase 2 Integration Test: All three model wrappers load successfully."""

    @pytest.fixture(autouse=True)
    def reset_health_service(self) -> None:  # noqa: PT004
        """Reset health service before each test for isolation."""
        health_service = get_health_service()
        health_service._models_loaded = False

    def test_phase2_all_model_wrappers_initialize(self) -> None:
        """All three model wrappers initialize successfully.

        WBS Phase 2 Integration Test from WBS_IMPLEMENTATION.md
        Now uses SBERT-based wrappers instead of local HuggingFace models.
        """
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        # All wrappers should initialize without error
        extractor = CodeT5Extractor()
        ranker = CodeBERTRanker()
        validator = GraphCodeBERTValidator()

        assert extractor is not None
        assert ranker is not None
        assert validator is not None

    def test_ready_endpoint_returns_200_when_models_loaded(self) -> None:
        """Ready endpoint returns 200 when models_loaded flag is True."""
        health_service = get_health_service()
        health_service.set_models_loaded(True)

        client = TestClient(app)
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["models_loaded"] is True

    def test_ready_endpoint_returns_503_when_models_not_loaded(self) -> None:
        """Ready endpoint returns 503 when models not loaded."""
        health_service = get_health_service()
        health_service.set_models_loaded(False)

        client = TestClient(app)
        response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["models_loaded"] is False

    def test_health_endpoint_always_returns_200(self) -> None:
        """Health endpoint returns 200 regardless of model status."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "code-orchestrator-service"


class TestPhase2ModelWrapperClasses:
    """Integration tests for model wrapper class existence and protocols."""

    def test_all_model_wrapper_classes_importable(self) -> None:
        """All model wrapper classes can be imported."""
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.codet5_extractor import CodeT5Extractor
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        assert CodeT5Extractor is not None
        assert GraphCodeBERTValidator is not None
        assert CodeBERTRanker is not None

    def test_all_protocols_importable(self) -> None:
        """All protocols can be imported."""
        from src.models.protocols import (
            ExtractorProtocol,
            RankerProtocol,
            ValidatorProtocol,
        )

        assert ExtractorProtocol is not None
        assert ValidatorProtocol is not None
        assert RankerProtocol is not None

    def test_extractor_produces_results(self) -> None:
        """CodeT5Extractor extracts terms from text."""
        from src.models.codet5_extractor import CodeT5Extractor

        extractor = CodeT5Extractor()
        result = extractor.extract_terms("distributed caching with Redis")

        assert hasattr(result, "primary_terms")
        assert hasattr(result, "related_terms")
        assert isinstance(result.primary_terms, list)

    def test_validator_validates_terms(self) -> None:
        """GraphCodeBERTValidator validates terms against query."""
        from src.models.graphcodebert_validator import GraphCodeBERTValidator

        validator = GraphCodeBERTValidator()
        result = validator.validate_terms(
            ["Redis", "caching", "distributed"],
            "Redis distributed caching",
            "systems",
        )

        assert hasattr(result, "valid_terms")
        assert hasattr(result, "rejected_terms")
        assert isinstance(result.valid_terms, list)

    def test_ranker_ranks_terms(self) -> None:
        """CodeBERTRanker ranks terms by relevance."""
        from src.models.codebert_ranker import CodeBERTRanker

        ranker = CodeBERTRanker()
        result = ranker.rank_terms(
            ["Redis", "caching", "database"],
            "Redis caching",
        )

        assert hasattr(result, "ranked_terms")
        assert len(result.ranked_terms) == 3
        # Terms should have scores
        for term in result.ranked_terms:
            assert hasattr(term, "term")
            assert hasattr(term, "score")
            assert 0.0 <= term.score <= 1.0
