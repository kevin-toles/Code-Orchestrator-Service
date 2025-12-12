"""
Phase 2 Integration Tests - Model Loading Infrastructure

WBS 2.x Integration Test:
- All three models load successfully and respond
- /ready endpoint returns model-specific status

This test can be run:
1. Against a running service: pytest tests/integration/ --service-url=http://localhost:8083
2. With TestClient (in-process): pytest tests/integration/

Pattern: Integration tests per CODING_PATTERNS_ANALYSIS.md
"""

import pytest
from fastapi.testclient import TestClient

from src.models.registry import FakeModelRegistry
from src.api.health import get_health_service
from src.main import app


class TestPhase2ModelsLoad:
    """Phase 2 Integration Test: All three models load successfully."""

    @pytest.fixture(autouse=True)
    def reset_health_service(self) -> None:  # noqa: PT004
        """Reset health service before each test for isolation."""
        # Reset health service registry to None
        health_service = get_health_service()
        health_service._model_registry = None
        health_service._models_loaded = False

    def test_phase2_all_models_load(self) -> None:
        """All three models load successfully and respond.

        WBS Phase 2 Integration Test from WBS_IMPLEMENTATION.md
        """
        # Set up fake registry with all models loaded
        fake_registry = FakeModelRegistry()
        fake_registry.register_model("codet5", {"model": "fake", "tokenizer": "fake"})
        fake_registry.register_model(
            "graphcodebert", {"model": "fake", "tokenizer": "fake"}
        )
        fake_registry.register_model("codebert", {"model": "fake", "tokenizer": "fake"})

        # Set registry on health service
        health_service = get_health_service()
        health_service.set_model_registry(fake_registry)

        client = TestClient(app)
        response = client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["models"]["codet5"] == "loaded"
        assert data["models"]["graphcodebert"] == "loaded"
        assert data["models"]["codebert"] == "loaded"

    def test_ready_endpoint_returns_200_when_all_models_loaded(self) -> None:
        """Ready endpoint returns 200 when all models are loaded."""
        fake_registry = FakeModelRegistry()
        fake_registry.register_model("codet5", {"model": "fake", "tokenizer": "fake"})
        fake_registry.register_model(
            "graphcodebert", {"model": "fake", "tokenizer": "fake"}
        )
        fake_registry.register_model("codebert", {"model": "fake", "tokenizer": "fake"})

        health_service = get_health_service()
        health_service.set_model_registry(fake_registry)

        client = TestClient(app)
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["models_loaded"] is True

    def test_ready_endpoint_returns_503_when_models_not_loaded(self) -> None:
        """Ready endpoint returns 503 when not all models are loaded."""
        # Only register codet5, not all three
        fake_registry = FakeModelRegistry()
        fake_registry.register_model("codet5", {"model": "fake", "tokenizer": "fake"})

        health_service = get_health_service()
        health_service.set_model_registry(fake_registry)

        client = TestClient(app)
        response = client.get("/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["models_loaded"] is False

    def test_ready_shows_partial_model_status(self) -> None:
        """Ready endpoint shows which specific models are loaded."""
        # Only register codet5
        fake_registry = FakeModelRegistry()
        fake_registry.register_model("codet5", {"model": "fake", "tokenizer": "fake"})

        health_service = get_health_service()
        health_service.set_model_registry(fake_registry)

        client = TestClient(app)
        response = client.get("/ready")

        data = response.json()
        assert data["models"]["codet5"] == "loaded"
        assert data["models"]["graphcodebert"] == "not_loaded"
        assert data["models"]["codebert"] == "not_loaded"

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
            ModelRegistryProtocol,
            RankerProtocol,
            ValidatorProtocol,
        )

        assert ModelRegistryProtocol is not None
        assert ExtractorProtocol is not None
        assert ValidatorProtocol is not None
        assert RankerProtocol is not None

    def test_fake_registry_implements_protocol(self) -> None:
        """FakeModelRegistry implements ModelRegistryProtocol interface."""
        from src.models.registry import FakeModelRegistry

        registry = FakeModelRegistry()

        # Check all required methods exist (per ModelRegistryProtocol)
        assert hasattr(registry, "register_model")
        assert hasattr(registry, "get_model")
        assert hasattr(registry, "is_loaded")
        assert hasattr(registry, "all_models_loaded")

        # Check methods are callable
        assert callable(registry.register_model)
        assert callable(registry.get_model)
        assert callable(registry.is_loaded)
        assert callable(registry.all_models_loaded)

        # Verify it works as expected
        registry.register_model("test", {"model": "fake"})
        assert registry.is_loaded("test")
        assert registry.get_model("test") == {"model": "fake"}
