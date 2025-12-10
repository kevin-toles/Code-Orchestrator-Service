"""
Code-Orchestrator-Service - Model Registry Tests

WBS 2.1: Model Registry Infrastructure
TDD Phase: RED - Write failing tests first

Tests for ModelRegistry class that manages HuggingFace model lifecycle.

Patterns Applied:
- Repository Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md line 130)
- FakeClient for Testing (CODING_PATTERNS_ANALYSIS.md line 150)
- Singleton pattern with lazy loading (Anti-Pattern #12, #16)

Anti-Patterns Avoided:
- #12: New client per request (ModelRegistry caches models)
- #16: Per-request configuration (Registry initializes once)
"""

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# WBS 2.1.1: ModelRegistry Singleton Pattern Tests
# =============================================================================


class TestModelRegistrySingleton:
    """Test ModelRegistry singleton pattern."""

    def test_registry_class_exists(self) -> None:
        """ModelRegistry class should exist in src.agents.registry."""
        from src.agents.registry import ModelRegistry

        assert ModelRegistry is not None

    def test_registry_is_singleton(self) -> None:
        """Multiple calls to get_registry() return same instance.

        Anti-Pattern #12 Prevention: Don't create new registry per request.
        """
        from src.agents.registry import ModelRegistry

        registry1 = ModelRegistry.get_registry()
        registry2 = ModelRegistry.get_registry()

        assert registry1 is registry2

    def test_registry_reset_creates_new_instance(self) -> None:
        """reset_registry() allows new instance creation for testing.

        Pattern: reset_logging() style for test isolation.
        """
        from src.agents.registry import ModelRegistry

        registry1 = ModelRegistry.get_registry()
        ModelRegistry.reset_registry()
        registry2 = ModelRegistry.get_registry()

        assert registry1 is not registry2

    def test_registry_accepts_settings(self) -> None:
        """Registry can be configured with custom settings.

        Pattern: Pydantic Settings injection for testing.
        """
        from src.agents.registry import ModelRegistry
        from src.core.config import Settings

        ModelRegistry.reset_registry()
        settings = Settings(model_cache_dir="./test_cache")
        registry = ModelRegistry.get_registry(settings=settings)

        assert registry.settings.model_cache_dir == "./test_cache"


# =============================================================================
# WBS 2.1.2: Model Caching Tests
# =============================================================================


class TestModelCaching:
    """Test model caching after first load."""

    def test_registry_has_model_cache(self) -> None:
        """Registry maintains internal model cache dictionary."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        assert hasattr(registry, "_models")
        assert isinstance(registry._models, dict)

    def test_get_model_returns_none_before_load(self) -> None:
        """get_model() returns None for unloaded model."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        result = registry.get_model("codet5")

        assert result is None

    def test_register_model_adds_to_cache(self) -> None:
        """register_model() adds model to internal cache."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()
        mock_model = MagicMock()

        registry.register_model("codet5", mock_model)
        result = registry.get_model("codet5")

        assert result is mock_model

    def test_second_get_returns_cached_model(self) -> None:
        """Second call to get_model returns same cached instance.

        Anti-Pattern #12 Prevention: Don't load model per request.
        """
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()
        mock_model = MagicMock()
        registry.register_model("graphcodebert", mock_model)

        result1 = registry.get_model("graphcodebert")
        result2 = registry.get_model("graphcodebert")

        assert result1 is result2

    def test_is_loaded_returns_false_initially(self) -> None:
        """is_loaded() returns False for unregistered model."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        assert registry.is_loaded("codet5") is False

    def test_is_loaded_returns_true_after_register(self) -> None:
        """is_loaded() returns True after model registered."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()
        registry.register_model("codebert", MagicMock())

        assert registry.is_loaded("codebert") is True


# =============================================================================
# WBS 2.1.3: Model Config File Tests
# =============================================================================


class TestModelConfig:
    """Test model configuration from config/models.json."""

    def test_registry_loads_model_config(self) -> None:
        """Registry loads model IDs from configuration."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        config = registry.get_model_config()

        assert "codet5" in config
        assert "graphcodebert" in config
        assert "codebert" in config

    def test_model_config_has_huggingface_ids(self) -> None:
        """Config contains HuggingFace model IDs per WBS spec."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        config = registry.get_model_config()

        assert config["codet5"]["hf_id"] == "Salesforce/codet5p-220m"
        assert config["graphcodebert"]["hf_id"] == "microsoft/graphcodebert-base"
        assert config["codebert"]["hf_id"] == "microsoft/codebert-base"

    def test_model_config_has_model_type(self) -> None:
        """Config specifies model type for each model."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        config = registry.get_model_config()

        assert config["codet5"]["type"] == "seq2seq"
        assert config["graphcodebert"]["type"] == "encoder"
        assert config["codebert"]["type"] == "encoder"

    def test_get_hf_model_id_returns_configured_value(self) -> None:
        """get_hf_model_id() returns HuggingFace model ID."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        assert registry.get_hf_model_id("codet5") == "Salesforce/codet5p-220m"


# =============================================================================
# WBS 2.1.4: Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Test graceful degradation on model load failures."""

    def test_registry_handles_oom_error(self) -> None:
        """Registry catches OOM errors and raises ModelLoadError."""
        from src.agents.registry import ModelRegistry
        from src.core.exceptions import ModelLoadError

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        with (
            pytest.raises(ModelLoadError) as exc_info,
            patch.object(
                registry,
                "_load_model_from_hf",
                side_effect=RuntimeError("CUDA out of memory"),
            ),
        ):
            registry.load_model("codet5")

        assert "CUDA out of memory" in str(exc_info.value)

    def test_registry_logs_load_failure(self) -> None:
        """Registry logs model load failures with structured logging."""
        import contextlib

        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        with (
            patch("src.agents.registry.logger") as mock_logger,
            patch.object(
                registry,
                "_load_model_from_hf",
                side_effect=RuntimeError("Network error"),
            ),
            contextlib.suppress(Exception),
        ):
            registry.load_model("codet5")

        mock_logger.error.assert_called()

    def test_all_models_loaded_returns_false_if_any_failed(self) -> None:
        """all_models_loaded() returns False if any model failed to load."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        # Only register 2 of 3 models
        registry.register_model("codet5", MagicMock())
        registry.register_model("graphcodebert", MagicMock())

        assert registry.all_models_loaded() is False

    def test_all_models_loaded_returns_true_when_complete(self) -> None:
        """all_models_loaded() returns True when all models loaded."""
        from src.agents.registry import ModelRegistry

        ModelRegistry.reset_registry()
        registry = ModelRegistry.get_registry()

        # Register all 3 models
        registry.register_model("codet5", MagicMock())
        registry.register_model("graphcodebert", MagicMock())
        registry.register_model("codebert", MagicMock())

        assert registry.all_models_loaded() is True


# =============================================================================
# WBS 2.1: Model Registry Protocol Tests
# =============================================================================


class TestModelRegistryProtocol:
    """Test Protocol typing for FakeModelRegistry.

    Pattern: Protocol + FakeClient per CODING_PATTERNS_ANALYSIS.md line 130.
    """

    def test_model_registry_protocol_exists(self) -> None:
        """ModelRegistryProtocol exists for duck typing."""
        from src.agents.registry import ModelRegistryProtocol

        assert ModelRegistryProtocol is not None

    def test_fake_model_registry_implements_protocol(self) -> None:
        """FakeModelRegistry implements ModelRegistryProtocol.

        Pattern: FakeClient for testing without real HuggingFace models.
        """
        from src.agents.registry import FakeModelRegistry

        fake = FakeModelRegistry()

        # Duck typing check - these should not raise
        assert hasattr(fake, "get_model")
        assert hasattr(fake, "is_loaded")
        assert hasattr(fake, "register_model")
        assert hasattr(fake, "all_models_loaded")

    def test_fake_registry_returns_mock_models(self) -> None:
        """FakeModelRegistry returns mock models for testing."""
        from src.agents.registry import FakeModelRegistry

        fake = FakeModelRegistry()
        fake.register_model("codet5", MagicMock(name="FakeCodeT5"))

        model = fake.get_model("codet5")

        assert model is not None

    def test_fake_registry_is_loaded_works(self) -> None:
        """FakeModelRegistry.is_loaded() works correctly."""
        from src.agents.registry import FakeModelRegistry

        fake = FakeModelRegistry()

        assert fake.is_loaded("codet5") is False

        fake.register_model("codet5", MagicMock())

        assert fake.is_loaded("codet5") is True


# =============================================================================
# Integration: Ready Endpoint Update
# =============================================================================


class TestReadyEndpointIntegration:
    """Test /ready endpoint integration with ModelRegistry."""

    def test_ready_checks_model_registry(self) -> None:
        """HealthService.check_readiness() queries ModelRegistry."""
        from src.agents.registry import FakeModelRegistry, ModelRegistry
        from src.api.health import HealthService

        # Reset and create fake registry
        ModelRegistry.reset_registry()
        fake = FakeModelRegistry()

        health_service = HealthService()
        health_service.set_model_registry(fake)

        # No models loaded
        _, is_ready = health_service.check_readiness()
        assert is_ready is False

        # Load all models
        fake.register_model("codet5", MagicMock())
        fake.register_model("graphcodebert", MagicMock())
        fake.register_model("codebert", MagicMock())

        _, is_ready = health_service.check_readiness()
        assert is_ready is True
