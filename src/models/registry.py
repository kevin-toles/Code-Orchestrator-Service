"""
Code-Orchestrator-Service - Model Registry

WBS 2.1: Model Registry Infrastructure
Singleton pattern with lazy loading for HuggingFace model management.

Patterns Applied:
- Singleton Pattern with get_registry() (Anti-Pattern #12 Prevention)
- One-time configuration (Anti-Pattern #16 Prevention)
- Protocol typing for FakeModelRegistry (CODING_PATTERNS_ANALYSIS.md line 130)
- Repository Pattern with model caching

Anti-Patterns Avoided:
- #12: New client per request (models cached after first load)
- #16: Per-request configuration (registry initialized once)
- #7, #13: Exception shadowing (uses ModelLoadError, not RuntimeError)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol

from src.core.config import Settings
from src.core.exceptions import ModelLoadError
from src.core.logging import get_logger


class ModelRegistryProtocol(Protocol):
    """Protocol for ModelRegistry duck typing.

    Enables FakeModelRegistry for testing without real HuggingFace models.
    Pattern: Repository Pattern + FakeClient per CODING_PATTERNS_ANALYSIS.md line 130
    """

    def get_model(self, model_name: str) -> Any | None:
        """Get a loaded model by name."""
        ...

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        ...

    def register_model(self, model_name: str, model: Any) -> None:
        """Register a model in the cache."""
        ...

    def all_models_loaded(self) -> bool:
        """Check if all required models are loaded."""
        ...

# Get logger
logger = get_logger(__name__)


# Model configuration - hardcoded for now, will be in config/models.json
_DEFAULT_MODEL_CONFIG: dict[str, dict[str, str]] = {
    "codet5": {
        "hf_id": "Salesforce/codet5p-220m",
        "type": "seq2seq",
        "description": "Code generation and term extraction",
    },
    "graphcodebert": {
        "hf_id": "microsoft/graphcodebert-base",
        "type": "encoder",
        "description": "Code understanding and validation",
    },
    "codebert": {
        "hf_id": "microsoft/codebert-base",
        "type": "encoder",
        "description": "Code embeddings and ranking",
    },
}


class ModelRegistry:
    """Singleton registry for HuggingFace model management.

    WBS 2.1: Manages model lifecycle with lazy loading and caching.

    Pattern: Singleton + Repository per CODING_PATTERNS_ANALYSIS.md
    - get_registry() returns singleton instance
    - reset_registry() allows test isolation
    - Models cached after first load (Anti-Pattern #12)
    - Configuration loaded once (Anti-Pattern #16)

    Usage:
        registry = ModelRegistry.get_registry()
        model = registry.get_model("codet5")
    """

    _instance: ClassVar[ModelRegistry | None] = None
    _initialized: ClassVar[bool] = False

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize model registry.

        Do not call directly - use get_registry() instead.

        Args:
            settings: Application settings for model paths
        """
        self._settings = settings or Settings()
        self._models: dict[str, Any] = {}
        self._model_config: dict[str, dict[str, str]] = _DEFAULT_MODEL_CONFIG.copy()
        self._load_config_file()

    @classmethod
    def get_registry(cls, settings: Settings | None = None) -> ModelRegistry:
        """Get singleton ModelRegistry instance.

        Creates instance on first call, returns cached instance after.
        Pattern: Singleton with lazy initialization.

        Args:
            settings: Optional settings override (only used on first call)

        Returns:
            Singleton ModelRegistry instance
        """
        if cls._instance is None:
            cls._instance = ModelRegistry(settings=settings)
            cls._initialized = True
            logger.info("model_registry_initialized")

        return cls._instance

    @classmethod
    def reset_registry(cls) -> None:
        """Reset singleton instance for testing.

        Pattern: reset_logging() style for test isolation.
        Called in test fixtures to ensure clean state.
        """
        cls._instance = None
        cls._initialized = False
        logger.debug("model_registry_reset")

    @property
    def settings(self) -> Settings:
        """Get settings used by this registry."""
        return self._settings

    def _load_config_file(self) -> None:
        """Load model configuration from config/models.json if available."""
        config_path = Path("config/models.json")
        if config_path.exists():
            try:
                with config_path.open() as f:
                    self._model_config = json.load(f)
                logger.info("model_config_loaded", path=str(config_path))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("model_config_load_failed", error=str(e))
                # Keep default config

    def get_model(self, model_name: str) -> Any | None:
        """Get a loaded model by name.

        Returns None if model not loaded yet.

        Args:
            model_name: Model identifier (codet5, graphcodebert, codebert)

        Returns:
            Model tuple (model, tokenizer) or None if not loaded
        """
        return self._models.get(model_name)

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded.

        Args:
            model_name: Model identifier

        Returns:
            True if model is in cache
        """
        return model_name in self._models

    def register_model(self, model_name: str, model: Any) -> None:
        """Register a model in the cache.

        Called after successful model load or for testing with mocks.

        Args:
            model_name: Model identifier
            model: Model instance (or tuple of model, tokenizer)
        """
        self._models[model_name] = model
        logger.info("model_registered", model=model_name)

    def all_models_loaded(self) -> bool:
        """Check if all required models are loaded.

        WBS 2.1.4: Returns False if any model failed to load.

        Returns:
            True if codet5, graphcodebert, and codebert are all loaded
        """
        required_models = ["codet5", "graphcodebert", "codebert"]
        return all(self.is_loaded(m) for m in required_models)

    def get_model_config(self) -> dict[str, dict[str, str]]:
        """Get model configuration dictionary.

        Returns:
            Dict mapping model names to config (hf_id, type, description)
        """
        return self._model_config.copy()

    def get_hf_model_id(self, model_name: str) -> str:
        """Get HuggingFace model ID for a model.

        Args:
            model_name: Model identifier

        Returns:
            HuggingFace model ID string

        Raises:
            KeyError: If model not in config
        """
        return self._model_config[model_name]["hf_id"]

    def load_model(self, model_name: str) -> Any:
        """Load a model from HuggingFace.

        WBS 2.1.4: Handles OOM and other load failures gracefully.

        Args:
            model_name: Model identifier

        Returns:
            Loaded model tuple (model, tokenizer)

        Raises:
            ModelLoadError: If model fails to load
        """
        try:
            model = self._load_model_from_hf(model_name)
            self.register_model(model_name, model)
            return model
        except Exception as e:
            logger.error(
                "model_load_failed",
                model=model_name,
                error=str(e),
            )
            raise ModelLoadError(f"Failed to load {model_name}: {e}") from e

    def _load_model_from_hf(self, model_name: str) -> Any:
        """Load model from HuggingFace hub.

        Override in subclass or mock for testing.

        Args:
            model_name: Model identifier

        Returns:
            Model tuple (model, tokenizer)

        Raises:
            RuntimeError: On load failure
        """
        # Import here to avoid import at module load
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

        hf_id = self.get_hf_model_id(model_name)
        model_type = self._model_config[model_name]["type"]

        logger.info("loading_model_from_hf", model=model_name, hf_id=hf_id)

        tokenizer = AutoTokenizer.from_pretrained(hf_id)  # type: ignore[no-untyped-call]

        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(hf_id)
        else:  # encoder
            model = AutoModel.from_pretrained(hf_id)

        return (model, tokenizer)


class FakeModelRegistry:
    """Fake registry for unit testing without real HuggingFace models.

    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md line 150.
    Implements ModelRegistryProtocol via duck typing.

    Usage in tests:
        fake_registry = FakeModelRegistry()
        fake_registry.register_model("codet5", mock_model)
        agent = CodeT5Agent(registry=fake_registry)
    """

    def __init__(self) -> None:
        """Initialize fake registry with empty model cache."""
        self._models: dict[str, Any] = {}

    def get_model(self, model_name: str) -> Any | None:
        """Get a mock model by name."""
        return self._models.get(model_name)

    def is_loaded(self, model_name: str) -> bool:
        """Check if a mock model is registered."""
        return model_name in self._models

    def register_model(self, model_name: str, model: Any) -> None:
        """Register a mock model."""
        self._models[model_name] = model

    def all_models_loaded(self) -> bool:
        """Check if all required models are registered."""
        required_models = ["codet5", "graphcodebert", "codebert"]
        return all(self.is_loaded(m) for m in required_models)
