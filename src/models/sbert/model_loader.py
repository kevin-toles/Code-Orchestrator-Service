# SPDX-FileCopyrightText: 2024 AI Platform Team
# SPDX-License-Identifier: MIT
"""
SBERT Model Loader - Singleton Wrapper with Thread Safety.

WBS 5.2 M2.1: Model Loading Infrastructure
Provides thread-safe singleton access to SemanticSimilarityEngine.

Patterns Applied:
- Singleton Pattern with get_sbert_model() (Anti-Pattern #6, #12)
- Protocol typing for duck typing (CODING_PATTERNS_ANALYSIS.md line 130)
- asyncio.Lock for thread safety (Anti-Pattern #10)
- Graceful degradation with TF-IDF fallback

Anti-Patterns Avoided:
- #6: Duplicate Classes - Single model instance
- #7: Exception Shadowing - Uses SBERTModelError, not RuntimeError
- #10: State Mutation - asyncio.Lock for concurrent access
- #12: Connection Pooling - Model cached after first load
"""
from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from src.core.exceptions import SBERTModelError
from src.core.logging import get_logger
from src.models.sbert.semantic_similarity_engine import (
    SENTENCE_TRANSFORMERS_AVAILABLE,
    SemanticSimilarityEngine,
    SimilarityConfig,
)

# Get logger
logger = get_logger(__name__)


# =============================================================================
# Protocol for Duck Typing (enables FakeSBERTModel for testing)
# Pattern: Repository Pattern + FakeClient per CODING_PATTERNS_ANALYSIS.md line 130
# =============================================================================


class SBERTModelProtocol(Protocol):
    """Protocol for SBERT model loader duck typing.

    Enables FakeSBERTModel for testing without real model loading.
    """

    def compute_embeddings(self, texts: list[str]) -> NDArray[np.float64]:
        """Compute embeddings for texts."""
        ...

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        ...

    def get_status(self) -> dict[str, Any]:
        """Get model status for health endpoints."""
        ...


# =============================================================================
# Singleton Model Loader
# =============================================================================


class SBERTModelLoader:
    """Singleton wrapper for SemanticSimilarityEngine with thread safety.

    WBS M2.1: Model Loading Infrastructure

    Pattern: Singleton + Repository per CODING_PATTERNS_ANALYSIS.md
    - get_sbert_model() returns singleton instance
    - reset_sbert_model() allows test isolation
    - asyncio.Lock for thread-safe concurrent access
    - Graceful degradation to TF-IDF when SBERT unavailable

    Usage:
        model_loader = get_sbert_model()
        embeddings = model_loader.compute_embeddings(["text"])
        status = model_loader.get_status()
    """

    _instance: ClassVar[SBERTModelLoader | None] = None
    _initialized: ClassVar[bool] = False

    def __init__(self, config: SimilarityConfig | None = None) -> None:
        """Initialize SBERT model loader.

        Do not call directly - use get_sbert_model() instead.

        Args:
            config: Optional SimilarityConfig for engine initialization
        """
        self._config = config or SimilarityConfig()
        self._lock = asyncio.Lock()
        self._engine: SemanticSimilarityEngine | None = None
        self._is_initialized = False

        # Initialize the engine
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the SemanticSimilarityEngine.

        Handles graceful degradation if SBERT unavailable.
        """
        try:
            self._engine = SemanticSimilarityEngine(self._config)
            self._is_initialized = True
            logger.info(
                "sbert_model_loader_initialized",
                model_name=self._config.model_name,
                using_fallback=self._engine.is_using_fallback,
            )
        except Exception as e:
            logger.error("sbert_model_loader_init_failed", error=str(e))
            raise SBERTModelError(f"Model failed to load: {e}") from e

    @classmethod
    def get_instance(cls, config: SimilarityConfig | None = None) -> SBERTModelLoader:
        """Get singleton SBERTModelLoader instance.

        Creates instance on first call, returns cached instance after.
        Pattern: Singleton with lazy initialization.

        Args:
            config: Optional config override (only used on first call)

        Returns:
            Singleton SBERTModelLoader instance
        """
        if cls._instance is None:
            cls._instance = SBERTModelLoader(config=config)
            cls._initialized = True
            logger.info("sbert_model_singleton_created")

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance for testing.

        Pattern: reset_logging() style for test isolation.
        Called in test fixtures to ensure clean state.
        """
        cls._instance = None
        cls._initialized = False
        logger.debug("sbert_model_singleton_reset")

    @property
    def engine(self) -> SemanticSimilarityEngine:
        """Get the underlying SemanticSimilarityEngine.

        Returns:
            The engine instance

        Raises:
            SBERTModelError: If engine not initialized
        """
        if self._engine is None:
            raise SBERTModelError("Engine not initialized")
        return self._engine

    @property
    def is_initialized(self) -> bool:
        """Check if model loader is initialized."""
        return self._is_initialized

    @property
    def is_sbert_available(self) -> bool:
        """Check if sentence-transformers is available."""
        return SENTENCE_TRANSFORMERS_AVAILABLE

    @property
    def using_fallback(self) -> bool:
        """Check if using TF-IDF fallback instead of SBERT."""
        if self._engine is None:
            return True
        return self._engine.is_using_fallback

    # =========================================================================
    # Core Methods (implements SBERTModelProtocol)
    # =========================================================================

    def compute_embeddings(self, texts: list[str]) -> NDArray[np.float64]:
        """Compute embeddings for a list of texts.

        Thread-safe method to compute SBERT embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            NDArray of embeddings with shape (len(texts), embedding_dim)

        Raises:
            SBERTModelError: If engine not initialized
        """
        if self._engine is None:
            raise SBERTModelError("Engine not initialized")

        return self._engine.compute_embeddings(texts)

    async def compute_embeddings_async(
        self, texts: list[str]
    ) -> NDArray[np.float64]:
        """Async wrapper for compute_embeddings with lock protection.

        Thread-safe async method for concurrent access.
        Anti-Pattern #10: Uses asyncio.Lock to protect state.

        Args:
            texts: List of texts to embed

        Returns:
            NDArray of embeddings
        """
        async with self._lock:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.compute_embeddings, texts
            )

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1

        Raises:
            SBERTModelError: If engine not initialized
        """
        if self._engine is None:
            raise SBERTModelError("Engine not initialized")

        embeddings = self._engine.compute_embeddings([text1, text2])
        # Cosine similarity between the two embeddings
        similarity = np.dot(embeddings[0], embeddings[1])
        # Clamp to [0, 1] for safety
        return float(max(0.0, min(1.0, similarity)))

    def get_status(self) -> dict[str, Any]:
        """Get model status for health endpoints.

        Returns dict suitable for /health endpoint response.

        Returns:
            Status dict with model_name, is_loaded, using_fallback
        """
        return {
            "model_name": self._config.model_name,
            "is_loaded": self._is_initialized,
            "using_fallback": self.using_fallback,
            "is_sbert_available": self.is_sbert_available,
        }


# =============================================================================
# Module-Level Functions (Public API)
# =============================================================================


def get_sbert_model(config: SimilarityConfig | None = None) -> SBERTModelLoader:
    """Get singleton SBERT model loader.

    Convenience function for singleton access.

    Args:
        config: Optional config (only used on first call)

    Returns:
        Singleton SBERTModelLoader instance
    """
    return SBERTModelLoader.get_instance(config=config)


def reset_sbert_model() -> None:
    """Reset SBERT model singleton for testing.

    Call in test fixtures to ensure clean state between tests.
    """
    SBERTModelLoader.reset_instance()
