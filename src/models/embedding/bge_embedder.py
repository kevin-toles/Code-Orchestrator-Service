"""
BGE Embedder Module

WBS: EEP-1.5.2 - Fine-tune BGE-large for text similarity
Wrapper for BGE-large embedding model.

Anti-Patterns Avoided:
- #12: Model cached, not loaded per request
- S6903: No exception shadowing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.models.embedding.config import DIM_BGE, MODEL_BGE_LARGE


class BGEEmbedder:
    """BGE-large embedding model wrapper.

    Provides text embeddings using BAAI/bge-large-en-v1.5.
    Supports lazy loading and caching.

    Attributes:
        model_name: HuggingFace model identifier
        embedding_dim: Output embedding dimension (1024)

    Example:
        >>> embedder = BGEEmbedder()
        >>> embedding = embedder.embed("Hello world")
        >>> embedding.shape
        (1024,)
    """

    def __init__(
        self,
        model_name: str = MODEL_BGE_LARGE,
        device: str | None = None,
    ):
        """Initialize BGE embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Torch device (auto-detected if None)
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._loaded = False
        self._embedding_dim = DIM_BGE

    def _ensure_loaded(self) -> None:
        """Lazy load model (Anti-Pattern #12 prevention)."""
        if not self._loaded:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                self._loaded = True
            except ImportError:
                # Use fake embedder for testing when sentence-transformers unavailable
                from src.models.embedding.fakes import FakeBGEEmbedder

                self._model = FakeBGEEmbedder(self._model_name)
                self._loaded = True

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            1024-dim embedding vector
        """
        self._ensure_loaded()

        if hasattr(self._model, "encode"):
            # Real SentenceTransformer
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.astype(np.float32)
        else:
            # Fake embedder fallback
            return self._model.embed(text)

    def batch_embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (len(texts), 1024)
        """
        self._ensure_loaded()

        if hasattr(self._model, "encode"):
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)
        else:
            return self._model.batch_embed(texts)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    def check_health(self) -> dict[str, Any]:
        """Return health status.

        Returns:
            Health status dictionary
        """
        try:
            self._ensure_loaded()
            return {
                "status": "healthy",
                "model_name": self._model_name,
                "embedding_dim": self._embedding_dim,
                "loaded": self._loaded,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_name": self._model_name,
                "error": str(e),
            }
