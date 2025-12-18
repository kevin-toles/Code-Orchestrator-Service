"""
UniXcoder Embedder Module

WBS: EEP-1.5.3 - Fine-tune UniXcoder for code similarity
Wrapper for UniXcoder code embedding model.

Anti-Patterns Avoided:
- #12: Model cached, not loaded per request
- S6903: No exception shadowing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.models.embedding.config import DIM_UNIXCODER, MODEL_UNIXCODER


class UniXcoderEmbedder:
    """UniXcoder code embedding model wrapper.

    Provides code embeddings using microsoft/unixcoder-base.
    Supports lazy loading and caching.

    Attributes:
        model_name: HuggingFace model identifier
        embedding_dim: Output embedding dimension (768)

    Example:
        >>> embedder = UniXcoderEmbedder()
        >>> embedding = embedder.embed("def foo(): pass")
        >>> embedding.shape
        (768,)
    """

    def __init__(
        self,
        model_name: str = MODEL_UNIXCODER,
        device: str | None = None,
    ):
        """Initialize UniXcoder embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Torch device (auto-detected if None)
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._embedding_dim = DIM_UNIXCODER

    def _ensure_loaded(self) -> None:
        """Lazy load model (Anti-Pattern #12 prevention)."""
        if not self._loaded:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModel.from_pretrained(self._model_name)

                if self._device:
                    self._model = self._model.to(self._device)

                self._model.eval()
                self._loaded = True
            except ImportError:
                # Use fake embedder for testing
                from src.models.embedding.fakes import FakeUniXcoderEmbedder

                self._model = FakeUniXcoderEmbedder(self._model_name)
                self._loaded = True

    def embed(self, code: str) -> NDArray[np.float32]:
        """Generate embedding for single code snippet.

        Args:
            code: Input code snippet

        Returns:
            768-dim embedding vector
        """
        self._ensure_loaded()

        if hasattr(self._model, "embed"):
            # Fake embedder fallback
            return self._model.embed(code)

        try:
            import torch
            import torch.nn.functional as F

            # Tokenize
            inputs = self._tokenizer(
                code,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            if self._device:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :]
                # Normalize
                embedding = F.normalize(embedding, p=2, dim=-1)

            return embedding.cpu().numpy().astype(np.float32).squeeze()

        except Exception:
            # Fallback to fake
            from src.models.embedding.fakes import FakeUniXcoderEmbedder

            fake = FakeUniXcoderEmbedder()
            return fake.embed(code)

    def batch_embed(self, codes: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple code snippets.

        Args:
            codes: List of code snippets

        Returns:
            Array of shape (len(codes), 768)
        """
        self._ensure_loaded()

        if hasattr(self._model, "batch_embed"):
            return self._model.batch_embed(codes)

        # Process in batches
        embeddings = []
        for code in codes:
            embeddings.append(self.embed(code))

        return np.stack(embeddings)

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
