"""
Fake Embedders for Testing

WBS: EEP-1.5 - Multi-Modal Embedding Architecture
Per CODING_PATTERNS_ANALYSIS.md Repository Pattern:
- FakeClient implementations for unit testing
- Deterministic outputs for reproducibility
- No real model loading required

Anti-Patterns Avoided:
- #12: FakeClients enable testing without model loading
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.models.embedding.config import (
    DIM_BGE,
    DIM_FUSED,
    DIM_INSTRUCTOR,
    DIM_UNIXCODER,
    DOMAIN_INSTRUCTIONS,
)


def _deterministic_embedding(text: str, dim: int, seed: int = 42) -> NDArray[np.float32]:
    """Generate deterministic embedding from text hash.

    Args:
        text: Input text to hash
        dim: Output embedding dimension
        seed: Random seed for reproducibility

    Returns:
        Deterministic embedding vector
    """
    # Create hash-based seed from text
    text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + text_hash)

    # Generate and normalize
    embedding = rng.standard_normal(dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


class FakeBGEEmbedder:
    """Fake BGE embedder for testing.

    Returns deterministic embeddings based on input hash.
    No actual model loading required.

    Example:
        >>> embedder = FakeBGEEmbedder()
        >>> emb1 = embedder.embed("test")
        >>> emb2 = embedder.embed("test")
        >>> assert (emb1 == emb2).all()
    """

    def __init__(self, model_name: str | None = None):
        """Initialize fake embedder.

        Args:
            model_name: Ignored, kept for API compatibility
        """
        self._model_name = model_name or "fake-bge"
        self._embedding_dim = DIM_BGE

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate deterministic embedding for text.

        Args:
            text: Input text

        Returns:
            1024-dim embedding vector
        """
        return _deterministic_embedding(text, self._embedding_dim, seed=42)

    def batch_embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (len(texts), 1024)
        """
        embeddings = [self.embed(text) for text in texts]
        return np.stack(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def check_health(self) -> dict[str, str | bool]:
        """Return health status."""
        return {
            "status": "healthy",
            "model_name": self._model_name,
            "is_fake": True,
        }


class FakeUniXcoderEmbedder:
    """Fake UniXcoder embedder for testing.

    Returns deterministic embeddings based on input hash.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize fake embedder."""
        self._model_name = model_name or "fake-unixcoder"
        self._embedding_dim = DIM_UNIXCODER

    def embed(self, code: str) -> NDArray[np.float32]:
        """Generate deterministic embedding for code.

        Args:
            code: Input code snippet

        Returns:
            768-dim embedding vector
        """
        return _deterministic_embedding(code, self._embedding_dim, seed=43)

    def batch_embed(self, codes: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple code snippets.

        Args:
            codes: List of code snippets

        Returns:
            Array of shape (len(codes), 768)
        """
        embeddings = [self.embed(code) for code in codes]
        return np.stack(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def check_health(self) -> dict[str, str | bool]:
        """Return health status."""
        return {
            "status": "healthy",
            "model_name": self._model_name,
            "is_fake": True,
        }


class FakeInstructorEmbedder:
    """Fake Instructor embedder for testing.

    Returns deterministic embeddings based on input hash.
    Supports domain instructions.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize fake embedder."""
        self._model_name = model_name or "fake-instructor"
        self._embedding_dim = DIM_INSTRUCTOR
        self._domain_instructions = DOMAIN_INSTRUCTIONS.copy()

    def embed(
        self,
        text: str,
        instruction: str | None = None,
    ) -> NDArray[np.float32]:
        """Generate deterministic embedding for text with instruction.

        Args:
            text: Input text
            instruction: Instruction prefix (combined with text for hash)

        Returns:
            768-dim embedding vector
        """
        combined = f"{instruction or ''}{text}"
        return _deterministic_embedding(combined, self._embedding_dim, seed=44)

    def batch_embed(
        self,
        texts: list[str],
        instruction: str | None = None,
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            instruction: Instruction prefix for all texts

        Returns:
            Array of shape (len(texts), 768)
        """
        embeddings = [self.embed(text, instruction) for text in texts]
        return np.stack(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    @property
    def domain_instructions(self) -> dict[str, str]:
        """Return domain instruction templates."""
        return self._domain_instructions

    def check_health(self) -> dict[str, str | bool]:
        """Return health status."""
        return {
            "status": "healthy",
            "model_name": self._model_name,
            "is_fake": True,
        }


class FakeFusionLayer:
    """Fake fusion layer for testing.

    Combines embeddings via simple averaging (not learned).
    """

    def __init__(self, output_dim: int = DIM_FUSED):
        """Initialize fake fusion layer.

        Args:
            output_dim: Output embedding dimension
        """
        self._output_dim = output_dim

    def fuse(
        self,
        text_embedding: NDArray[np.float32],
        code_embedding: NDArray[np.float32],
        domain_embedding: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Fuse three embeddings via simple projection.

        Args:
            text_embedding: BGE text embedding (1024-dim)
            code_embedding: UniXcoder code embedding (768-dim)
            domain_embedding: Instructor domain embedding (768-dim)

        Returns:
            Fused embedding of output_dim
        """
        # Simple concatenation + projection via random matrix
        # (deterministic based on input)
        combined_text = f"{text_embedding.tobytes()}"
        fused = _deterministic_embedding(combined_text, self._output_dim, seed=45)
        return fused

    def batch_fuse(
        self,
        text_embeddings: NDArray[np.float32],
        code_embeddings: NDArray[np.float32],
        domain_embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Fuse batches of embeddings.

        Args:
            text_embeddings: Array of shape (batch, 1024)
            code_embeddings: Array of shape (batch, 768)
            domain_embeddings: Array of shape (batch, 768)

        Returns:
            Array of shape (batch, output_dim)
        """
        batch_size = text_embeddings.shape[0]
        fused = []
        for i in range(batch_size):
            fused.append(
                self.fuse(
                    text_embeddings[i],
                    code_embeddings[i],
                    domain_embeddings[i],
                )
            )
        return np.stack(fused)

    @property
    def output_dim(self) -> int:
        """Return output embedding dimension."""
        return self._output_dim

    def check_health(self) -> dict[str, str | bool | int]:
        """Return health status."""
        return {
            "status": "healthy",
            "components": {
                "fusion": "fake",
            },
            "output_dim": self._output_dim,
            "is_fake": True,
        }
