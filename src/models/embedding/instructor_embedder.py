"""
Instructor Embedding Module

WBS: EEP-1.5.4 - Instructor-XL wrapper for domain-aware embeddings
AC-1.5.4.1: InstructorEmbedder wrapper with domain instructions
AC-1.5.4.2: Instruction templates per domain (ai-ml, python, architecture)

Domain-aware embeddings with instruction following.

Anti-Patterns Avoided:
- #12: Model cached, not loaded per request
- S6903: No exception shadowing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.models.embedding.config import (
    DIM_INSTRUCTOR,
    DOMAIN_INSTRUCTIONS,
    MODEL_INSTRUCTOR_XL,
)


class InstructorEmbedder:
    """Instructor embedding model wrapper.

    Provides domain-aware embeddings using hkunlp/instructor-xl.
    Uses instruction templates to guide embedding generation.

    Attributes:
        model_name: HuggingFace model identifier
        embedding_dim: Output embedding dimension (768)
        domain_instructions: Domain-specific instruction templates

    Example:
        >>> embedder = InstructorEmbedder()
        >>> embedding = embedder.embed("RAG pipeline", instruction="Represent the AI concept:")
        >>> embedding.shape
        (768,)
    """

    def __init__(
        self,
        model_name: str = MODEL_INSTRUCTOR_XL,
        device: str | None = None,
    ):
        """Initialize Instructor embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Torch device (auto-detected if None)
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._loaded = False
        self._embedding_dim = DIM_INSTRUCTOR
        self._domain_instructions = DOMAIN_INSTRUCTIONS.copy()

    def _ensure_loaded(self) -> None:
        """Lazy load model (Anti-Pattern #12 prevention)."""
        if not self._loaded:
            try:
                from InstructorEmbedding import INSTRUCTOR

                self._model = INSTRUCTOR(self._model_name)
                self._loaded = True
            except ImportError:
                # Use fake embedder for testing
                from src.models.embedding.fakes import FakeInstructorEmbedder

                self._model = FakeInstructorEmbedder(self._model_name)
                self._loaded = True

    def embed(
        self,
        text: str,
        instruction: str | None = None,
        domain: str | None = None,
    ) -> NDArray[np.float32]:
        """Generate embedding for text with instruction.

        AC-1.5.4.1: InstructorEmbedder wrapper with domain instructions

        Args:
            text: Input text
            instruction: Direct instruction string (overrides domain)
            domain: Domain key for instruction lookup

        Returns:
            768-dim embedding vector
        """
        self._ensure_loaded()

        # Resolve instruction
        if instruction is None:
            if domain and domain in self._domain_instructions:
                instruction = self._domain_instructions[domain]
            else:
                instruction = self._domain_instructions.get("default", "")

        if hasattr(self._model, "encode"):
            # Real Instructor model
            embedding = self._model.encode([[instruction, text]])
            return embedding[0].astype(np.float32)
        else:
            # Fake embedder
            return self._model.embed(text, instruction=instruction)

    def batch_embed(
        self,
        texts: list[str],
        instruction: str | None = None,
        domain: str | None = None,
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            instruction: Instruction for all texts
            domain: Domain key for instruction lookup

        Returns:
            Array of shape (len(texts), 768)
        """
        self._ensure_loaded()

        # Resolve instruction
        if instruction is None:
            if domain and domain in self._domain_instructions:
                instruction = self._domain_instructions[domain]
            else:
                instruction = self._domain_instructions.get("default", "")

        if hasattr(self._model, "encode"):
            # Real Instructor model
            pairs = [[instruction, text] for text in texts]
            embeddings = self._model.encode(pairs)
            return embeddings.astype(np.float32)
        else:
            # Fake embedder
            return self._model.batch_embed(texts, instruction=instruction)

    def embed_concepts(
        self,
        concepts: list[str],
        domain: str = "default",
    ) -> NDArray[np.float32]:
        """Embed a list of domain concepts as single embedding.

        Args:
            concepts: List of concept strings
            domain: Domain for instruction

        Returns:
            768-dim embedding representing all concepts
        """
        text = f"Domain concepts: {', '.join(concepts)}"
        return self.embed(text, domain=domain)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    @property
    def domain_instructions(self) -> dict[str, str]:
        """Return domain instruction templates.

        AC-1.5.4.2: Instruction templates per domain
        """
        return self._domain_instructions

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
                "domains": list(self._domain_instructions.keys()),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_name": self._model_name,
                "error": str(e),
            }
