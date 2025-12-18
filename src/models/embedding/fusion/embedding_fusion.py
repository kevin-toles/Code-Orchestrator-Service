"""
Embedding Fusion Layer Module

WBS: EEP-1.5.5 - Learned Fusion Layer
AC-1.5.5.1: FusionLayer(nn.Module) with cross-attention
AC-1.5.5.2: Configurable output dimension (default 512)
AC-1.5.5.3: forward() accepts three embeddings

Multi-modal embedding fusion using cross-attention and MLP.

Anti-Patterns Avoided:
- S3776: Helper functions for cognitive complexity < 15
- S6903: No exception shadowing
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from typing import Any

from src.models.embedding.config import (
    DIM_BGE,
    DIM_FUSION_OUTPUT,
    DIM_INSTRUCTOR,
    DIM_UNIXCODER,
)


class FusionLayer(nn.Module):
    """Multi-modal embedding fusion layer.

    AC-1.5.5.1: FusionLayer(nn.Module) with cross-attention

    Combines embeddings from BGE, UniXcoder, and Instructor models
    using learned cross-attention and MLP projection.

    Architecture:
        1. Project all embeddings to common dimension
        2. Apply cross-attention between modalities
        3. Concatenate attended outputs
        4. MLP projection to output dimension

    Attributes:
        bge_dim: BGE embedding dimension (1024)
        unixcoder_dim: UniXcoder embedding dimension (768)
        instructor_dim: Instructor embedding dimension (768)
        output_dim: Output embedding dimension (default 512)

    Example:
        >>> fusion = FusionLayer(output_dim=512)
        >>> bge = torch.randn(1, 1024)
        >>> unixcoder = torch.randn(1, 768)
        >>> instructor = torch.randn(1, 768)
        >>> fused = fusion(bge, unixcoder, instructor)
        >>> fused.shape
        torch.Size([1, 512])
    """

    def __init__(
        self,
        bge_dim: int = DIM_BGE,
        unixcoder_dim: int = DIM_UNIXCODER,
        instructor_dim: int = DIM_INSTRUCTOR,
        output_dim: int = DIM_FUSION_OUTPUT,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize fusion layer.

        AC-1.5.5.2: Configurable output dimension (default 512)

        Args:
            bge_dim: BGE embedding dimension
            unixcoder_dim: UniXcoder embedding dimension
            instructor_dim: Instructor embedding dimension
            output_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self._bge_dim = bge_dim
        self._unixcoder_dim = unixcoder_dim
        self._instructor_dim = instructor_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim

        # Project all embeddings to common dimension
        self._bge_proj = nn.Linear(bge_dim, hidden_dim)
        self._unixcoder_proj = nn.Linear(unixcoder_dim, hidden_dim)
        self._instructor_proj = nn.Linear(instructor_dim, hidden_dim)

        # Cross-attention layers
        self._cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Public alias for testing/inspection
        self.cross_attention = self._cross_attn

        # MLP for final projection
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        # Public alias for testing/inspection
        self.mlp = self._mlp

        # Layer normalization
        self._layer_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        bge_embedding: torch.Tensor,
        unixcoder_embedding: torch.Tensor,
        instructor_embedding: torch.Tensor,
        has_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass combining three embeddings.

        AC-1.5.5.3: forward() accepts three embeddings
        AC-1.5.5.6: Handle missing modalities via has_code mask
        AC-1.5.5.7: Forward pass with attention mask for missing code

        Args:
            bge_embedding: BGE embedding [batch, 1024]
            unixcoder_embedding: UniXcoder embedding [batch, 768]
            instructor_embedding: Instructor embedding [batch, 768]
            has_code: Boolean mask [batch] indicating code presence (optional)

        Returns:
            Fused embedding [batch, output_dim], L2-normalized
        """
        batch_size = bge_embedding.size(0)
        device = bge_embedding.device

        # AC-1.5.5.6: Handle missing modalities
        if has_code is None:
            # Default: assume all samples have code
            has_code = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Project to common dimension (S3776 compliant helper calls)
        bge_proj = self._project_embedding(bge_embedding, self._bge_proj)
        unixcoder_proj = self._project_embedding(unixcoder_embedding, self._unixcoder_proj)
        instructor_proj = self._project_embedding(instructor_embedding, self._instructor_proj)

        # AC-1.5.5.6: Zero out code embeddings for samples without code blocks
        if not has_code.all():
            # Create mask for code embeddings [batch, 1, hidden_dim]
            code_mask = has_code.unsqueeze(-1).unsqueeze(-1).float()
            unixcoder_proj = unixcoder_proj * code_mask

        # AC-1.5.5.7: Create attention mask for missing code modality
        # mask=True means IGNORE this position in MultiheadAttention
        attn_mask = self._create_attention_mask(has_code, device)

        # Apply cross-attention with optional mask
        attended = self._apply_cross_attention(
            bge_proj, unixcoder_proj, instructor_proj, attn_mask
        )

        # MLP projection
        fused = self._mlp(attended)

        # Layer normalization
        fused = self._layer_norm(fused)

        # AC-1.5.5.5: L2 normalize for cosine similarity
        fused = F.normalize(fused, p=2, dim=-1)

        return fused

    def _create_attention_mask(
        self,
        has_code: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Create attention mask for missing code modality.

        AC-1.5.5.7: key_padding_mask for missing code

        Args:
            has_code: Boolean mask [batch] indicating code presence
            device: Torch device

        Returns:
            Attention mask [batch, 3] where True = ignore position, or None if all have code
        """
        if has_code.all():
            return None

        batch_size = has_code.size(0)
        # [batch, 3] mask: text=False, code=~has_code, concept=False
        attn_mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        attn_mask[:, 1] = ~has_code  # Mask code position if no code
        return attn_mask

    def fuse(
        self,
        bge_embedding: torch.Tensor | np.ndarray,
        unixcoder_embedding: torch.Tensor | np.ndarray,
        instructor_embedding: torch.Tensor | np.ndarray,
        has_code: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Alias for forward() - fuse three embeddings.

        AC-1.5.5.6: Supports has_code mask for missing modalities

        Args:
            bge_embedding: BGE embedding [batch, 1024] or [1024]
            unixcoder_embedding: UniXcoder embedding [batch, 768] or [768]
            instructor_embedding: Instructor embedding [batch, 768] or [768]
            has_code: Boolean mask [batch] indicating code presence (optional)

        Returns:
            Fused embedding [batch, output_dim] or [output_dim], L2-normalized
        """
        squeeze_output = False

        # Convert numpy arrays to tensors
        if isinstance(bge_embedding, np.ndarray):
            bge_embedding = torch.tensor(bge_embedding, dtype=torch.float32)
        if isinstance(unixcoder_embedding, np.ndarray):
            unixcoder_embedding = torch.tensor(unixcoder_embedding, dtype=torch.float32)
        if isinstance(instructor_embedding, np.ndarray):
            instructor_embedding = torch.tensor(instructor_embedding, dtype=torch.float32)
        if isinstance(has_code, np.ndarray):
            has_code = torch.tensor(has_code, dtype=torch.bool)

        # Handle 1D inputs by adding batch dimension
        if bge_embedding.dim() == 1:
            bge_embedding = bge_embedding.unsqueeze(0)
            squeeze_output = True
        if unixcoder_embedding.dim() == 1:
            unixcoder_embedding = unixcoder_embedding.unsqueeze(0)
        if instructor_embedding.dim() == 1:
            instructor_embedding = instructor_embedding.unsqueeze(0)

        result = self.forward(
            bge_embedding, unixcoder_embedding, instructor_embedding, has_code
        )

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def batch_fuse(
        self,
        bge_embeddings: torch.Tensor | np.ndarray,
        unixcoder_embeddings: torch.Tensor | np.ndarray,
        instructor_embeddings: torch.Tensor | np.ndarray,
        has_code: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Fuse batch of embeddings.

        AC-1.5.5.6: Supports has_code mask for missing modalities

        Args:
            bge_embeddings: BGE embeddings [batch, 1024]
            unixcoder_embeddings: UniXcoder embeddings [batch, 768]
            instructor_embeddings: Instructor embeddings [batch, 768]
            has_code: Boolean mask [batch] indicating code presence (optional)

        Returns:
            Fused embeddings [batch, output_dim], L2-normalized
        """
        return self.fuse(
            bge_embeddings, unixcoder_embeddings, instructor_embeddings, has_code
        )

    def _project_embedding(
        self,
        embedding: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        """Project embedding to hidden dimension.

        Args:
            embedding: Input embedding
            projection: Linear projection layer

        Returns:
            Projected embedding
        """
        # Add sequence dimension if not present
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(1)
        return projection(embedding)

    # Alias for S3776 compliance tests
    _project_embeddings = _project_embedding

    def _apply_cross_attention(
        self,
        bge_proj: torch.Tensor,
        unixcoder_proj: torch.Tensor,
        instructor_proj: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply cross-attention between modalities.

        AC-1.5.5.7: Supports key_padding_mask for missing modalities

        Args:
            bge_proj: Projected BGE embedding
            unixcoder_proj: Projected UniXcoder embedding
            instructor_proj: Projected Instructor embedding
            key_padding_mask: Optional mask [batch, 3] where True = ignore

        Returns:
            Concatenated attended outputs
        """
        # Stack as key-value pairs
        kv = torch.cat([bge_proj, unixcoder_proj, instructor_proj], dim=1)

        # Each modality attends to all modalities (with optional mask)
        bge_attended, _ = self._cross_attn(
            bge_proj, kv, kv, key_padding_mask=key_padding_mask
        )
        unixcoder_attended, _ = self._cross_attn(
            unixcoder_proj, kv, kv, key_padding_mask=key_padding_mask
        )
        instructor_attended, _ = self._cross_attn(
            instructor_proj, kv, kv, key_padding_mask=key_padding_mask
        )

        # Squeeze sequence dimension and concatenate
        bge_out = bge_attended.squeeze(1)
        unixcoder_out = unixcoder_attended.squeeze(1)
        instructor_out = instructor_attended.squeeze(1)

        return torch.cat([bge_out, unixcoder_out, instructor_out], dim=-1)

    @property
    def output_dim(self) -> int:
        """Return output embedding dimension."""
        return self._output_dim

    @property
    def bge_dim(self) -> int:
        """Return BGE input dimension."""
        return self._bge_dim

    @property
    def unixcoder_dim(self) -> int:
        """Return UniXcoder input dimension."""
        return self._unixcoder_dim

    @property
    def instructor_dim(self) -> int:
        """Return Instructor input dimension."""
        return self._instructor_dim

    def check_health(self) -> dict[str, Any]:
        """Return health status of the fusion layer.

        Returns:
            Health status dictionary
        """
        try:
            # Simple forward pass test with dummy inputs
            with torch.no_grad():
                test_bge = torch.randn(1, self._bge_dim)
                test_unixcoder = torch.randn(1, self._unixcoder_dim)
                test_instructor = torch.randn(1, self._instructor_dim)
                _ = self.forward(test_bge, test_unixcoder, test_instructor)

            return {
                "status": "healthy",
                "output_dim": self._output_dim,
                "bge_dim": self._bge_dim,
                "unixcoder_dim": self._unixcoder_dim,
                "instructor_dim": self._instructor_dim,
                "components": {
                    "bge_proj": "Linear",
                    "unixcoder_proj": "Linear",
                    "instructor_proj": "Linear",
                    "cross_attention": "MultiheadAttention",
                    "mlp": "Sequential",
                },
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def get_attention_weights(
        self,
        bge_embedding: torch.Tensor,
        unixcoder_embedding: torch.Tensor,
        instructor_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Get attention weights for interpretability.

        Args:
            bge_embedding: BGE embedding
            unixcoder_embedding: UniXcoder embedding
            instructor_embedding: Instructor embedding

        Returns:
            Dictionary with attention weights per modality
        """
        # Project embeddings
        bge_proj = self._project_embedding(bge_embedding, self._bge_proj)
        unixcoder_proj = self._project_embedding(unixcoder_embedding, self._unixcoder_proj)
        instructor_proj = self._project_embedding(instructor_embedding, self._instructor_proj)

        kv = torch.cat([bge_proj, unixcoder_proj, instructor_proj], dim=1)

        # Get attention weights
        _, bge_attn = self._cross_attn(bge_proj, kv, kv, need_weights=True)
        _, unixcoder_attn = self._cross_attn(unixcoder_proj, kv, kv, need_weights=True)
        _, instructor_attn = self._cross_attn(instructor_proj, kv, kv, need_weights=True)

        return {
            "bge_attention": bge_attn,
            "unixcoder_attention": unixcoder_attn,
            "instructor_attention": instructor_attn,
        }
