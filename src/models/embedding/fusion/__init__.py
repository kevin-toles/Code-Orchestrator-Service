"""
Fusion module for combining multi-modal embeddings.

WBS: EEP-1.5.5, EEP-1.5.6
"""

from src.models.embedding.fusion.embedding_fusion import (
    FusionLayer,
)
from src.models.embedding.fusion.fusion_trainer import (
    ContrastiveLoss,
    FusionDataset,
    FusionTrainer,
)

__all__ = [
    "FusionLayer",
    "FusionTrainer",
    "FusionDataset",
    "ContrastiveLoss",
]
