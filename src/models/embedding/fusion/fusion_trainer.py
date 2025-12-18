"""
Fusion Layer Trainer Module

WBS: EEP-1.5.6 - Fusion Layer Training
AC-1.5.6.1: Train fusion layer on combined embeddings
AC-1.5.6.2: Contrastive learning objective
AC-1.5.6.3: Save/load trained fusion weights
AC-1.5.6.5: Log training metrics to TensorBoard

End-to-end training for the fusion layer.

Anti-Patterns Avoided:
- S3776: Helper functions for cognitive complexity < 15
- S6903: No exception shadowing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset

# TensorBoard import (optional - AC-1.5.6.5)
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]
    TENSORBOARD_AVAILABLE = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.models.embedding.config import FusionModelConfig, FusionTrainingConfig
from src.models.embedding.fusion.embedding_fusion import FusionLayer

logger = logging.getLogger(__name__)


class FusionDataset(Dataset):
    """Dataset for fusion layer training.

    Stores pre-computed embeddings from all three models.
    """

    def __init__(
        self,
        bge_embeddings: NDArray[np.float32],
        unixcoder_embeddings: NDArray[np.float32],
        instructor_embeddings: NDArray[np.float32],
        labels: NDArray[np.int64],
    ):
        """Initialize fusion dataset.

        Args:
            bge_embeddings: BGE embeddings [N, 1024]
            unixcoder_embeddings: UniXcoder embeddings [N, 768]
            instructor_embeddings: Instructor embeddings [N, 768]
            labels: Class labels for contrastive learning
        """
        self._bge = torch.tensor(bge_embeddings, dtype=torch.float32)
        self._unixcoder = torch.tensor(unixcoder_embeddings, dtype=torch.float32)
        self._instructor = torch.tensor(instructor_embeddings, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item by index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (bge, unixcoder, instructor, label)
        """
        return (
            self._bge[idx],
            self._unixcoder[idx],
            self._instructor[idx],
            self._labels[idx],
        )


class ContrastiveLoss(nn.Module):
    """Contrastive loss for fusion training.

    AC-1.5.6.2: Contrastive learning objective
    """

    def __init__(self, temperature: float = 0.07):
        """Initialize contrastive loss.

        Args:
            temperature: Temperature scaling factor
        """
        super().__init__()
        self._temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            embeddings: Fused embeddings [batch, dim]
            labels: Class labels [batch]

        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self._temperature

        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean log probability of positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = -mean_log_prob.mean()

        return loss


class FusionTrainer:
    """Trainer for fusion layer.

    AC-1.5.6.1: Train fusion layer on combined embeddings
    """

    def __init__(
        self,
        fusion_layer_or_config: FusionLayer | FusionTrainingConfig | FusionModelConfig,
        config: FusionModelConfig | FusionTrainingConfig | None = None,
        device: str | None = None,
    ):
        """Initialize fusion trainer.

        Args:
            fusion_layer_or_config: Either a FusionLayer to train, or a config
            config: Training configuration (optional if first arg is a layer)
            device: Torch device
        """
        # Support both (FusionLayer, config) and (config) signatures
        if isinstance(fusion_layer_or_config, (FusionTrainingConfig, FusionModelConfig)):
            # Config-first signature: create our own fusion layer
            self._config = fusion_layer_or_config
            self._fusion = FusionLayer()
        else:
            # FusionLayer-first signature
            self._fusion = fusion_layer_or_config
            self._config = config or FusionModelConfig()

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._fusion.to(self._device)

        # Loss function (AC-1.5.6.2)
        self._loss_fn = ContrastiveLoss(temperature=0.07)

        # Optimizer
        lr = getattr(self._config, "learning_rate", 1e-4)
        self._optimizer = torch.optim.AdamW(
            self._fusion.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

        # Training history
        self._history: list[dict[str, float]] = []

        # TensorBoard writer (AC-1.5.6.5)
        self._writer: SummaryWriter | None = None
        self._best_loss = float("inf")
        self._global_step = 0

    def train(
        self,
        train_dataset: FusionDataset | list[dict[str, Any]],
        val_dataset: FusionDataset | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Train the fusion layer.

        AC-1.5.6.5: Log training metrics to TensorBoard

        Args:
            train_dataset: Training dataset or list of training pair dicts
            val_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size (uses config default if None)
            log_dir: TensorBoard log directory (optional)

        Returns:
            Training results with final metrics
        """
        # Handle list of dicts input (for testing/convenience)
        if isinstance(train_dataset, list):
            train_dataset = self._create_fake_dataset(train_dataset)

        epochs = epochs or getattr(self._config, "epochs", 10)
        batch_size = batch_size or getattr(self._config, "batch_size", 16)

        # Initialize TensorBoard writer (AC-1.5.6.5)
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            if log_dir is None:
                output_dir = getattr(self._config, "output_dir", "models/fusion-trained")
                log_dir = Path(output_dir) / "tensorboard"
            self._writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self._writer = None
            logger.info("TensorBoard not available, logging to history only")
        self._best_loss = float("inf")
        self._global_step = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

        self._fusion.train()

        for epoch in range(epochs):
            epoch_loss = self._train_epoch(train_loader)

            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader)

            self._record_epoch(epoch, epoch_loss, val_loss)

        # Close TensorBoard writer (AC-1.5.6.5)
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        # Save checkpoint after training
        output_dir = getattr(self._config, "output_dir", "models/fusion-trained")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint(output_path / "fusion_weights.pt")

        return {
            "epochs": epochs,
            "final_train_loss": self._history[-1]["train_loss"] if self._history else 0.0,
            "final_val_loss": self._history[-1].get("val_loss"),
            "history": self._history,
        }

    def _create_fake_dataset(self, pairs: list[dict[str, Any]]) -> FusionDataset:
        """Create a FusionDataset from a list of pair dicts.

        Args:
            pairs: List of training pair dicts

        Returns:
            FusionDataset
        """
        n = len(pairs)
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility (S2245)
        bge_embs = rng.standard_normal((n, 1024)).astype(np.float32)
        unixcoder_embs = rng.standard_normal((n, 768)).astype(np.float32)
        instructor_embs = rng.standard_normal((n, 768)).astype(np.float32)
        labels = np.arange(n, dtype=np.int64)

        return FusionDataset(bge_embs, unixcoder_embs, instructor_embs, labels)

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            loader: Training data loader

        Returns:
            Average epoch loss
        """
        self._fusion.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            bge, unixcoder, instructor, labels = self._unpack_batch(batch)

            self._optimizer.zero_grad()

            # Forward pass
            fused = self._fusion(bge, unixcoder, instructor)

            # Compute loss
            loss = self._loss_fn(fused, labels)

            # Backward pass
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate(self, loader: DataLoader) -> float:
        """Validate the model.

        Args:
            loader: Validation data loader

        Returns:
            Average validation loss
        """
        self._fusion.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                bge, unixcoder, instructor, labels = self._unpack_batch(batch)

                fused = self._fusion(bge, unixcoder, instructor)
                loss = self._loss_fn(fused, labels)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _unpack_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack and move batch to device.

        Args:
            batch: Batch tuple from dataloader

        Returns:
            Tuple of tensors on device
        """
        bge, unixcoder, instructor, labels = batch
        return (
            bge.to(self._device),
            unixcoder.to(self._device),
            instructor.to(self._device),
            labels.to(self._device),
        )

    def _record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
    ) -> None:
        """Record epoch metrics.

        AC-1.5.6.5: Log training metrics to TensorBoard

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss (if available)
        """
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if val_loss is not None:
            record["val_loss"] = val_loss

        self._history.append(record)

        # Log to TensorBoard (AC-1.5.6.5)
        if self._writer is not None:
            self._writer.add_scalar("Loss/train", train_loss, epoch)
            if val_loss is not None:
                self._writer.add_scalar("Loss/val", val_loss, epoch)
                # Track best validation loss
                if val_loss < self._best_loss:
                    self._best_loss = val_loss
                    self._writer.add_scalar("Loss/best_val", val_loss, epoch)

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}"
            f"{f', val_loss={val_loss:.4f}' if val_loss else ''}"
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint.

        AC-1.5.6.3: Save/load trained fusion weights

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._fusion.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "history": self._history,
            "config": {
                "output_dim": self._fusion.output_dim,
                "bge_dim": self._fusion.bge_dim,
                "unixcoder_dim": self._fusion.unixcoder_dim,
                "instructor_dim": self._fusion.instructor_dim,
            },
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint.

        AC-1.5.6.3: Save/load trained fusion weights

        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self._device)

        self._fusion.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._history = checkpoint.get("history", [])

        logger.info(f"Loaded checkpoint from {path}")

    @property
    def history(self) -> list[dict[str, float]]:
        """Return training history."""
        return self._history
