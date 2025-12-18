"""
Embedding Configuration Module

WBS: EEP-1.5 - Multi-Modal Embedding Architecture
Pydantic Settings for embedding model configuration.

Per Comp_Static_Analysis_Report: Pydantic Settings Pattern
- Loads from environment variables with env_prefix
- Provides sensible defaults
- Type-safe configuration

Anti-Patterns Avoided:
- S1192: All constants defined here and reused
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

# Model identifiers
MODEL_BGE_LARGE: str = "BAAI/bge-large-en-v1.5"
MODEL_UNIXCODER: str = "microsoft/unixcoder-base"
MODEL_INSTRUCTOR_XL: str = "hkunlp/instructor-xl"

# Embedding dimensions
DIM_BGE: int = 1024
DIM_UNIXCODER: int = 768
DIM_INSTRUCTOR: int = 768
DIM_FUSED: int = 512  # Output dimension after fusion
DIM_FUSION_OUTPUT: int = DIM_FUSED  # Alias for consistency

# Training constants
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_EPOCHS: int = 5
DEFAULT_LEARNING_RATE: float = 2e-5

# Domain instruction templates
DOMAIN_INSTRUCTIONS: dict[str, str] = {
    "ai-ml": "Represent the AI/ML concept for retrieval:",
    "python": "Represent the Python programming concept for retrieval:",
    "architecture": "Represent the software architecture pattern for retrieval:",
    "default": "Represent the technical concept for retrieval:",
}


# =============================================================================
# Pydantic Settings Classes
# =============================================================================


class FusionModelConfig(BaseSettings):
    """Configuration for multi-modal fusion model.

    Loads from environment variables with EEP15_ prefix.

    Attributes:
        bge_model: BGE model identifier
        unixcoder_model: UniXcoder model identifier
        instructor_model: Instructor model identifier
        output_dim: Fused embedding output dimension
        num_attention_heads: Cross-attention heads
        dropout: Dropout rate
        device: Torch device (cuda/cpu/mps)

    Example:
        >>> config = FusionModelConfig()
        >>> config.bge_model
        'BAAI/bge-large-en-v1.5'

        # Override via environment:
        # EEP15_BGE_MODEL=custom/model
        # EEP15_OUTPUT_DIM=256
    """

    model_config = SettingsConfigDict(
        env_prefix="EEP15_",
        env_file=".env",
        extra="ignore",
    )

    # Model identifiers
    bge_model: str = MODEL_BGE_LARGE
    unixcoder_model: str = MODEL_UNIXCODER
    instructor_model: str = MODEL_INSTRUCTOR_XL

    # Embedding dimensions
    bge_dim: int = DIM_BGE
    unixcoder_dim: int = DIM_UNIXCODER
    instructor_dim: int = DIM_INSTRUCTOR
    output_dim: int = DIM_FUSED

    # Architecture parameters
    num_attention_heads: int = 8
    dropout: float = 0.1
    hidden_dim: int = 768

    # Training parameters
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE

    # Inference
    device: str = "cpu"
    max_length: int = 512

    @property
    def bge_model_id(self) -> str:
        """Alias for bge_model for API consistency."""
        return self.bge_model

    @property
    def unixcoder_model_id(self) -> str:
        """Alias for unixcoder_model for API consistency."""
        return self.unixcoder_model

    @property
    def instructor_model_id(self) -> str:
        """Alias for instructor_model for API consistency."""
        return self.instructor_model

    @property
    def fusion_output_dim(self) -> int:
        """Alias for output_dim for API consistency."""
        return self.output_dim


class BGETrainingConfig(BaseSettings):
    """Configuration for BGE fine-tuning.

    Attributes:
        model_name: Base model to fine-tune
        output_dir: Directory to save checkpoints
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_ratio: Warmup steps ratio
    """

    model_config = SettingsConfigDict(
        env_prefix="BGE_",
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    model_name: str = MODEL_BGE_LARGE
    output_dir: str | Path = "models/bge-finetuned"
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    fp16: bool = True
    save_strategy: str = "epoch"
    save_total_limit: int = 2


class UniXcoderTrainingConfig(BaseSettings):
    """Configuration for UniXcoder fine-tuning.

    Attributes:
        model_name: Base model to fine-tune
        output_dir: Directory to save checkpoints
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
    """

    model_config = SettingsConfigDict(
        env_prefix="UNIXCODER_",
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    model_name: str = MODEL_UNIXCODER
    output_dir: str | Path = "models/unixcoder-finetuned"
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-5
    max_length: int = 512


class FusionTrainingConfig(BaseSettings):
    """Configuration for fusion model training.

    Attributes:
        output_dir: Directory to save checkpoints
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        temperature: InfoNCE temperature
    """

    model_config = SettingsConfigDict(
        env_prefix="FUSION_",
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    output_dir: str | Path = "models/fusion-trained"
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    patience: int = 3
    temperature: float = 0.07
