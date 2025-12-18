"""
Multi-Modal Embedding Architecture

WBS: EEP-1.5 - Multi-Modal Embedding Architecture
Repository: Code-Orchestrator-Service (Sous Chef)

This package implements the Layered + Fine-Tuned + Learned Fusion embedding strategy:
- BGE-large (fine-tuned) - Text/NL similarity
- UniXcoder (fine-tuned) - Code block similarity
- Instructor-XL - Domain-aware concept embeddings
- Learned Fusion Layer - PyTorch cross-attention + MLP fusion

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Constants defined in config.py
- S3776: Cognitive complexity managed via helper methods
- S1172: No unused parameters
- S6903: Namespaced exceptions (no shadowing)
- #12: Repository Pattern with FakeClients for testing
"""

from src.models.embedding.config import (
    DIM_BGE,
    DIM_FUSED,
    DIM_FUSION_OUTPUT,
    DIM_INSTRUCTOR,
    DIM_UNIXCODER,
    DOMAIN_INSTRUCTIONS,
    FusionModelConfig,
    MODEL_BGE_LARGE,
    MODEL_INSTRUCTOR_XL,
    MODEL_UNIXCODER,
)
from src.models.embedding.bge_embedder import BGEEmbedder
from src.models.embedding.bge_trainer import BGETrainer
from src.models.embedding.evaluation import (
    BenchmarkResult,
    EmbeddingBenchmark,
    ModelComparison,
    load_enriched_ground_truth,
    mean_average_precision,
    recall_at_k,
)
from src.models.embedding.fakes import (
    FakeBGEEmbedder,
    FakeFusionLayer,
    FakeInstructorEmbedder,
    FakeUniXcoderEmbedder,
)
from src.models.embedding.fusion import FusionLayer
from src.models.embedding.fusion.fusion_trainer import (
    ContrastiveLoss,
    FusionDataset,
    FusionTrainer,
)
from src.models.embedding.instructor_embedder import InstructorEmbedder
from src.models.embedding.training_data import TrainingPairGenerator
from src.models.embedding.unixcoder_embedder import UniXcoderEmbedder
from src.models.embedding.unixcoder_trainer import UniXcoderTrainer

__all__ = [
    # Config
    "FusionModelConfig",
    "MODEL_BGE_LARGE",
    "MODEL_UNIXCODER",
    "MODEL_INSTRUCTOR_XL",
    "DIM_BGE",
    "DIM_UNIXCODER",
    "DIM_INSTRUCTOR",
    "DIM_FUSED",
    "DIM_FUSION_OUTPUT",
    "DOMAIN_INSTRUCTIONS",
    # Embedders
    "BGEEmbedder",
    "UniXcoderEmbedder",
    "InstructorEmbedder",
    # Trainers
    "BGETrainer",
    "UniXcoderTrainer",
    "FusionTrainer",
    # Fusion
    "FusionLayer",
    "FusionDataset",
    "ContrastiveLoss",
    # Evaluation
    "EmbeddingBenchmark",
    "ModelComparison",
    "BenchmarkResult",
    "recall_at_k",
    "mean_average_precision",
    "load_enriched_ground_truth",
    # Fakes (for testing)
    "FakeBGEEmbedder",
    "FakeUniXcoderEmbedder",
    "FakeInstructorEmbedder",
    "FakeFusionLayer",
    # Data
    "TrainingPairGenerator",
]
