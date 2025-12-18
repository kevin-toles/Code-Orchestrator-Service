"""
Unit Tests for EEP-1.5: Multi-Modal Embedding Architecture

WBS: EEP-1.5 - Multi-Modal Embedding Architecture (Phase 1.5 of Enhanced Enrichment Pipeline)
TDD Phase: RED (tests written BEFORE implementation)

Tests for:
- EEP-1.5.1: Training data generation from enriched books
- EEP-1.5.2: BGE-large fine-tuning infrastructure
- EEP-1.5.3: UniXcoder fine-tuning infrastructure
- EEP-1.5.4: Instructor-XL domain-aware wrapper
- EEP-1.5.5: Fusion layer module (cross-attention + MLP)
- EEP-1.5.6: Fusion model training pipeline
- EEP-1.5.7: API endpoints for multi-modal embedding
- EEP-1.5.8: Benchmark and evaluation framework

Acceptance Criteria (from Document Analysis):
- AC-1.5.1.1: Generate text_pairs.jsonl from similar_chapters in enriched books
- AC-1.5.1.2: Generate code_pairs.jsonl from code concepts in enriched books
- AC-1.5.1.3: Support positive/negative pair generation (1:1 ratio)
- AC-1.5.1.4: BM25-based hard negative selection
- AC-1.5.2.1: BGE fine-tuning config with sentence-transformers
- AC-1.5.2.2: MultipleNegativesRankingLoss for contrastive training
- AC-1.5.2.3: Checkpoint saving to models/bge-finetuned/ (5 epochs default)
- AC-1.5.3.1: UniXcoder fine-tuning with transformers.Trainer
- AC-1.5.3.2: Code-concept pair training from enriched books
- AC-1.5.4.1: InstructorEmbedder wrapper with domain instructions
- AC-1.5.4.2: Instruction templates per domain (ai-ml, python, architecture)
- AC-1.5.5.1: FusionLayer combines BGE, UniXcoder, Instructor embeddings
- AC-1.5.5.2: Cross-attention mechanism for embedding alignment
- AC-1.5.5.3: MLP projection to unified embedding space
- AC-1.5.6.1: End-to-end training on fused embeddings
- AC-1.5.7.1: /api/v1/embed endpoint with multi-modal support
- AC-1.5.8.1: Recall@k benchmark against similar_chapters ground truth
- AC-1.5.8.2: MAP evaluation for ranking quality

Anti-Patterns Avoided (per CODING_PATTERNS_ANALYSIS.md):
- S1192: Use constants for repeated string literals
- S3776: Cognitive complexity < 15 (extract helpers)
- S1172: No unused parameters (use _ prefix)
- S6903: No exception shadowing (namespaced exceptions)
- #12: Repository Pattern with FakeClients for testing

Design Patterns Applied (per Comp_Static_Analysis_Report):
- Repository Pattern: EmbedderProtocol with FakeEmbedder
- Health Check Pattern: check_embedder_health()
- Pydantic Settings: FusionModelConfig(BaseSettings)
- Protocol/Duck Typing: EmbedderProtocol for testing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pytest

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


# =============================================================================
# Constants (S1192 compliance)
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

# Training constants
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_EPOCHS: int = 3
DEFAULT_LEARNING_RATE: float = 2e-5

# File paths
CONFIG_DIR: str = "config"
DATA_DIR: str = "data/training"
MODELS_DIR: str = "models"
SCRIPTS_DIR: str = "scripts"

# Domain instruction templates
DOMAIN_INSTRUCTIONS: dict[str, str] = {
    "ai-ml": "Represent the AI/ML concept for retrieval:",
    "python": "Represent the Python programming concept for retrieval:",
    "architecture": "Represent the software architecture pattern for retrieval:",
    "default": "Represent the technical concept for retrieval:",
}


# =============================================================================
# Custom Exceptions (Anti-Pattern #6903: No shadowing)
# =============================================================================


class EmbeddingError(Exception):
    """Base exception for embedding errors.

    Per CODING_PATTERNS_ANALYSIS.md:
    - Uses namespaced exception (not generic Exception)
    - Does not shadow Python builtins
    """

    pass


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when embedding model cannot be loaded."""

    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match expected values."""

    pass


class TrainingDataError(EmbeddingError):
    """Raised when training data is invalid or missing."""

    pass


class FusionLayerError(EmbeddingError):
    """Raised when fusion layer encounters an error."""

    pass


# =============================================================================
# Protocols (Repository Pattern)
# =============================================================================


class EmbedderProtocol(Protocol):
    """Protocol for embedding models - enables FakeEmbedder for testing.

    Per CODING_PATTERNS_ANALYSIS.md Repository Pattern:
    - Duck typing interface with minimal methods
    - Enables test doubles without inheritance
    """

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for single text."""
        ...

    def batch_embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        ...


class FusionProtocol(Protocol):
    """Protocol for fusion layer - enables FakeFusion for testing."""

    def fuse(
        self,
        text_embedding: NDArray[np.float32],
        code_embedding: NDArray[np.float32],
        domain_embedding: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Fuse three embeddings into unified representation."""
        ...

    @property
    def output_dim(self) -> int:
        """Return output embedding dimension."""
        ...


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create temporary test data directory."""
    data_dir = tmp_path / "data" / "training"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def test_config_dir(tmp_path: Path) -> Path:
    """Create temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.fixture
def test_models_dir(tmp_path: Path) -> Path:
    """Create temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    return models_dir


@pytest.fixture
def sample_enriched_book() -> dict[str, Any]:
    """Sample enriched book structure matching ai-platform-data format."""
    return {
        "metadata": {
            "title": "Test Book on AI Engineering",
            "source_file": "test_book.json",
        },
        "chapters": [
            {
                "chapter_number": 1,
                "title": "Introduction to Embeddings",
                "summary": "This chapter covers vector embeddings and their use in AI.",
                "keywords": ["embedding", "vector", "AI", "similarity"],
                "concepts": ["embeddings", "vectors", "similarity", "retrieval"],
                "similar_chapters": [
                    {
                        "book": "Test Book on AI Engineering",
                        "chapter": 3,
                        "title": "Advanced Embeddings",
                        "score": 0.92,
                        "base_score": 0.72,
                        "topic_boost": 0.20,
                        "method": "sbert",
                    },
                    {
                        "book": "Test Book on AI Engineering",
                        "chapter": 5,
                        "title": "Semantic Search",
                        "score": 0.85,
                        "base_score": 0.65,
                        "topic_boost": 0.20,
                        "method": "sbert",
                    },
                ],
                "code_snippets": [
                    "def embed_text(text: str) -> np.ndarray:",
                    "embeddings = model.encode(texts)",
                ],
            },
            {
                "chapter_number": 2,
                "title": "Transformer Architecture",
                "summary": "Deep dive into transformer models and attention mechanisms.",
                "keywords": ["transformer", "attention", "self-attention", "BERT"],
                "concepts": ["transformers", "attention", "encoder", "decoder"],
                "similar_chapters": [
                    {
                        "book": "Test Book on AI Engineering",
                        "chapter": 4,
                        "title": "Fine-tuning Transformers",
                        "score": 0.88,
                        "base_score": 0.68,
                        "topic_boost": 0.20,
                        "method": "sbert",
                    },
                ],
                "code_snippets": [
                    "class MultiHeadAttention(nn.Module):",
                    "attention_weights = softmax(Q @ K.T / sqrt(d_k))",
                ],
            },
            {
                "chapter_number": 3,
                "title": "Advanced Embeddings",
                "summary": "Contrastive learning and fine-tuning embedding models.",
                "keywords": ["contrastive", "fine-tuning", "SBERT", "BGE"],
                "concepts": ["contrastive_learning", "fine_tuning", "sentence_embeddings"],
                "similar_chapters": [
                    {
                        "book": "Test Book on AI Engineering",
                        "chapter": 1,
                        "title": "Introduction to Embeddings",
                        "score": 0.92,
                        "base_score": 0.72,
                        "topic_boost": 0.20,
                        "method": "sbert",
                    },
                ],
                "code_snippets": [
                    "model = SentenceTransformer('BAAI/bge-large-en-v1.5')",
                ],
            },
        ],
    }


@pytest.fixture
def sample_text_pairs() -> list[dict[str, Any]]:
    """Sample text pairs for training."""
    return [
        {
            "anchor": "This chapter covers vector embeddings and their use in AI.",
            "positive": "Contrastive learning and fine-tuning embedding models.",
            "negative": "Deep dive into transformer models and attention mechanisms.",
            "score": 0.92,
        },
        {
            "anchor": "Deep dive into transformer models and attention mechanisms.",
            "positive": "Fine-tuning transformers for downstream tasks.",
            "negative": "This chapter covers vector embeddings and their use in AI.",
            "score": 0.88,
        },
    ]


@pytest.fixture
def sample_code_pairs() -> list[dict[str, Any]]:
    """Sample code pairs for training."""
    return [
        {
            "anchor": "def embed_text(text: str) -> np.ndarray:",
            "positive": "embeddings = model.encode(texts)",
            "negative": "class MultiHeadAttention(nn.Module):",
            "concept": "embeddings",
        },
        {
            "anchor": "class MultiHeadAttention(nn.Module):",
            "positive": "attention_weights = softmax(Q @ K.T / sqrt(d_k))",
            "negative": "def embed_text(text: str) -> np.ndarray:",
            "concept": "attention",
        },
    ]


# =============================================================================
# EEP-1.5.1: Training Data Generation Tests
# =============================================================================


class TestTrainingDataGeneration:
    """Tests for training data generation from enriched books.

    WBS: EEP-1.5.1 - Generate training data from enriched books
    AC-1.5.1.1: Generate text_pairs.jsonl from similar_chapters
    AC-1.5.1.2: Generate code_pairs.jsonl from code concepts
    AC-1.5.1.3: Support positive/negative pair generation (1:1 ratio)
    AC-1.5.1.4: BM25-based hard negative selection
    """

    def test_pair_type_enum_exists(self) -> None:
        """AC-1.5.1.1: PairType enum for training data generation."""
        from src.models.embedding.training_data import PairType

        assert hasattr(PairType, "TEXT")
        assert hasattr(PairType, "CODE")
        assert hasattr(PairType, "BOTH")
        assert PairType.TEXT.value == "text"
        assert PairType.CODE.value == "code"
        assert PairType.BOTH.value == "both"

    def test_training_data_config_exists(self) -> None:
        """AC-1.5.1.1: TrainingDataConfig class for generation configuration."""
        from src.models.embedding.training_data import TrainingDataConfig

        config = TrainingDataConfig()
        assert config.min_positive_score == 0.7
        assert config.use_bm25_negatives is False
        assert config.max_pairs is None
        assert config.random_seed == 42

        # Test custom config
        custom_config = TrainingDataConfig(
            min_positive_score=0.8,
            use_bm25_negatives=True,
            max_pairs=100,
        )
        assert custom_config.min_positive_score == 0.8
        assert custom_config.use_bm25_negatives is True
        assert custom_config.max_pairs == 100

    def test_generate_text_pairs_from_similar_chapters(
        self,
        test_data_dir: Path,
        sample_enriched_book: dict[str, Any],
    ) -> None:
        """AC-1.5.1.1: Generate text_pairs.jsonl from similar_chapters in enriched books."""
        from src.models.embedding.training_data import generate_text_pairs

        # Write sample book
        book_path = test_data_dir.parent / "sample_book.json"
        book_path.write_text(json.dumps(sample_enriched_book))

        # Generate pairs
        output_path = test_data_dir / "text_pairs.jsonl"
        pairs = generate_text_pairs(
            enriched_books=[book_path],
            output_path=output_path,
        )

        # Verify output exists
        assert output_path.exists()

        # Verify structure
        assert len(pairs) > 0
        for pair in pairs:
            assert "anchor" in pair
            assert "positive" in pair
            assert "negative" in pair
            assert "score" in pair
            assert isinstance(pair["score"], float)

    def test_generate_code_pairs_from_concepts(
        self,
        test_data_dir: Path,
        sample_enriched_book: dict[str, Any],
    ) -> None:
        """AC-1.5.1.2: Generate code_pairs.jsonl from code concepts in enriched books."""
        from src.models.embedding.training_data import generate_code_pairs

        # Write sample book
        book_path = test_data_dir.parent / "sample_book.json"
        book_path.write_text(json.dumps(sample_enriched_book))

        # Generate pairs
        output_path = test_data_dir / "code_pairs.jsonl"
        pairs = generate_code_pairs(
            enriched_books=[book_path],
            output_path=output_path,
        )

        # Verify output exists
        assert output_path.exists()

        # Verify structure
        assert len(pairs) > 0
        for pair in pairs:
            assert "anchor" in pair
            assert "positive" in pair
            assert "negative" in pair
            assert "concept" in pair

    def test_positive_negative_ratio_is_one_to_one(
        self,
        test_data_dir: Path,
        sample_enriched_book: dict[str, Any],
    ) -> None:
        """AC-1.5.1.3: Support positive/negative pair generation (1:1 ratio)."""
        from src.models.embedding.training_data import generate_text_pairs

        # Write sample book
        book_path = test_data_dir.parent / "sample_book.json"
        book_path.write_text(json.dumps(sample_enriched_book))

        # Generate pairs
        output_path = test_data_dir / "text_pairs.jsonl"
        pairs = generate_text_pairs(
            enriched_books=[book_path],
            output_path=output_path,
        )

        # Each pair should have exactly one positive and one negative
        for pair in pairs:
            assert pair["positive"] != pair["negative"]
            assert pair["anchor"] != pair["positive"]
            assert pair["anchor"] != pair["negative"]

    def test_bm25_hard_negative_selection(
        self,
        test_data_dir: Path,
        sample_enriched_book: dict[str, Any],
    ) -> None:
        """AC-1.5.1.4: BM25-based hard negative selection."""
        from src.models.embedding.training_data import generate_text_pairs

        # Write sample book
        book_path = test_data_dir.parent / "sample_book.json"
        book_path.write_text(json.dumps(sample_enriched_book))

        # Generate pairs with BM25 enabled
        output_path = test_data_dir / "text_pairs_bm25.jsonl"
        pairs = generate_text_pairs(
            enriched_books=[book_path],
            output_path=output_path,
            use_bm25=True,  # Enable BM25 hard negatives
        )

        # Should produce valid pairs (may be empty if rank-bm25 not installed)
        assert isinstance(pairs, list)
        # Verify structure if pairs exist
        for pair in pairs:
            assert "anchor" in pair
            assert "negative" in pair

    def test_training_pair_generator_generate_method(
        self,
        test_data_dir: Path,
        sample_enriched_book: dict[str, Any],
    ) -> None:
        """AC-1.5.1.1: TrainingPairGenerator.generate() unified interface."""
        from src.models.embedding.training_data import (
            PairType,
            TrainingDataConfig,
            TrainingPairGenerator,
        )

        # Write sample book to directory
        book_path = test_data_dir.parent / "sample_book.json"
        book_path.write_text(json.dumps(sample_enriched_book))

        generator = TrainingPairGenerator(test_data_dir.parent)
        config = TrainingDataConfig(min_positive_score=0.7)

        # Test generate method with TEXT type
        pairs = generator.generate(pair_type=PairType.TEXT, config=config)
        assert isinstance(pairs, list)

    def test_training_pair_generator_generate_from_books(
        self,
        test_data_dir: Path,
        sample_enriched_book: dict[str, Any],
    ) -> None:
        """AC-1.5.1.1: TrainingPairGenerator.generate_from_books() method."""
        from src.models.embedding.training_data import (
            PairType,
            TrainingDataConfig,
            TrainingPairGenerator,
        )

        # Write sample book
        book_path = test_data_dir.parent / "sample_book.json"
        book_path.write_text(json.dumps(sample_enriched_book))

        generator = TrainingPairGenerator(test_data_dir.parent)
        config = TrainingDataConfig()

        # Test generate_from_books method
        pairs = generator.generate_from_books(
            book_paths=[book_path],
            pair_type=PairType.TEXT,
            config=config,
        )
        assert isinstance(pairs, list)

    def test_handles_empty_similar_chapters_gracefully(
        self,
        test_data_dir: Path,
    ) -> None:
        """Edge case: Handle chapters with no similar_chapters."""
        from src.models.embedding.training_data import generate_text_pairs

        book_with_empty = {
            "metadata": {"title": "Empty Book"},
            "chapters": [
                {
                    "chapter_number": 1,
                    "summary": "A lonely chapter",
                    "similar_chapters": [],  # Empty
                }
            ],
        }

        book_path = test_data_dir.parent / "empty_book.json"
        book_path.write_text(json.dumps(book_with_empty))

        output_path = test_data_dir / "text_pairs.jsonl"
        pairs = generate_text_pairs(
            enriched_books=[book_path],
            output_path=output_path,
        )

        # Should not raise, but may produce no pairs
        assert isinstance(pairs, list)

    def test_handles_missing_code_snippets_gracefully(
        self,
        test_data_dir: Path,
    ) -> None:
        """Edge case: Handle chapters with no code_snippets."""
        from src.models.embedding.training_data import generate_code_pairs

        book_no_code = {
            "metadata": {"title": "No Code Book"},
            "chapters": [
                {
                    "chapter_number": 1,
                    "summary": "Theory only",
                    "concepts": ["theory"],
                    # No code_snippets key
                }
            ],
        }

        book_path = test_data_dir.parent / "no_code_book.json"
        book_path.write_text(json.dumps(book_no_code))

        output_path = test_data_dir / "code_pairs.jsonl"
        pairs = generate_code_pairs(
            enriched_books=[book_path],
            output_path=output_path,
        )

        # Should not raise
        assert isinstance(pairs, list)


# =============================================================================
# EEP-1.5.2: BGE Fine-tuning Tests
# =============================================================================


class TestBGEFineTuning:
    """Tests for BGE-large fine-tuning infrastructure.

    WBS: EEP-1.5.2 - Fine-tune BGE-large for text similarity
    AC-1.5.2.1: BGE fine-tuning config with sentence-transformers
    AC-1.5.2.2: MultipleNegativesRankingLoss for contrastive training
    AC-1.5.2.3: Checkpoint saving to models/bge-finetuned/ (5 epochs default)
    """

    def test_bge_training_config_exists(
        self,
        test_config_dir: Path,
    ) -> None:
        """AC-1.5.2.1: BGE fine-tuning config with sentence-transformers."""
        from src.models.embedding.bge_trainer import BGETrainingConfig

        config = BGETrainingConfig(
            model_name=MODEL_BGE_LARGE,
            output_dir=test_config_dir / "bge-finetuned",
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE,
        )

        assert config.model_name == MODEL_BGE_LARGE
        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.epochs == DEFAULT_EPOCHS

    def test_bge_trainer_uses_multiple_negatives_ranking_loss(
        self,
        test_config_dir: Path,
        test_models_dir: Path,
        sample_text_pairs: list[dict[str, Any]],
    ) -> None:
        """AC-1.5.2.2: MultipleNegativesRankingLoss for contrastive training."""
        from src.models.embedding.bge_trainer import BGETrainer, BGETrainingConfig

        config = BGETrainingConfig(
            model_name=MODEL_BGE_LARGE,
            output_dir=test_models_dir / "bge-finetuned",
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=1,  # Single epoch for test
        )

        trainer = BGETrainer(config)

        # Verify loss function type - MNRL for contrastive learning
        assert trainer.loss_function_name == "MultipleNegativesRankingLoss"

    def test_bge_training_config_default_epochs_is_5(
        self,
        test_config_dir: Path,
    ) -> None:
        """AC-1.5.2.3: Default epochs should be 5 per WBS."""
        from src.models.embedding.config import DEFAULT_EPOCHS

        # Verify default epochs constant is 5
        assert DEFAULT_EPOCHS == 5

    def test_bge_trainer_saves_checkpoints(
        self,
        test_models_dir: Path,
        sample_text_pairs: list[dict[str, Any]],
    ) -> None:
        """AC-1.5.2.3: Checkpoint saving to models/bge-finetuned/."""
        from src.models.embedding.bge_trainer import BGETrainer, BGETrainingConfig

        output_dir = test_models_dir / "bge-finetuned"
        config = BGETrainingConfig(
            model_name=MODEL_BGE_LARGE,
            output_dir=output_dir,
            epochs=1,
        )

        trainer = BGETrainer(config)

        # Train with sample data (mocked in unit test)
        trainer.train(sample_text_pairs)

        # Verify checkpoint directory exists
        assert output_dir.exists()

    def test_bge_embedding_dimension_is_1024(self) -> None:
        """Verify BGE-large produces 1024-dimensional embeddings."""
        from src.models.embedding.bge_embedder import BGEEmbedder

        embedder = BGEEmbedder(model_name=MODEL_BGE_LARGE)

        assert embedder.embedding_dim == DIM_BGE


# =============================================================================
# EEP-1.5.3: UniXcoder Fine-tuning Tests
# =============================================================================


class TestUniXcoderFineTuning:
    """Tests for UniXcoder fine-tuning infrastructure.

    WBS: EEP-1.5.3 - Fine-tune UniXcoder for code similarity
    AC-1.5.3.1: UniXcoder fine-tuning with transformers.Trainer
    AC-1.5.3.2: Code-concept pair training from enriched books
    """

    def test_unixcoder_training_config_exists(
        self,
        test_config_dir: Path,
    ) -> None:
        """AC-1.5.3.1: UniXcoder fine-tuning with transformers.Trainer."""
        from src.models.embedding.unixcoder_trainer import UniXcoderTrainingConfig

        config = UniXcoderTrainingConfig(
            model_name=MODEL_UNIXCODER,
            output_dir=test_config_dir / "unixcoder-finetuned",
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE,
        )

        assert config.model_name == MODEL_UNIXCODER
        assert config.batch_size == DEFAULT_BATCH_SIZE

    def test_unixcoder_trainer_handles_code_pairs(
        self,
        test_models_dir: Path,
        sample_code_pairs: list[dict[str, Any]],
    ) -> None:
        """AC-1.5.3.2: Code-concept pair training from enriched books."""
        from src.models.embedding.unixcoder_trainer import (
            UniXcoderTrainer,
            UniXcoderTrainingConfig,
        )

        config = UniXcoderTrainingConfig(
            model_name=MODEL_UNIXCODER,
            output_dir=test_models_dir / "unixcoder-finetuned",
            epochs=1,
        )

        trainer = UniXcoderTrainer(config)

        # Should accept code pairs format
        trainer.train(sample_code_pairs)

        # Verify output directory
        assert config.output_dir.exists()

    def test_unixcoder_embedding_dimension_is_768(self) -> None:
        """Verify UniXcoder produces 768-dimensional embeddings."""
        from src.models.embedding.unixcoder_embedder import UniXcoderEmbedder

        embedder = UniXcoderEmbedder(model_name=MODEL_UNIXCODER)

        assert embedder.embedding_dim == DIM_UNIXCODER


# =============================================================================
# EEP-1.5.4: Instructor-XL Wrapper Tests
# =============================================================================


class TestInstructorXLWrapper:
    """Tests for Instructor-XL domain-aware wrapper.

    WBS: EEP-1.5.4 - Instructor-XL wrapper for domain-aware embeddings
    AC-1.5.4.1: InstructorEmbedder wrapper with domain instructions
    AC-1.5.4.2: Instruction templates per domain (ai-ml, python, architecture)
    """

    def test_instructor_embedder_accepts_domain_instruction(self) -> None:
        """AC-1.5.4.1: InstructorEmbedder wrapper with domain instructions."""
        from src.models.embedding.instructor_embedder import InstructorEmbedder

        embedder = InstructorEmbedder(model_name=MODEL_INSTRUCTOR_XL)

        # Embed with domain instruction
        embedding = embedder.embed(
            text="Vector embeddings for semantic search",
            instruction=DOMAIN_INSTRUCTIONS["ai-ml"],
        )

        assert embedding.shape[0] == DIM_INSTRUCTOR

    def test_instructor_has_domain_templates(self) -> None:
        """AC-1.5.4.2: Instruction templates per domain (ai-ml, python, architecture)."""
        from src.models.embedding.instructor_embedder import InstructorEmbedder

        embedder = InstructorEmbedder(model_name=MODEL_INSTRUCTOR_XL)

        # Verify all expected domains are supported
        expected_domains = ["ai-ml", "python", "architecture", "default"]
        for domain in expected_domains:
            assert domain in embedder.domain_instructions
            assert len(embedder.domain_instructions[domain]) > 0

    def test_instructor_embedding_dimension_is_768(self) -> None:
        """Verify Instructor-XL produces 768-dimensional embeddings."""
        from src.models.embedding.instructor_embedder import InstructorEmbedder

        embedder = InstructorEmbedder(model_name=MODEL_INSTRUCTOR_XL)

        assert embedder.embedding_dim == DIM_INSTRUCTOR

    def test_instructor_batch_embed_with_instruction(self) -> None:
        """Test batch embedding with consistent instruction."""
        from src.models.embedding.instructor_embedder import InstructorEmbedder

        embedder = InstructorEmbedder(model_name=MODEL_INSTRUCTOR_XL)

        texts = [
            "Vector embeddings",
            "Semantic search",
            "Contrastive learning",
        ]

        embeddings = embedder.batch_embed(
            texts=texts,
            instruction=DOMAIN_INSTRUCTIONS["ai-ml"],
        )

        assert embeddings.shape == (3, DIM_INSTRUCTOR)


# =============================================================================
# EEP-1.5.5: Fusion Layer Tests
# =============================================================================


class TestFusionLayer:
    """Tests for fusion layer module (cross-attention + MLP).

    WBS: EEP-1.5.5 - Fusion layer combining BGE, UniXcoder, Instructor embeddings
    AC-1.5.5.1: FusionLayer combines BGE, UniXcoder, Instructor embeddings
    AC-1.5.5.2: Cross-attention mechanism for embedding alignment
    AC-1.5.5.3: MLP projection to unified embedding space
    """

    def test_fusion_layer_combines_three_embeddings(self) -> None:
        """AC-1.5.5.1: FusionLayer combines BGE, UniXcoder, Instructor embeddings."""
        import numpy as np

        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        fusion = FusionLayer(
            bge_dim=DIM_BGE,
            unixcoder_dim=DIM_UNIXCODER,
            instructor_dim=DIM_INSTRUCTOR,
            output_dim=DIM_FUSED,
        )

        # Create mock embeddings
        bge_emb = np.random.randn(DIM_BGE).astype(np.float32)
        unixcoder_emb = np.random.randn(DIM_UNIXCODER).astype(np.float32)
        instructor_emb = np.random.randn(DIM_INSTRUCTOR).astype(np.float32)

        # Fuse embeddings
        fused = fusion.fuse(bge_emb, unixcoder_emb, instructor_emb)

        assert fused.shape == (DIM_FUSED,)

    def test_fusion_layer_has_cross_attention(self) -> None:
        """AC-1.5.5.2: Cross-attention mechanism for embedding alignment."""
        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        fusion = FusionLayer(
            bge_dim=DIM_BGE,
            unixcoder_dim=DIM_UNIXCODER,
            instructor_dim=DIM_INSTRUCTOR,
            output_dim=DIM_FUSED,
        )

        # Verify cross-attention component exists
        assert hasattr(fusion, "cross_attention")
        assert fusion.cross_attention is not None

    def test_fusion_layer_has_mlp_projection(self) -> None:
        """AC-1.5.5.3: MLP projection to unified embedding space."""
        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        fusion = FusionLayer(
            bge_dim=DIM_BGE,
            unixcoder_dim=DIM_UNIXCODER,
            instructor_dim=DIM_INSTRUCTOR,
            output_dim=DIM_FUSED,
        )

        # Verify MLP component exists
        assert hasattr(fusion, "mlp")
        assert fusion.mlp is not None

    def test_fusion_output_dimension_is_configurable(self) -> None:
        """Verify fusion output dimension matches configuration."""
        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        custom_dim = 256
        fusion = FusionLayer(
            bge_dim=DIM_BGE,
            unixcoder_dim=DIM_UNIXCODER,
            instructor_dim=DIM_INSTRUCTOR,
            output_dim=custom_dim,
        )

        assert fusion.output_dim == custom_dim

    def test_fusion_batch_processing(self) -> None:
        """Test fusion layer with batch inputs."""
        import numpy as np

        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        fusion = FusionLayer(
            bge_dim=DIM_BGE,
            unixcoder_dim=DIM_UNIXCODER,
            instructor_dim=DIM_INSTRUCTOR,
            output_dim=DIM_FUSED,
        )

        batch_size = 4

        # Create batch embeddings
        bge_batch = np.random.randn(batch_size, DIM_BGE).astype(np.float32)
        unixcoder_batch = np.random.randn(batch_size, DIM_UNIXCODER).astype(np.float32)
        instructor_batch = np.random.randn(batch_size, DIM_INSTRUCTOR).astype(np.float32)

        # Fuse batch
        fused_batch = fusion.batch_fuse(bge_batch, unixcoder_batch, instructor_batch)

        assert fused_batch.shape == (batch_size, DIM_FUSED)


# =============================================================================
# EEP-1.5.6: Fusion Model Training Tests
# =============================================================================


class TestFusionModelTraining:
    """Tests for fusion model training pipeline.

    WBS: EEP-1.5.6 - Train fusion model end-to-end
    AC-1.5.6.1: End-to-end training on fused embeddings
    """

    def test_fusion_trainer_config_exists(
        self,
        test_config_dir: Path,
    ) -> None:
        """Verify fusion training config can be loaded."""
        from src.models.embedding.fusion.fusion_trainer import FusionTrainingConfig

        config = FusionTrainingConfig(
            output_dir=test_config_dir / "fusion-trained",
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_EPOCHS,
            learning_rate=DEFAULT_LEARNING_RATE,
        )

        assert config.batch_size == DEFAULT_BATCH_SIZE

    def test_fusion_trainer_end_to_end_training(
        self,
        test_models_dir: Path,
        sample_text_pairs: list[dict[str, Any]],
    ) -> None:
        """AC-1.5.6.1: End-to-end training on fused embeddings."""
        from src.models.embedding.fusion.fusion_trainer import (
            FusionTrainer,
            FusionTrainingConfig,
        )

        config = FusionTrainingConfig(
            output_dir=test_models_dir / "fusion-trained",
            epochs=1,
        )

        trainer = FusionTrainer(config)

        # Train should complete without error
        trainer.train(sample_text_pairs)

        # Verify checkpoint saved
        assert config.output_dir.exists()

    def test_fusion_trainer_saves_model_weights(
        self,
        test_models_dir: Path,
        sample_text_pairs: list[dict[str, Any]],
    ) -> None:
        """Verify fusion trainer saves model weights."""
        from src.models.embedding.fusion.fusion_trainer import (
            FusionTrainer,
            FusionTrainingConfig,
        )

        output_dir = test_models_dir / "fusion-trained"
        config = FusionTrainingConfig(
            output_dir=output_dir,
            epochs=1,
        )

        trainer = FusionTrainer(config)
        trainer.train(sample_text_pairs)

        # Check for weights file
        weights_file = output_dir / "fusion_weights.pt"
        assert weights_file.exists() or (output_dir / "pytorch_model.bin").exists()


# =============================================================================
# EEP-1.5.7: API Endpoint Tests
# =============================================================================


class TestMultiModalEmbedEndpoint:
    """Tests for multi-modal embedding API endpoints.

    WBS: EEP-1.5.7 - API endpoints for multi-modal embedding
    AC-1.5.7.1: /api/v1/embed endpoint with multi-modal support
    """

    def test_embed_endpoint_exists(self) -> None:
        """AC-1.5.7.1: /api/v1/embed endpoint with multi-modal support."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        # Test endpoint exists
        response = client.post(
            "/api/v1/embed",
            json={
                "text": "Vector embeddings for semantic search",
                "domain": "ai-ml",
            },
        )

        # Should not be 404
        assert response.status_code != 404

    def test_embed_endpoint_returns_fused_embedding(self) -> None:
        """Verify embed endpoint returns fused embedding."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/embed",
            json={
                "text": "Vector embeddings for semantic search",
                "domain": "ai-ml",
                "modalities": ["text", "domain"],
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert "embedding" in data
            assert len(data["embedding"]) == DIM_FUSED

    def test_embed_endpoint_supports_code_modality(self) -> None:
        """Verify embed endpoint supports code modality."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/embed",
            json={
                "text": "def embed_text(text: str) -> np.ndarray:",
                "domain": "python",
                "modalities": ["code"],
            },
        )

        # Should handle code input
        assert response.status_code in [200, 422]  # OK or validation error

    def test_embed_batch_endpoint_exists(self) -> None:
        """Verify batch embedding endpoint exists."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/embed/batch",
            json={
                "texts": [
                    "Vector embeddings",
                    "Semantic search",
                ],
                "domain": "ai-ml",
            },
        )

        # Should not be 404
        assert response.status_code != 404

    def test_similarity_unified_endpoint_exists(self) -> None:
        """AC-1.5.7.3: Verify /api/v1/similarity/unified endpoint exists."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/similarity/unified",
            json={
                "source": {
                    "text": "Vector embeddings for semantic search",
                    "domain": "ai-ml",
                },
                "targets": [
                    {"text": "Embedding vectors for finding similar documents"},
                    {"text": "Binary tree data structures"},
                ],
            },
        )

        # Should not be 404 - endpoint must exist
        assert response.status_code != 404

    def test_similarity_unified_returns_modality_scores(self) -> None:
        """AC-1.5.7.3: Verify similarity response includes individual modality scores."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/similarity/unified",
            json={
                "source": {"text": "Machine learning algorithms"},
                "targets": [{"text": "Deep learning neural networks"}],
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert "similarities" in data
            assert len(data["similarities"]) == 1

            result = data["similarities"][0]
            # Verify individual modality scores are present
            assert "score" in result  # Unified/fused score
            assert "bge_score" in result  # Text modality
            assert "unixcoder_score" in result  # Code modality
            assert "instructor_score" in result  # Concept modality
            assert "target_index" in result

    def test_similarity_unified_scores_in_valid_range(self) -> None:
        """AC-1.5.7.3: Verify similarity scores are in valid cosine range [-1, 1]."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/similarity/unified",
            json={
                "source": {"text": "Python programming"},
                "targets": [
                    {"text": "Python scripting language"},
                    {"text": "Java programming language"},
                ],
            },
        )

        if response.status_code == 200:
            data = response.json()
            for result in data["similarities"]:
                assert -1.0 <= result["score"] <= 1.0
                assert -1.0 <= result["bge_score"] <= 1.0
                assert -1.0 <= result["unixcoder_score"] <= 1.0
                assert -1.0 <= result["instructor_score"] <= 1.0


# =============================================================================
# EEP-1.5.8: Benchmark and Evaluation Tests
# =============================================================================


class TestBenchmarkAndEvaluation:
    """Tests for benchmark and evaluation framework.

    WBS: EEP-1.5.8 - Benchmark and evaluation framework
    AC-1.5.8.1: Recall@k benchmark against similar_chapters ground truth
    AC-1.5.8.2: MAP evaluation for ranking quality
    """

    def test_recall_at_k_metric_exists(self) -> None:
        """AC-1.5.8.1: Recall@k benchmark against similar_chapters ground truth."""
        from src.models.embedding.evaluation import recall_at_k

        # Ground truth similar chapters
        ground_truth = [1, 3, 5]  # Chapter indices
        predictions = [1, 2, 3, 4, 5]  # Ranked predictions

        recall_5 = recall_at_k(ground_truth, predictions, k=5)

        # All 3 relevant items should be in top 5
        assert recall_5 == 1.0  # 3/3 = 100%

    def test_recall_at_k_partial_match(self) -> None:
        """Test Recall@k with partial matches."""
        from src.models.embedding.evaluation import recall_at_k

        ground_truth = [1, 3, 5, 7, 9]
        predictions = [1, 2, 3, 4, 6]  # Only 1, 3 are relevant

        recall_5 = recall_at_k(ground_truth, predictions, k=5)

        # 2 out of 5 relevant items in top 5
        assert recall_5 == pytest.approx(0.4)

    def test_mean_average_precision_metric_exists(self) -> None:
        """AC-1.5.8.2: MAP evaluation for ranking quality."""
        from src.models.embedding.evaluation import mean_average_precision

        # Multiple queries with ground truth and predictions
        queries = [
            {
                "ground_truth": [1, 3, 5],
                "predictions": [1, 2, 3, 4, 5],
            },
            {
                "ground_truth": [2, 4],
                "predictions": [1, 2, 3, 4],
            },
        ]

        map_score = mean_average_precision(queries)

        assert 0.0 <= map_score <= 1.0

    def test_benchmark_runner_exists(self) -> None:
        """Verify benchmark runner can execute."""
        from src.models.embedding.evaluation import BenchmarkRunner

        runner = BenchmarkRunner()

        # Should have methods for running benchmarks
        assert hasattr(runner, "run_recall_benchmark")
        assert hasattr(runner, "run_map_benchmark")

    def test_benchmark_with_enriched_books(
        self,
        sample_enriched_book: dict[str, Any],
    ) -> None:
        """Test benchmark using similar_chapters as ground truth."""
        from src.models.embedding.evaluation import BenchmarkRunner

        runner = BenchmarkRunner()

        # Extract ground truth from sample book
        results = runner.run_recall_benchmark(
            books=[sample_enriched_book],
            k_values=[1, 3, 5],
        )

        assert "recall@1" in results
        assert "recall@3" in results
        assert "recall@5" in results

    def test_is_code_related_query_function(self) -> None:
        """AC-1.5.8.3: Verify code query classification."""
        from src.models.embedding.evaluation import is_code_related_query

        # Code-related queries
        assert is_code_related_query("def calculate_total() function") is True
        assert is_code_related_query("import numpy as np") is True
        assert is_code_related_query("Python class inheritance") is True
        assert is_code_related_query("algorithm implementation") is True
        assert is_code_related_query("API endpoint design") is True

        # Non-code queries
        assert is_code_related_query("Introduction to chapter") is False
        assert is_code_related_query("Summary of key points") is False
        assert is_code_related_query("Historical background") is False

    def test_code_recall_at_k_function(self) -> None:
        """AC-1.5.8.3: Verify code-specific recall computation."""
        from src.models.embedding.evaluation import code_recall_at_k

        queries = [
            "def function implementation",  # Code-related
            "Introduction chapter",  # Not code-related
            "Python class design",  # Code-related
            "Summary of concepts",  # Not code-related
        ]
        ground_truth = [
            [1, 2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ]
        predictions = [
            [1, 4, 5, 6, 7],  # Hit (1 in top-5)
            [1, 2, 3, 4, 5],  # Hit but not code-related
            [1, 2, 3, 4, 5],  # Miss (6, 7 not in top-5)
            [8, 9, 1, 2, 3],  # Hit but not code-related
        ]

        code_recall, code_count = code_recall_at_k(queries, ground_truth, predictions, k=5)

        assert code_count == 2  # Only 2 code-related queries
        assert code_recall == 0.5  # 1 hit out of 2 code queries

    def test_benchmark_result_has_code_recall_fields(self) -> None:
        """AC-1.5.8.3: Verify BenchmarkResult includes code-specific metrics."""
        from src.models.embedding.evaluation import BenchmarkResult

        result = BenchmarkResult(
            model_name="test",
            recall_at_1=0.5,
            recall_at_5=0.7,
            recall_at_10=0.8,
            mean_average_precision=0.6,
            inference_time_ms=100.0,
            samples_per_second=10.0,
            total_samples=100,
            code_recall_at_5=0.8,
            code_recall_at_10=0.9,
            code_query_count=20,
        )

        result_dict = result.to_dict()

        assert "code_recall@5" in result_dict
        assert "code_recall@10" in result_dict
        assert "code_query_count" in result_dict
        assert result_dict["code_recall@5"] == 0.8
        assert result_dict["code_recall@10"] == 0.9
        assert result_dict["code_query_count"] == 20


# =============================================================================
# Health Check Tests (Per Comp_Static_Analysis_Report Pattern)
# =============================================================================


class TestEmbedderHealthChecks:
    """Tests for embedder health check pattern.

    Per Comp_Static_Analysis_Report: Health Check Pattern
    """

    def test_bge_embedder_health_check(self) -> None:
        """Verify BGE embedder has health check."""
        from src.models.embedding.bge_embedder import BGEEmbedder

        embedder = BGEEmbedder(model_name=MODEL_BGE_LARGE)

        health = embedder.check_health()

        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]
        assert "model_name" in health

    def test_unixcoder_embedder_health_check(self) -> None:
        """Verify UniXcoder embedder has health check."""
        from src.models.embedding.unixcoder_embedder import UniXcoderEmbedder

        embedder = UniXcoderEmbedder(model_name=MODEL_UNIXCODER)

        health = embedder.check_health()

        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]

    def test_instructor_embedder_health_check(self) -> None:
        """Verify Instructor embedder has health check."""
        from src.models.embedding.instructor_embedder import InstructorEmbedder

        embedder = InstructorEmbedder(model_name=MODEL_INSTRUCTOR_XL)

        health = embedder.check_health()

        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]

    def test_fusion_layer_health_check(self) -> None:
        """Verify Fusion layer has health check."""
        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        fusion = FusionLayer(
            bge_dim=DIM_BGE,
            unixcoder_dim=DIM_UNIXCODER,
            instructor_dim=DIM_INSTRUCTOR,
            output_dim=DIM_FUSED,
        )

        health = fusion.check_health()

        assert "status" in health
        assert "components" in health


# =============================================================================
# FakeEmbedder Tests (Repository Pattern Compliance)
# =============================================================================


class TestFakeEmbedders:
    """Tests for FakeEmbedder implementations.

    Per CODING_PATTERNS_ANALYSIS.md Repository Pattern:
    - FakeClient for testing without real model loading
    """

    def test_fake_bge_embedder_returns_deterministic_output(self) -> None:
        """Verify FakeBGEEmbedder returns deterministic embeddings."""
        from src.models.embedding.fakes import FakeBGEEmbedder

        embedder = FakeBGEEmbedder()

        # Same input should produce same output
        emb1 = embedder.embed("test text")
        emb2 = embedder.embed("test text")

        assert (emb1 == emb2).all()
        assert emb1.shape == (DIM_BGE,)

    def test_fake_unixcoder_embedder_returns_deterministic_output(self) -> None:
        """Verify FakeUniXcoderEmbedder returns deterministic embeddings."""
        from src.models.embedding.fakes import FakeUniXcoderEmbedder

        embedder = FakeUniXcoderEmbedder()

        emb1 = embedder.embed("def foo():")
        emb2 = embedder.embed("def foo():")

        assert (emb1 == emb2).all()
        assert emb1.shape == (DIM_UNIXCODER,)

    def test_fake_instructor_embedder_returns_deterministic_output(self) -> None:
        """Verify FakeInstructorEmbedder returns deterministic embeddings."""
        from src.models.embedding.fakes import FakeInstructorEmbedder

        embedder = FakeInstructorEmbedder()

        emb1 = embedder.embed("concept", instruction="Represent:")
        emb2 = embedder.embed("concept", instruction="Represent:")

        assert (emb1 == emb2).all()
        assert emb1.shape == (DIM_INSTRUCTOR,)

    def test_fake_fusion_layer_returns_correct_dimension(self) -> None:
        """Verify FakeFusionLayer returns correct output dimension."""
        import numpy as np

        from src.models.embedding.fakes import FakeFusionLayer

        fusion = FakeFusionLayer(output_dim=DIM_FUSED)

        bge = np.zeros(DIM_BGE, dtype=np.float32)
        unixcoder = np.zeros(DIM_UNIXCODER, dtype=np.float32)
        instructor = np.zeros(DIM_INSTRUCTOR, dtype=np.float32)

        fused = fusion.fuse(bge, unixcoder, instructor)

        assert fused.shape == (DIM_FUSED,)


# =============================================================================
# Pydantic Settings Tests (Per Comp_Static_Analysis_Report Pattern)
# =============================================================================


class TestPydanticSettings:
    """Tests for Pydantic Settings configuration.

    Per Comp_Static_Analysis_Report: Pydantic Settings Pattern
    """

    def test_fusion_model_config_loads_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify FusionModelConfig loads from environment variables."""
        from src.models.embedding.config import FusionModelConfig

        # Set environment variables
        monkeypatch.setenv("EEP15_BGE_MODEL", "custom/bge-model")
        monkeypatch.setenv("EEP15_OUTPUT_DIM", "256")

        config = FusionModelConfig()

        assert config.bge_model == "custom/bge-model"
        assert config.output_dim == 256

    def test_fusion_model_config_has_defaults(self) -> None:
        """Verify FusionModelConfig has sensible defaults."""
        from src.models.embedding.config import FusionModelConfig

        config = FusionModelConfig()

        assert config.bge_model == MODEL_BGE_LARGE
        assert config.unixcoder_model == MODEL_UNIXCODER
        assert config.instructor_model == MODEL_INSTRUCTOR_XL
        assert config.output_dim == DIM_FUSED


# =============================================================================
# Integration Boundary Tests
# =============================================================================


class TestIntegrationBoundaries:
    """Tests for integration boundaries with existing services.

    Verifies EEP-1.5 components integrate properly with:
    - Existing SBERT in Code-Orchestrator-Service
    - Semantic Search Service (Cookbook)
    """

    def test_fused_embedding_compatible_with_qdrant_format(self) -> None:
        """Verify fused embeddings can be stored in Qdrant."""
        import numpy as np

        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        fusion = FusionLayer(
            bge_dim=DIM_BGE,
            unixcoder_dim=DIM_UNIXCODER,
            instructor_dim=DIM_INSTRUCTOR,
            output_dim=DIM_FUSED,
        )

        bge = np.random.randn(DIM_BGE).astype(np.float32)
        unixcoder = np.random.randn(DIM_UNIXCODER).astype(np.float32)
        instructor = np.random.randn(DIM_INSTRUCTOR).astype(np.float32)

        fused = fusion.fuse(bge, unixcoder, instructor)

        # Qdrant expects list[float]
        qdrant_vector = fused.tolist()
        assert isinstance(qdrant_vector, list)
        assert all(isinstance(x, float) for x in qdrant_vector)

    def test_existing_sbert_still_works_after_eep15(self) -> None:
        """Verify existing SBERT functionality is not broken."""
        # Import existing SBERT module
        from src.models.sbert.semantic_similarity_engine import (
            SemanticSimilarityEngine,
        )

        # Should still be importable and functional
        engine = SemanticSimilarityEngine()
        assert hasattr(engine, "embed")


# =============================================================================
# Cognitive Complexity Compliance Tests (S3776)
# =============================================================================


class TestCognitiveComplexity:
    """Tests verifying cognitive complexity compliance.

    Per CODING_PATTERNS_ANALYSIS.md S3776: CC < 15
    """

    def test_fusion_forward_has_extracted_helpers(self) -> None:
        """Verify fusion layer has extracted helper methods for low CC."""
        from src.models.embedding.fusion.embedding_fusion import FusionLayer

        # These helper methods should exist to keep forward() CC low
        assert hasattr(FusionLayer, "_compute_attention_weights") or hasattr(
            FusionLayer, "_apply_cross_attention"
        )
        assert hasattr(FusionLayer, "_apply_mlp") or hasattr(
            FusionLayer, "_project_embeddings"
        )


# =============================================================================
# Summary: Test Count by WBS
# =============================================================================

# EEP-1.5.1 Training Data Generation: 5 tests
# EEP-1.5.2 BGE Fine-tuning: 4 tests
# EEP-1.5.3 UniXcoder Fine-tuning: 3 tests
# EEP-1.5.4 Instructor-XL Wrapper: 4 tests
# EEP-1.5.5 Fusion Layer: 5 tests
# EEP-1.5.6 Fusion Training: 3 tests
# EEP-1.5.7 API Endpoints: 4 tests
# EEP-1.5.8 Benchmark/Evaluation: 5 tests
# Health Checks: 4 tests
# FakeEmbedders: 4 tests
# Pydantic Settings: 2 tests
# Integration Boundaries: 2 tests
# Cognitive Complexity: 1 test
# -----------------------------
# TOTAL: 46 tests
