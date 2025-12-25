"""
Tests for scripts/train_classifier.py - WBS-AC7.

TDD RED Phase Tests for:
- AC-7.1: Training data loads from validated_term_filter.json
- AC-7.2: Model trained on SBERT embeddings (LogisticRegression)
- AC-7.3: Evaluation reports accuracy, precision, recall, F1
- AC-7.4: Model exported to .joblib file
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import will fail until train_classifier.py is implemented
from scripts.train_classifier import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_NAME,
    EvaluationMetrics,
    TrainingConfig,
    TrainingData,
    evaluate_model,
    prepare_training_data,
    save_model,
    train_classifier,
)

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_aggregated_data() -> dict:
    """Sample validated_term_filter.json structure."""
    return {
        "timestamp": "2025-12-23",
        "description": "Test data",
        "summary": {
            "total_concepts": 5,
            "total_keywords": 5,
        },
        "concepts": [
            "microservices",
            "kubernetes",
            "machine learning",
            "api gateway",
            "load balancer",
        ],
        "keywords": [
            "docker",
            "python",
            "aws",
            "terraform",
            "jenkins",
        ],
    }


@pytest.fixture
def temp_data_file(sample_aggregated_data: dict) -> Path:
    """Create a temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(sample_aggregated_data, f)
        return Path(f.name)


@pytest.fixture
def temp_model_path() -> Path:
    """Create a temporary path for model output."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        return Path(f.name)


# =============================================================================
# AC-7.1: Test prepare_training_data loads taxonomy
# =============================================================================


class TestPrepareTrainingData:
    """Tests for prepare_training_data() - AC-7.1."""

    def test_loads_from_json_file(self, temp_data_file: Path) -> None:
        """AC7.1: prepare_training_data loads from JSON file."""
        result = prepare_training_data(temp_data_file)

        assert isinstance(result, TrainingData)

    def test_extracts_concepts(self, temp_data_file: Path) -> None:
        """AC7.1: Extracts concepts from JSON."""
        result = prepare_training_data(temp_data_file)

        assert "microservices" in result.terms
        assert "kubernetes" in result.terms

    def test_extracts_keywords(self, temp_data_file: Path) -> None:
        """AC7.1: Extracts keywords from JSON."""
        result = prepare_training_data(temp_data_file)

        assert "docker" in result.terms
        assert "python" in result.terms

    def test_returns_training_data_dataclass(self, temp_data_file: Path) -> None:
        """AC7.1: Returns TrainingData with terms and labels."""
        result = prepare_training_data(temp_data_file)

        assert hasattr(result, "terms")
        assert hasattr(result, "labels")
        assert len(result.terms) == len(result.labels)

    def test_file_not_found_raises_error(self) -> None:
        """AC7.1: Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            prepare_training_data(Path("/nonexistent/file.json"))


# =============================================================================
# AC-7.1: Test labels assigned correctly
# =============================================================================


class TestLabelsAssignment:
    """Tests for correct label assignment - AC-7.1."""

    def test_concepts_labeled_as_zero(self, temp_data_file: Path) -> None:
        """AC7.2: Concepts get label 0."""
        result = prepare_training_data(temp_data_file)

        # Find a concept and check its label
        concept_idx = result.terms.index("microservices")
        assert result.labels[concept_idx] == 0

    def test_keywords_labeled_as_one(self, temp_data_file: Path) -> None:
        """AC7.2: Keywords get label 1."""
        result = prepare_training_data(temp_data_file)

        # Find a keyword and check its label
        keyword_idx = result.terms.index("docker")
        assert result.labels[keyword_idx] == 1

    def test_all_concepts_have_label_zero(self, temp_data_file: Path) -> None:
        """AC7.2: All concepts have label 0."""
        result = prepare_training_data(temp_data_file)

        concepts = ["microservices", "kubernetes", "machine learning", "api gateway", "load balancer"]
        for concept in concepts:
            idx = result.terms.index(concept)
            assert result.labels[idx] == 0, f"Concept '{concept}' should have label 0"

    def test_all_keywords_have_label_one(self, temp_data_file: Path) -> None:
        """AC7.2: All keywords have label 1."""
        result = prepare_training_data(temp_data_file)

        keywords = ["docker", "python", "aws", "terraform", "jenkins"]
        for keyword in keywords:
            idx = result.terms.index(keyword)
            assert result.labels[idx] == 1, f"Keyword '{keyword}' should have label 1"

    def test_combined_count_matches(self, temp_data_file: Path) -> None:
        """AC7.2: Total terms = concepts + keywords."""
        result = prepare_training_data(temp_data_file)

        assert len(result.terms) == 10  # 5 concepts + 5 keywords
        assert len(result.labels) == 10


# =============================================================================
# AC-7.2: Test train_classifier returns model
# =============================================================================


class TestTrainClassifier:
    """Tests for train_classifier() - AC-7.2."""

    def test_returns_trained_model(self, temp_data_file: Path) -> None:
        """AC7.3: train_classifier returns a trained model."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        # Should be a scikit-learn model with predict_proba
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_uses_logistic_regression(self, temp_data_file: Path) -> None:
        """AC7.3: Uses LogisticRegression as the classifier."""
        from sklearn.linear_model import LogisticRegression

        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        assert isinstance(model, LogisticRegression)

    def test_model_can_predict(self, temp_data_file: Path) -> None:
        """AC7.3: Model can make predictions on embeddings."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        # Create a dummy embedding (384 dimensions for all-MiniLM-L6-v2)
        dummy_embedding = np.random.rand(1, 384)
        prediction = model.predict(dummy_embedding)

        assert prediction[0] in [0, 1]

    def test_model_returns_probabilities(self, temp_data_file: Path) -> None:
        """AC7.3: Model returns class probabilities."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        dummy_embedding = np.random.rand(1, 384)
        proba = model.predict_proba(dummy_embedding)

        assert proba.shape == (1, 2)  # 2 classes: concept, keyword
        assert 0 <= proba[0][0] <= 1
        assert 0 <= proba[0][1] <= 1

    def test_with_training_config(self, temp_data_file: Path) -> None:
        """AC7.3: Accepts TrainingConfig for hyperparameters."""
        training_data = prepare_training_data(temp_data_file)
        config = TrainingConfig(
            test_size=0.2,
            random_state=42,
            max_iter=1000,
        )
        model = train_classifier(training_data, config=config)

        assert hasattr(model, "predict")


# =============================================================================
# AC-7.3: Test model evaluation
# =============================================================================


class TestEvaluateModel:
    """Tests for evaluate_model() - AC-7.3."""

    def test_returns_evaluation_metrics(self, temp_data_file: Path) -> None:
        """AC7.8: evaluate_model returns EvaluationMetrics."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        metrics = evaluate_model(model, training_data)

        assert isinstance(metrics, EvaluationMetrics)

    def test_metrics_has_accuracy(self, temp_data_file: Path) -> None:
        """AC7.8: Metrics includes accuracy."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        metrics = evaluate_model(model, training_data)

        assert hasattr(metrics, "accuracy")
        assert 0 <= metrics.accuracy <= 1

    def test_metrics_has_precision(self, temp_data_file: Path) -> None:
        """AC7.8: Metrics includes precision."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        metrics = evaluate_model(model, training_data)

        assert hasattr(metrics, "precision")
        assert 0 <= metrics.precision <= 1

    def test_metrics_has_recall(self, temp_data_file: Path) -> None:
        """AC7.8: Metrics includes recall."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        metrics = evaluate_model(model, training_data)

        assert hasattr(metrics, "recall")
        assert 0 <= metrics.recall <= 1

    def test_metrics_has_f1_score(self, temp_data_file: Path) -> None:
        """AC7.8: Metrics includes F1 score."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        metrics = evaluate_model(model, training_data)

        assert hasattr(metrics, "f1_score")
        assert 0 <= metrics.f1_score <= 1


# =============================================================================
# AC-7.4: Test model saved to path
# =============================================================================


class TestSaveModel:
    """Tests for save_model() - AC-7.4."""

    def test_saves_model_to_joblib(
        self, temp_data_file: Path, temp_model_path: Path
    ) -> None:
        """AC7.4: Model saved to .joblib file."""
        import joblib

        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        save_model(model, temp_model_path)

        assert temp_model_path.exists()

        # Verify it can be loaded
        loaded_model = joblib.load(temp_model_path)
        assert hasattr(loaded_model, "predict")

    def test_creates_parent_directories(self, temp_data_file: Path) -> None:
        """AC7.4: Creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "models" / "nested" / "classifier.joblib"

            training_data = prepare_training_data(temp_data_file)
            model = train_classifier(training_data)

            save_model(model, nested_path)

            assert nested_path.exists()

    def test_overwrites_existing_file(
        self, temp_data_file: Path, temp_model_path: Path
    ) -> None:
        """AC7.4: Overwrites existing model file."""
        training_data = prepare_training_data(temp_data_file)
        model = train_classifier(training_data)

        # Save twice
        save_model(model, temp_model_path)
        original_size = temp_model_path.stat().st_size

        save_model(model, temp_model_path)
        new_size = temp_model_path.stat().st_size

        # Size should be same (same model)
        assert abs(original_size - new_size) < 100  # Small variance allowed


# =============================================================================
# Configuration Tests
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_training_config_defaults(self) -> None:
        """TrainingConfig has sensible defaults."""
        config = TrainingConfig()

        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.max_iter == 1000

    def test_training_config_custom_values(self) -> None:
        """TrainingConfig accepts custom values."""
        config = TrainingConfig(
            test_size=0.3,
            random_state=123,
            max_iter=2000,
        )

        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.max_iter == 2000


class TestConstants:
    """Tests for module constants."""

    def test_default_model_name(self) -> None:
        """DEFAULT_MODEL_NAME is all-MiniLM-L6-v2."""
        assert DEFAULT_MODEL_NAME == "all-MiniLM-L6-v2"

    def test_default_confidence_threshold(self) -> None:
        """DEFAULT_CONFIDENCE_THRESHOLD is 0.7."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.7


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullTrainingPipeline:
    """Integration tests for the full training pipeline."""

    def test_end_to_end_training(
        self, temp_data_file: Path, temp_model_path: Path
    ) -> None:
        """Full pipeline: load -> train -> evaluate -> save."""
        import joblib

        # Load data
        training_data = prepare_training_data(temp_data_file)
        assert len(training_data.terms) > 0

        # Train model
        model = train_classifier(training_data)
        assert hasattr(model, "predict")

        # Evaluate
        metrics = evaluate_model(model, training_data)
        assert metrics.accuracy > 0

        # Save
        save_model(model, temp_model_path)
        assert temp_model_path.exists()

        # Verify loaded model works
        loaded_model = joblib.load(temp_model_path)
        dummy_embedding = np.random.rand(1, 384)
        prediction = loaded_model.predict(dummy_embedding)
        assert prediction[0] in [0, 1]
