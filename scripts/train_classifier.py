#!/usr/bin/env python3
"""
Train Classifier for Hybrid Tiered Classifier (Tier 2).

This script trains a LogisticRegression classifier on SBERT embeddings
for concept vs keyword classification.

Usage:
    python scripts/train_classifier.py

Output:
    models/concept_classifier.joblib

Data Source:
    data/validated_term_filter.json

AC-7.1: Training data loads from validated_term_filter.json
AC-7.2: Model trained on SBERT embeddings (LogisticRegression)
AC-7.3: Evaluation reports accuracy, precision, recall, F1
AC-7.4: Model exported to .joblib file
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import joblib  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent

DEFAULT_INPUT_FILE: Final[Path] = (
    PROJECT_ROOT / "data" / "validated_term_filter.json"
)
DEFAULT_OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "models" / "concept_classifier.joblib"

# Model constants
DEFAULT_MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.7

# Label mapping: 0 = concept, 1 = keyword
LABEL_CONCEPT: Final[int] = 0
LABEL_KEYWORD: Final[int] = 1


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class TrainingData:
    """
    Container for training data.

    Attributes:
        terms: List of term strings.
        labels: List of labels (0=concept, 1=keyword).
        embeddings: Optional pre-computed embeddings.
    """

    terms: list[str]
    labels: list[int]
    embeddings: NDArray[np.float32] | None = field(default=None)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Configuration for model training.

    Attributes:
        test_size: Fraction of data to use for testing (default 0.2).
        random_state: Random seed for reproducibility (default 42).
        max_iter: Maximum iterations for LogisticRegression (default 1000).
        model_name: SBERT model name (default all-MiniLM-L6-v2).
    """

    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 1000
    model_name: str = DEFAULT_MODEL_NAME


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """
    Evaluation metrics for the trained model.

    Attributes:
        accuracy: Overall accuracy score.
        precision: Precision score (weighted average).
        recall: Recall score (weighted average).
        f1_score: F1 score (weighted average).
        classification_report: Full classification report string.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    classification_report: str


# =============================================================================
# Data Loading
# =============================================================================


def prepare_training_data(input_path: Path) -> TrainingData:
    """
    Load and prepare training data from validated_term_filter.json.

    Reads concepts and keywords from the JSON file and assigns labels:
    - Concepts get label 0
    - Keywords get label 1

    Args:
        input_path: Path to the validated_term_filter.json file.

    Returns:
        TrainingData with terms and labels.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Training data file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    concepts: list[str] = data.get("concepts", [])
    keywords: list[str] = data.get("keywords", [])

    # Combine terms and assign labels
    terms: list[str] = []
    labels: list[int] = []

    for concept in concepts:
        terms.append(concept)
        labels.append(LABEL_CONCEPT)

    for keyword in keywords:
        terms.append(keyword)
        labels.append(LABEL_KEYWORD)

    return TrainingData(terms=terms, labels=labels)


# =============================================================================
# Training
# =============================================================================


def train_classifier(
    training_data: TrainingData,
    config: TrainingConfig | None = None,
) -> LogisticRegression:
    """
    Train a LogisticRegression classifier on SBERT embeddings.

    Args:
        training_data: TrainingData with terms and labels.
        config: Optional TrainingConfig for hyperparameters.

    Returns:
        Trained LogisticRegression model.
    """
    if config is None:
        config = TrainingConfig()

    print(f"\nLoading SBERT model: {config.model_name}")
    embedder = SentenceTransformer(config.model_name)

    print(f"Generating embeddings for {len(training_data.terms):,} terms...")
    embeddings = embedder.encode(
        training_data.terms,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"\nSplitting data (test_size={config.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        training_data.labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=training_data.labels,
    )

    print(f"Training LogisticRegression (max_iter={config.max_iter})...")
    model = LogisticRegression(
        max_iter=config.max_iter,
        random_state=config.random_state,
        solver="lbfgs",
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Quick validation
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return model


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_model(
    model: LogisticRegression,
    training_data: TrainingData,
    config: TrainingConfig | None = None,
) -> EvaluationMetrics:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained LogisticRegression model.
        training_data: TrainingData with terms and labels.
        config: Optional TrainingConfig for evaluation parameters.

    Returns:
        EvaluationMetrics with accuracy, precision, recall, F1.
    """
    if config is None:
        config = TrainingConfig()

    # Generate embeddings if not present
    embedder = SentenceTransformer(config.model_name)
    embeddings = embedder.encode(
        training_data.terms,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        training_data.labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=training_data.labels,
    )

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=["concept", "keyword"],
        zero_division=0,
    )

    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        classification_report=report,
    )


# =============================================================================
# Model Persistence
# =============================================================================


def save_model(model: LogisticRegression, output_path: Path) -> None:
    """
    Save the trained model to a .joblib file.

    Args:
        model: Trained LogisticRegression model.
        output_path: Path to save the model file.
    """
    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, output_path)

    print(f"✓ Model saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for training the classifier."""
    print("=" * 60)
    print("Training Concept Classifier for Hybrid Tiered Classifier")
    print("=" * 60)

    # Load training data
    print(f"\nLoading training data from: {DEFAULT_INPUT_FILE}")
    if not DEFAULT_INPUT_FILE.exists():
        print(f"ERROR: Training data file not found: {DEFAULT_INPUT_FILE}")
        sys.exit(1)

    training_data = prepare_training_data(DEFAULT_INPUT_FILE)
    concept_count = sum(1 for l in training_data.labels if l == LABEL_CONCEPT)
    keyword_count = sum(1 for l in training_data.labels if l == LABEL_KEYWORD)
    print(f"  Total terms: {len(training_data.terms):,}")
    print(f"  Concepts: {concept_count:,}")
    print(f"  Keywords: {keyword_count:,}")

    # Configure training
    config = TrainingConfig(
        test_size=0.2,
        random_state=42,
        max_iter=1000,
    )

    # Train model
    print("\n" + "-" * 60)
    model = train_classifier(training_data, config)

    # Evaluate model
    print("\n" + "-" * 60)
    print("Evaluating model...")
    metrics = evaluate_model(model, training_data, config)

    print("\nEvaluation Results:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1_score:.4f}")

    print("\nClassification Report:")
    print(metrics.classification_report)

    # Check accuracy target
    target_accuracy = 0.98
    if metrics.accuracy >= target_accuracy:
        print(f"✓ Accuracy target met: {metrics.accuracy:.4f} >= {target_accuracy}")
    else:
        print(f"⚠ Accuracy below target: {metrics.accuracy:.4f} < {target_accuracy}")

    # Save model
    print("\n" + "-" * 60)
    save_model(model, DEFAULT_OUTPUT_PATH)

    # Save metadata
    metadata_path = DEFAULT_OUTPUT_PATH.with_suffix(".json")
    metadata = {
        "trained": datetime.now(tz=None).isoformat(),
        "model_name": config.model_name,
        "input_file": str(DEFAULT_INPUT_FILE),
        "total_terms": len(training_data.terms),
        "concept_count": concept_count,
        "keyword_count": keyword_count,
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
        },
        "config": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "max_iter": config.max_iter,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
