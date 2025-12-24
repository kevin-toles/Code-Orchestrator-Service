"""
Trained Classifier (Tier 2) - Hybrid Tiered Classifier.

This module provides ML-based classification for terms not found in Tier 1.
It uses SBERT embeddings with a trained LogisticRegression model.

Pattern: Embeddings (Machine Learning Design Patterns, Ch. 2)

AC-2.1: ConceptClassifierProtocol is runtime_checkable
AC-2.2: High confidence (>=0.7) returns concept or keyword
AC-2.3: Low confidence (<0.7) returns unknown
AC-2.4: Uses all-MiniLM-L6-v2 embedder
AC-2.5: Loads model from .joblib file
AC-2.6: Raises ConceptClassifierError if model not loaded
AC-2.7: predict_batch() works for multiple terms
AC-2.8: FakeClassifier passes Protocol and returns configured responses
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import joblib  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.classifiers.exceptions import ConceptClassifierError

# =============================================================================
# Constants
# =============================================================================

CONFIDENCE_THRESHOLD: Final[float] = 0.7
SBERT_MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
TIER_TRAINED_CLASSIFIER: Final[int] = 2
LABEL_CONCEPT: Final[str] = "concept"
LABEL_KEYWORD: Final[str] = "keyword"
LABEL_UNKNOWN: Final[str] = "unknown"
DEFAULT_UNKNOWN_CONFIDENCE: Final[float] = 0.5

# Label mapping for model predictions
# Model outputs: 0 = concept, 1 = keyword
LABEL_MAP: Final[dict[int, str]] = {
    0: LABEL_CONCEPT,
    1: LABEL_KEYWORD,
}


# =============================================================================
# Protocol Definitions
# =============================================================================


class SklearnClassifierProtocol(Protocol):
    """Protocol for sklearn-compatible classifiers with predict_proba."""

    def predict_proba(
        self, X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return probability predictions for X."""
        ...


@runtime_checkable
class ConceptClassifierProtocol(Protocol):
    """
    Protocol defining the interface for concept classifiers.

    This protocol enables duck typing for any classifier implementation,
    allowing FakeClassifier to be used interchangeably with TrainedClassifier
    in tests without requiring inheritance.

    Methods:
        predict: Classify a single term.
        predict_batch: Classify multiple terms.
    """

    def predict(self, term: str) -> "ClassificationResult":
        """
        Classify a single term.

        Args:
            term: The term to classify.

        Returns:
            ClassificationResult with predicted_label, confidence, tier_used.
        """
        ...

    def predict_batch(self, terms: list[str]) -> list["ClassificationResult"]:
        """
        Classify multiple terms.

        Args:
            terms: List of terms to classify.

        Returns:
            List of ClassificationResult objects in same order as input.
        """
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """
    Result from Tier 2 trained classifier.

    Attributes:
        predicted_label: Either 'concept', 'keyword', or 'unknown'.
        confidence: Model confidence score (0.0 to 1.0).
        tier_used: Always 2 for trained classifier results.
    """

    predicted_label: str
    confidence: float
    tier_used: int


# =============================================================================
# Trained Classifier Implementation
# =============================================================================


class TrainedClassifier:
    """
    ML-based classifier using SBERT embeddings (Tier 2).

    This classifier uses a pre-trained sentence transformer to embed terms,
    then applies a LogisticRegression model to classify them as either
    'concept' or 'keyword'. If confidence is below the threshold (0.7),
    it returns 'unknown' to allow fallback to Tier 3.

    Example:
        >>> classifier = TrainedClassifier(model_path=Path("models/classifier.joblib"))
        >>> result = classifier.predict("microservice")
        >>> result.predicted_label
        'concept'
        >>> result.confidence
        0.92

    Attributes:
        _model: The loaded scikit-learn classification model.
        _embedder: SentenceTransformer for generating embeddings.
    """

    __slots__ = ("_model", "_embedder")

    _model: SklearnClassifierProtocol
    _embedder: SentenceTransformer

    def __init__(self, model_path: Path) -> None:
        """
        Initialize the trained classifier.

        Args:
            model_path: Path to the .joblib model file.

        Raises:
            ConceptClassifierError: If model file not found or invalid.
        """
        self._model = self._load_model(model_path)
        self._embedder = SentenceTransformer(SBERT_MODEL_NAME)

    def _load_model(self, model_path: Path) -> SklearnClassifierProtocol:
        """
        Load the classification model from a joblib file.

        Args:
            model_path: Path to the .joblib file.

        Returns:
            Loaded sklearn model with predict_proba method.

        Raises:
            ConceptClassifierError: If file not found or load fails.
        """
        if not model_path.exists():
            raise ConceptClassifierError(
                f"Model file not found: {model_path}"
            )

        try:
            model: SklearnClassifierProtocol = joblib.load(model_path)
            return model
        except Exception as e:
            raise ConceptClassifierError(
                f"Failed to load model from {model_path}: {e}"
            ) from e

    def _embed(self, terms: list[str]) -> NDArray[np.floating]:
        """
        Generate SBERT embeddings for terms.

        Args:
            terms: List of terms to embed.

        Returns:
            NumPy array of embeddings, shape (n_terms, embedding_dim).
        """
        embeddings: NDArray[np.floating] = self._embedder.encode(terms)
        return embeddings

    def predict(self, term: str) -> ClassificationResult:
        """
        Classify a single term.

        Args:
            term: The term to classify.

        Returns:
            ClassificationResult with predicted_label, confidence, tier_used.
            If confidence < 0.7, predicted_label will be 'unknown'.
        """
        # Get embedding for the term
        embedding = self._embed([term])

        # Get prediction probabilities
        probas: NDArray[np.floating] = self._model.predict_proba(embedding)
        max_proba = float(np.max(probas[0]))
        predicted_class = int(np.argmax(probas[0]))

        # Apply confidence threshold
        if max_proba < CONFIDENCE_THRESHOLD:
            return ClassificationResult(
                predicted_label=LABEL_UNKNOWN,
                confidence=max_proba,
                tier_used=TIER_TRAINED_CLASSIFIER,
            )

        # Map class index to label
        predicted_label = LABEL_MAP.get(predicted_class, LABEL_UNKNOWN)

        return ClassificationResult(
            predicted_label=predicted_label,
            confidence=max_proba,
            tier_used=TIER_TRAINED_CLASSIFIER,
        )

    def predict_batch(self, terms: list[str]) -> list[ClassificationResult]:
        """
        Classify multiple terms efficiently.

        Args:
            terms: List of terms to classify.

        Returns:
            List of ClassificationResult objects in same order as input.
        """
        if not terms:
            return []

        # Batch embed all terms
        embeddings = self._embed(terms)

        # Get all predictions at once
        probas: NDArray[np.floating] = self._model.predict_proba(embeddings)

        results: list[ClassificationResult] = []
        for proba in probas:
            max_proba = float(np.max(proba))
            predicted_class = int(np.argmax(proba))

            if max_proba < CONFIDENCE_THRESHOLD:
                result = ClassificationResult(
                    predicted_label=LABEL_UNKNOWN,
                    confidence=max_proba,
                    tier_used=TIER_TRAINED_CLASSIFIER,
                )
            else:
                predicted_label = LABEL_MAP.get(predicted_class, LABEL_UNKNOWN)
                result = ClassificationResult(
                    predicted_label=predicted_label,
                    confidence=max_proba,
                    tier_used=TIER_TRAINED_CLASSIFIER,
                )
            results.append(result)

        return results


# =============================================================================
# Fake Classifier for Testing
# =============================================================================


class FakeClassifier:
    """
    Test double for TrainedClassifier (AC-8.4 Anti-Pattern #12 compliance).

    This fake implements ConceptClassifierProtocol and returns pre-configured
    responses, allowing unit tests to run without loading real ML models.

    Example:
        >>> fake = FakeClassifier(responses={"microservice": ("concept", 0.95)})
        >>> result = fake.predict("microservice")
        >>> result.predicted_label
        'concept'

    Attributes:
        _responses: Dictionary mapping terms to (label, confidence) tuples.
    """

    __slots__ = ("_responses",)

    def __init__(self, responses: dict[str, tuple[str, float]]) -> None:
        """
        Initialize FakeClassifier with configured responses.

        Args:
            responses: Dict mapping term -> (label, confidence).
                       Terms not in dict return 'unknown'.
        """
        self._responses = responses

    def predict(self, term: str) -> ClassificationResult:
        """
        Return configured response for term.

        Args:
            term: The term to classify.

        Returns:
            ClassificationResult from configured responses,
            or unknown if term not configured.
        """
        if term in self._responses:
            label, confidence = self._responses[term]
            return ClassificationResult(
                predicted_label=label,
                confidence=confidence,
                tier_used=TIER_TRAINED_CLASSIFIER,
            )

        return ClassificationResult(
            predicted_label=LABEL_UNKNOWN,
            confidence=DEFAULT_UNKNOWN_CONFIDENCE,
            tier_used=TIER_TRAINED_CLASSIFIER,
        )

    def predict_batch(self, terms: list[str]) -> list[ClassificationResult]:
        """
        Return configured responses for multiple terms.

        Args:
            terms: List of terms to classify.

        Returns:
            List of ClassificationResult objects.
        """
        return [self.predict(term) for term in terms]
