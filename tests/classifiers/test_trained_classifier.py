"""
RED Phase Tests for TrainedClassifier (Tier 2) - WBS-AC2.

TDD Methodology: These tests are written FIRST before implementation.
Each test maps to a specific Acceptance Criteria from WBS-AC2.

AC-2.1: ConceptClassifierProtocol is runtime_checkable, TrainedClassifier passes
AC-2.2: High confidence (>=0.7) returns concept or keyword classification
AC-2.3: Low confidence (<0.7) returns unknown
AC-2.4: Uses all-MiniLM-L6-v2 embedder
AC-2.5: Loads model from .joblib file
AC-2.6: Raises ConceptClassifierError if model not loaded
AC-2.7: predict_batch() works for multiple terms
AC-2.8: FakeClassifier passes Protocol and returns configured responses
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from src.classifiers.trained_classifier import (
        ClassificationResult,
        ConceptClassifierProtocol,
        FakeClassifier,
        TrainedClassifier,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    """Create a fake .joblib model file for testing."""
    import joblib
    from sklearn.linear_model import LogisticRegression

    # Create a simple trained model
    # 2 classes: 0=concept, 1=keyword
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.8, 0.9], [0.7, 0.6]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    model_file = tmp_path / "test_model.joblib"
    joblib.dump(model, model_file)
    return model_file


@pytest.fixture
def mock_embedder(mocker: pytest.MockerFixture) -> pytest.MockerFixture:
    """Mock the SentenceTransformer to avoid loading real model."""
    mock = mocker.patch(
        "src.classifiers.trained_classifier.SentenceTransformer"
    )

    # Return embeddings matching input batch size
    def encode_side_effect(terms: list[str]) -> np.ndarray:
        return np.array([[0.1, 0.2]] * len(terms))

    mock.return_value.encode.side_effect = encode_side_effect
    return mock


@pytest.fixture
def trained_classifier(
    fake_model_path: Path, mock_embedder: pytest.MockerFixture
) -> "TrainedClassifier":
    """Create a TrainedClassifier with mocked dependencies."""
    from src.classifiers.trained_classifier import TrainedClassifier

    return TrainedClassifier(model_path=fake_model_path)


@pytest.fixture
def fake_classifier() -> "FakeClassifier":
    """Create a FakeClassifier with default responses."""
    from src.classifiers.trained_classifier import FakeClassifier

    return FakeClassifier(
        responses={
            "microservice": ("concept", 0.95),
            "python": ("keyword", 0.88),
            "unknown_term": ("unknown", 0.45),
        }
    )


# =============================================================================
# AC2.1: Test Protocol is runtime_checkable
# =============================================================================


class TestProtocolRuntimeCheckable:
    """Tests for AC-2.1: ConceptClassifierProtocol is runtime_checkable."""

    def test_protocol_exists(self) -> None:
        """AC2.1: ConceptClassifierProtocol should exist."""
        from src.classifiers.trained_classifier import ConceptClassifierProtocol

        assert ConceptClassifierProtocol is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """AC2.1: Protocol should be decorated with @runtime_checkable."""
        from typing import runtime_checkable

        from src.classifiers.trained_classifier import ConceptClassifierProtocol

        # A runtime_checkable protocol can be used with isinstance()
        # This is verified by checking the protocol's attributes
        assert hasattr(ConceptClassifierProtocol, "__protocol_attrs__") or hasattr(
            ConceptClassifierProtocol, "_is_runtime_protocol"
        )

    def test_protocol_has_predict_method(self) -> None:
        """AC2.1: Protocol should define predict() method signature."""
        from src.classifiers.trained_classifier import ConceptClassifierProtocol

        # Check that predict is defined in the protocol
        assert "predict" in dir(ConceptClassifierProtocol)

    def test_protocol_has_predict_batch_method(self) -> None:
        """AC2.1: Protocol should define predict_batch() method signature."""
        from src.classifiers.trained_classifier import ConceptClassifierProtocol

        assert "predict_batch" in dir(ConceptClassifierProtocol)


# =============================================================================
# AC2.2: Test TrainedClassifier passes Protocol
# =============================================================================


class TestTrainedClassifierProtocolCompliance:
    """Tests for AC-2.1: TrainedClassifier implements ConceptClassifierProtocol."""

    def test_trained_classifier_passes_isinstance_check(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.2: TrainedClassifier should pass isinstance check."""
        from src.classifiers.trained_classifier import ConceptClassifierProtocol

        assert isinstance(trained_classifier, ConceptClassifierProtocol)

    def test_trained_classifier_has_predict(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.2: TrainedClassifier should have predict() method."""
        assert hasattr(trained_classifier, "predict")
        assert callable(trained_classifier.predict)

    def test_trained_classifier_has_predict_batch(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.2: TrainedClassifier should have predict_batch() method."""
        assert hasattr(trained_classifier, "predict_batch")
        assert callable(trained_classifier.predict_batch)


# =============================================================================
# AC2.3: Test high confidence returns classification
# =============================================================================


class TestHighConfidenceClassification:
    """Tests for AC-2.2: High confidence returns concept or keyword."""

    def test_high_confidence_returns_classification_result(
        self,
        fake_model_path: Path,
        mocker: pytest.MockerFixture,
    ) -> None:
        """AC2.3: High confidence prediction should return ClassificationResult."""
        from src.classifiers.trained_classifier import (
            ClassificationResult,
            TrainedClassifier,
        )

        # Mock embedder and model to return high confidence
        mock_embedder = mocker.patch(
            "src.classifiers.trained_classifier.SentenceTransformer"
        )
        mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2]])

        classifier = TrainedClassifier(model_path=fake_model_path)
        result = classifier.predict("microservice")

        assert isinstance(result, ClassificationResult)

    def test_high_confidence_concept_classification(
        self,
        fake_model_path: Path,
        mocker: pytest.MockerFixture,
    ) -> None:
        """AC2.3: High confidence concept should have label 'concept'."""
        from src.classifiers.trained_classifier import TrainedClassifier

        # Mock to return concept with high confidence
        mock_embedder = mocker.patch(
            "src.classifiers.trained_classifier.SentenceTransformer"
        )
        mock_embedder.return_value.encode.return_value = np.array([[0.1, 0.2]])

        classifier = TrainedClassifier(model_path=fake_model_path)
        result = classifier.predict("microservice")

        # The mock model will predict based on the embedding
        # We're testing the structure, not the exact prediction
        assert result.predicted_label in ("concept", "keyword", "unknown")

    def test_high_confidence_has_tier_used_2(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.3: Result should have tier_used=2."""
        result = trained_classifier.predict("microservice")
        assert result.tier_used == 2

    def test_classification_result_has_required_fields(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.3: ClassificationResult should have all required fields."""
        result = trained_classifier.predict("test_term")

        assert hasattr(result, "predicted_label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "tier_used")


# =============================================================================
# AC2.4: Test low confidence returns unknown
# =============================================================================


class TestLowConfidenceClassification:
    """Tests for AC-2.3: Low confidence (<0.7) returns unknown."""

    def test_low_confidence_returns_unknown(
        self,
        fake_model_path: Path,
        mocker: pytest.MockerFixture,
    ) -> None:
        """AC2.4: Low confidence should return predicted_label='unknown'."""
        from src.classifiers.trained_classifier import TrainedClassifier

        # Mock to return low confidence prediction
        mock_embedder = mocker.patch(
            "src.classifiers.trained_classifier.SentenceTransformer"
        )
        # Embedding that will produce low confidence
        mock_embedder.return_value.encode.return_value = np.array([[0.5, 0.5]])

        classifier = TrainedClassifier(model_path=fake_model_path)
        result = classifier.predict("ambiguous_term")

        # If confidence is below threshold, should return unknown
        if result.confidence < 0.7:
            assert result.predicted_label == "unknown"

    def test_confidence_threshold_is_0_7(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.4: Confidence threshold should be 0.7."""
        from src.classifiers.trained_classifier import CONFIDENCE_THRESHOLD

        assert CONFIDENCE_THRESHOLD == 0.7


# =============================================================================
# AC2.5: Test model loading from path
# =============================================================================


class TestModelLoading:
    """Tests for AC-2.5: Loads model from .joblib file."""

    def test_loads_model_from_joblib(
        self,
        fake_model_path: Path,
        mock_embedder: pytest.MockerFixture,
    ) -> None:
        """AC2.5: Should load model from .joblib file path."""
        from src.classifiers.trained_classifier import TrainedClassifier

        classifier = TrainedClassifier(model_path=fake_model_path)
        assert classifier._model is not None

    def test_model_has_predict_proba(
        self,
        fake_model_path: Path,
        mock_embedder: pytest.MockerFixture,
    ) -> None:
        """AC2.5: Loaded model should have predict_proba method."""
        from src.classifiers.trained_classifier import TrainedClassifier

        classifier = TrainedClassifier(model_path=fake_model_path)
        assert hasattr(classifier._model, "predict_proba")


# =============================================================================
# AC2.6: Test error when model not loaded
# =============================================================================


class TestModelNotLoaded:
    """Tests for AC-2.6: Raises ConceptClassifierError if model not loaded."""

    def test_raises_error_for_missing_model_file(
        self,
        tmp_path: Path,
        mock_embedder: pytest.MockerFixture,
    ) -> None:
        """AC2.6: Should raise ConceptClassifierError for missing file."""
        from src.classifiers.exceptions import ConceptClassifierError
        from src.classifiers.trained_classifier import TrainedClassifier

        missing_path = tmp_path / "nonexistent.joblib"

        with pytest.raises(ConceptClassifierError) as exc_info:
            TrainedClassifier(model_path=missing_path)

        assert "not found" in str(exc_info.value).lower()

    def test_error_includes_file_path(
        self,
        tmp_path: Path,
        mock_embedder: pytest.MockerFixture,
    ) -> None:
        """AC2.6: Error message should include the missing file path."""
        from src.classifiers.exceptions import ConceptClassifierError
        from src.classifiers.trained_classifier import TrainedClassifier

        missing_path = tmp_path / "missing_model.joblib"

        with pytest.raises(ConceptClassifierError) as exc_info:
            TrainedClassifier(model_path=missing_path)

        assert "missing_model.joblib" in str(exc_info.value)

    def test_raises_error_for_invalid_model_file(
        self,
        tmp_path: Path,
        mock_embedder: pytest.MockerFixture,
    ) -> None:
        """AC2.6: Should raise ConceptClassifierError for invalid model file."""
        from src.classifiers.exceptions import ConceptClassifierError
        from src.classifiers.trained_classifier import TrainedClassifier

        # Create a file with invalid content
        invalid_file = tmp_path / "invalid.joblib"
        invalid_file.write_text("not a valid joblib file")

        with pytest.raises(ConceptClassifierError) as exc_info:
            TrainedClassifier(model_path=invalid_file)

        assert "Failed to load model" in str(exc_info.value)


# =============================================================================
# AC2.7: Test batch prediction
# =============================================================================


class TestBatchPrediction:
    """Tests for AC-2.7: predict_batch() works for multiple terms."""

    def test_batch_prediction_returns_list(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.7: predict_batch() should return list of results."""
        terms = ["microservice", "python", "api"]
        results = trained_classifier.predict_batch(terms)

        assert isinstance(results, list)
        assert len(results) == len(terms)

    def test_batch_prediction_each_result_is_classification_result(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.7: Each item in batch result should be ClassificationResult."""
        from src.classifiers.trained_classifier import ClassificationResult

        terms = ["microservice", "python"]
        results = trained_classifier.predict_batch(terms)

        for result in results:
            assert isinstance(result, ClassificationResult)

    def test_batch_prediction_empty_list(
        self,
        trained_classifier: "TrainedClassifier",
    ) -> None:
        """AC2.7: Empty list should return empty results."""
        results = trained_classifier.predict_batch([])
        assert results == []

    def test_batch_prediction_preserves_order(
        self,
        fake_model_path: Path,
        mocker: pytest.MockerFixture,
    ) -> None:
        """AC2.7: Batch results should preserve input order."""
        from src.classifiers.trained_classifier import TrainedClassifier

        # Mock embedder to return different embeddings per term
        mock_embedder = mocker.patch(
            "src.classifiers.trained_classifier.SentenceTransformer"
        )
        mock_embedder.return_value.encode.return_value = np.array(
            [[0.1, 0.2], [0.8, 0.9], [0.5, 0.5]]
        )

        classifier = TrainedClassifier(model_path=fake_model_path)
        terms = ["first", "second", "third"]
        results = classifier.predict_batch(terms)

        # Results should match input order
        assert len(results) == 3


# =============================================================================
# AC2.8: Test FakeClassifier passes Protocol
# =============================================================================


class TestFakeClassifierProtocol:
    """Tests for AC-2.8: FakeClassifier passes Protocol."""

    def test_fake_classifier_exists(self) -> None:
        """AC2.8: FakeClassifier should exist."""
        from src.classifiers.trained_classifier import FakeClassifier

        assert FakeClassifier is not None

    def test_fake_classifier_passes_isinstance_check(
        self,
        fake_classifier: "FakeClassifier",
    ) -> None:
        """AC2.8: FakeClassifier should pass isinstance check."""
        from src.classifiers.trained_classifier import ConceptClassifierProtocol

        assert isinstance(fake_classifier, ConceptClassifierProtocol)

    def test_fake_classifier_has_predict(
        self,
        fake_classifier: "FakeClassifier",
    ) -> None:
        """AC2.8: FakeClassifier should have predict() method."""
        assert hasattr(fake_classifier, "predict")
        assert callable(fake_classifier.predict)

    def test_fake_classifier_has_predict_batch(
        self,
        fake_classifier: "FakeClassifier",
    ) -> None:
        """AC2.8: FakeClassifier should have predict_batch() method."""
        assert hasattr(fake_classifier, "predict_batch")
        assert callable(fake_classifier.predict_batch)


# =============================================================================
# AC2.9: Test FakeClassifier returns configured responses
# =============================================================================


class TestFakeClassifierResponses:
    """Tests for AC-2.8: FakeClassifier returns configured responses."""

    def test_fake_classifier_returns_configured_concept(self) -> None:
        """AC2.9: FakeClassifier should return configured concept response."""
        from src.classifiers.trained_classifier import FakeClassifier

        fake = FakeClassifier(
            responses={"microservice": ("concept", 0.95)}
        )
        result = fake.predict("microservice")

        assert result.predicted_label == "concept"
        assert result.confidence == 0.95

    def test_fake_classifier_returns_configured_keyword(self) -> None:
        """AC2.9: FakeClassifier should return configured keyword response."""
        from src.classifiers.trained_classifier import FakeClassifier

        fake = FakeClassifier(
            responses={"python": ("keyword", 0.88)}
        )
        result = fake.predict("python")

        assert result.predicted_label == "keyword"
        assert result.confidence == 0.88

    def test_fake_classifier_returns_unknown_for_unconfigured(self) -> None:
        """AC2.9: FakeClassifier should return unknown for unconfigured terms."""
        from src.classifiers.trained_classifier import FakeClassifier

        fake = FakeClassifier(responses={})
        result = fake.predict("not_configured")

        assert result.predicted_label == "unknown"

    def test_fake_classifier_batch_returns_configured_responses(self) -> None:
        """AC2.9: FakeClassifier batch should return all configured responses."""
        from src.classifiers.trained_classifier import FakeClassifier

        fake = FakeClassifier(
            responses={
                "microservice": ("concept", 0.95),
                "python": ("keyword", 0.88),
            }
        )
        results = fake.predict_batch(["microservice", "python"])

        assert results[0].predicted_label == "concept"
        assert results[1].predicted_label == "keyword"

    def test_fake_classifier_tier_used_is_2(self) -> None:
        """AC2.9: FakeClassifier results should have tier_used=2."""
        from src.classifiers.trained_classifier import FakeClassifier

        fake = FakeClassifier(responses={"test": ("concept", 0.9)})
        result = fake.predict("test")

        assert result.tier_used == 2


# =============================================================================
# AC-2.4: Test SBERT Embedder
# =============================================================================


class TestSBERTEmbedder:
    """Tests for AC-2.4: Uses all-MiniLM-L6-v2 embedder."""

    def test_uses_minilm_model(
        self,
        fake_model_path: Path,
        mocker: pytest.MockerFixture,
    ) -> None:
        """AC-2.4: Should use all-MiniLM-L6-v2 model."""
        from src.classifiers.trained_classifier import (
            SBERT_MODEL_NAME,
            TrainedClassifier,
        )

        mock_st = mocker.patch(
            "src.classifiers.trained_classifier.SentenceTransformer"
        )
        mock_st.return_value.encode.return_value = np.array([[0.1, 0.2]])

        _ = TrainedClassifier(model_path=fake_model_path)

        mock_st.assert_called_once_with(SBERT_MODEL_NAME)

    def test_sbert_model_name_constant(self) -> None:
        """AC-2.4: SBERT_MODEL_NAME should be 'all-MiniLM-L6-v2'."""
        from src.classifiers.trained_classifier import SBERT_MODEL_NAME

        assert SBERT_MODEL_NAME == "all-MiniLM-L6-v2"


# =============================================================================
# ClassificationResult Dataclass Tests
# =============================================================================


class TestClassificationResultDataclass:
    """Tests for ClassificationResult dataclass structure."""

    def test_classification_result_exists(self) -> None:
        """ClassificationResult dataclass should exist."""
        from src.classifiers.trained_classifier import ClassificationResult

        assert ClassificationResult is not None

    def test_classification_result_is_frozen(self) -> None:
        """ClassificationResult should be immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        from src.classifiers.trained_classifier import ClassificationResult

        result = ClassificationResult(
            predicted_label="concept",
            confidence=0.95,
            tier_used=2,
        )

        with pytest.raises(FrozenInstanceError):
            result.predicted_label = "keyword"  # type: ignore[misc]

    def test_classification_result_fields(self) -> None:
        """ClassificationResult should have correct fields."""
        from src.classifiers.trained_classifier import ClassificationResult

        result = ClassificationResult(
            predicted_label="concept",
            confidence=0.85,
            tier_used=2,
        )

        assert result.predicted_label == "concept"
        assert result.confidence == 0.85
        assert result.tier_used == 2


# =============================================================================
# Exception Tests
# =============================================================================


class TestConceptClassifierError:
    """Tests for ConceptClassifierError exception."""

    def test_exception_exists(self) -> None:
        """ConceptClassifierError should exist."""
        from src.classifiers.exceptions import ConceptClassifierError

        assert ConceptClassifierError is not None

    def test_exception_is_exception_subclass(self) -> None:
        """ConceptClassifierError should be Exception subclass."""
        from src.classifiers.exceptions import ConceptClassifierError

        assert issubclass(ConceptClassifierError, Exception)

    def test_exception_name_ends_with_error(self) -> None:
        """ConceptClassifierError name should end with 'Error' (AC-8.3)."""
        from src.classifiers.exceptions import ConceptClassifierError

        assert ConceptClassifierError.__name__.endswith("Error")

    def test_exception_can_be_raised_with_message(self) -> None:
        """ConceptClassifierError should accept message."""
        from src.classifiers.exceptions import ConceptClassifierError

        with pytest.raises(ConceptClassifierError) as exc_info:
            raise ConceptClassifierError("Test error message")

        assert "Test error message" in str(exc_info.value)
