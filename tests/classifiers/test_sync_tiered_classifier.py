"""Unit tests for SyncTieredClassifier.

Tests the synchronous 3-tier classifier (excludes Tier 4 LLM).

WBS Reference: HTC-1.0 - Tiers 1-3 Sync Processing
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.classifiers import (
    AliasLookup,
    AliasLookupResult,
    ClassificationResponse,
    FakeClassifier,
    FakeHeuristicFilter,
    HeuristicFilterResult,
    SyncTieredClassifier,
)
from src.classifiers.trained_classifier import ClassificationResult


@pytest.fixture
def empty_alias_file() -> Path:
    """Create a temporary empty alias lookup file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump({}, f)
        return Path(f.name)


@pytest.fixture
def alias_file_with_kubernetes() -> Path:
    """Create alias lookup file with kubernetes entry."""
    data = {
        "kubernetes": {
            "canonical_term": "kubernetes",
            "classification": "concept",
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        return Path(f.name)


class TestSyncTieredClassifierExists:
    """Test that SyncTieredClassifier is properly exported."""

    def test_sync_tiered_classifier_importable(self) -> None:
        """Verify SyncTieredClassifier can be imported."""
        from src.classifiers import SyncTieredClassifier
        assert SyncTieredClassifier is not None

    def test_sync_tiered_classifier_has_classify(self) -> None:
        """Verify SyncTieredClassifier has classify method."""
        assert hasattr(SyncTieredClassifier, "classify")

    def test_sync_tiered_classifier_has_classify_batch(self) -> None:
        """Verify SyncTieredClassifier has classify_batch method."""
        assert hasattr(SyncTieredClassifier, "classify_batch")


class TestSyncTieredClassifierTier1:
    """Test Tier 1 (Alias Lookup) behavior."""

    def test_tier_1_returns_immediately(
        self, alias_file_with_kubernetes: Path
    ) -> None:
        """Verify known term returns immediately from Tier 1."""
        alias_lookup = AliasLookup(lookup_path=alias_file_with_kubernetes)
        # FakeClassifier requires dict of {term: (label, confidence)}
        fake_classifier = FakeClassifier(responses={})
        fake_filter = FakeHeuristicFilter()
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        result = classifier.classify("kubernetes")
        assert result.tier_used == 1
        assert result.classification == "concept"
        assert result.confidence == 1.0

    def test_tier_1_case_insensitive(
        self, alias_file_with_kubernetes: Path
    ) -> None:
        """Verify Tier 1 lookup is case insensitive."""
        alias_lookup = AliasLookup(lookup_path=alias_file_with_kubernetes)
        fake_classifier = FakeClassifier(responses={})
        fake_filter = FakeHeuristicFilter()
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        result = classifier.classify("Kubernetes")
        assert result.tier_used == 1
        assert result.canonical_term == "kubernetes"


class TestSyncTieredClassifierTier2:
    """Test Tier 2 (Trained Classifier) behavior."""

    def test_tier_2_used_when_tier_1_misses(
        self, empty_alias_file: Path
    ) -> None:
        """Verify Tier 2 is used when Tier 1 misses."""
        alias_lookup = AliasLookup(lookup_path=empty_alias_file)
        
        # Configure Tier 2 to return confident classification
        # FakeClassifier takes {term: (label, confidence)}
        fake_classifier = FakeClassifier(
            responses={
                "novel_concept": ("concept", 0.85),
            }
        )
        fake_filter = FakeHeuristicFilter()
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        result = classifier.classify("novel_concept")
        assert result.tier_used == 2
        assert result.classification == "concept"
        assert result.confidence >= 0.7


class TestSyncTieredClassifierTier3:
    """Test Tier 3 (Heuristic Filter) behavior."""

    def test_tier_3_rejects_noise(self, empty_alias_file: Path) -> None:
        """Verify Tier 3 rejects noise terms."""
        alias_lookup = AliasLookup(lookup_path=empty_alias_file)
        
        # Configure Tier 2 to return low confidence (falls through)
        fake_classifier = FakeClassifier(
            responses={
                "oceanofpdf": ("unknown", 0.3),
            }
        )
        
        # Configure Tier 3 to detect noise
        fake_filter = FakeHeuristicFilter(
            responses={
                "oceanofpdf": HeuristicFilterResult(
                    rejection_reason="noise_watermarks",
                    matched_term="oceanofpdf",
                    category="watermarks",
                ),
            }
        )
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        result = classifier.classify("oceanofpdf")
        assert result.tier_used == 3
        assert result.classification == "rejected"
        assert result.rejection_reason == "noise_watermarks"


class TestSyncTieredClassifierUnknown:
    """Test behavior when no tier produces result."""

    def test_returns_unknown_when_all_miss(self, empty_alias_file: Path) -> None:
        """Verify unknown returned when all tiers miss."""
        alias_lookup = AliasLookup(lookup_path=empty_alias_file)
        fake_classifier = FakeClassifier(responses={})  # Returns unknown by default
        fake_filter = FakeHeuristicFilter()  # Returns None by default
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        result = classifier.classify("completely_unknown_term")
        assert result.classification == "unknown"
        assert result.confidence == 0.0
        assert result.tier_used == 3  # Last tier checked


class TestSyncTieredClassifierBatch:
    """Test batch classification."""

    @pytest.fixture
    def mixed_alias_file(self) -> Path:
        """Create alias file with kubernetes."""
        data = {
            "kubernetes": {
                "canonical_term": "kubernetes",
                "classification": "concept",
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            return Path(f.name)

    def test_batch_returns_list(self, mixed_alias_file: Path) -> None:
        """Verify batch returns list of results."""
        alias_lookup = AliasLookup(lookup_path=mixed_alias_file)
        fake_classifier = FakeClassifier(
            responses={
                "api": ("keyword", 0.75),
            }
        )
        fake_filter = FakeHeuristicFilter()
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        terms = ["kubernetes", "api", "unknown"]
        results = classifier.classify_batch(terms)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_preserves_order(self, mixed_alias_file: Path) -> None:
        """Verify batch preserves input order."""
        alias_lookup = AliasLookup(lookup_path=mixed_alias_file)
        fake_classifier = FakeClassifier(
            responses={
                "api": ("keyword", 0.75),
            }
        )
        fake_filter = FakeHeuristicFilter()
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        terms = ["kubernetes", "api", "unknown"]
        results = classifier.classify_batch(terms)
        
        assert results[0].term == "kubernetes"
        assert results[1].term == "api"
        assert results[2].term == "unknown"

    def test_batch_empty_list(self, mixed_alias_file: Path) -> None:
        """Verify batch handles empty list."""
        alias_lookup = AliasLookup(lookup_path=mixed_alias_file)
        fake_classifier = FakeClassifier(responses={})
        fake_filter = FakeHeuristicFilter()
        
        classifier = SyncTieredClassifier(
            alias_lookup=alias_lookup,
            trained_classifier=fake_classifier,
            heuristic_filter=fake_filter,
        )
        
        results = classifier.classify_batch([])
        assert results == []


class TestClassificationResponse:
    """Test ClassificationResponse data class."""

    def test_response_has_required_fields(self) -> None:
        """Verify response has all required fields."""
        response = ClassificationResponse(
            term="test",
            classification="concept",
            confidence=0.9,
            canonical_term="test",
            tier_used=1,
        )
        
        assert response.term == "test"
        assert response.classification == "concept"
        assert response.confidence == 0.9
        assert response.canonical_term == "test"
        assert response.tier_used == 1
        assert response.rejection_reason is None

    def test_response_with_rejection_reason(self) -> None:
        """Verify response can include rejection reason."""
        response = ClassificationResponse(
            term="noise",
            classification="rejected",
            confidence=1.0,
            canonical_term="noise",
            tier_used=3,
            rejection_reason="noise_watermarks",
        )
        
        assert response.rejection_reason == "noise_watermarks"
