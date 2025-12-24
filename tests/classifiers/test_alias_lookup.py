"""
RED Phase Tests for AliasLookup (Tier 1) - WBS-AC1.

TDD Methodology: These tests are written FIRST before implementation.
Each test maps to a specific Acceptance Criteria from WBS-AC1.

AC-1.1: Exact match returns AliasLookupResult with confidence=1.0, tier_used=1
AC-1.2: Case-insensitive lookup works
AC-1.3: Unknown term returns None
AC-1.4: Alias resolves to canonical term
AC-1.5: Loads from alias_lookup.json at startup
AC-1.6: O(1) lookup performance
"""

from __future__ import annotations

import json
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from src.classifiers.alias_lookup import AliasLookup, AliasLookupResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_alias_data() -> dict[str, dict[str, str]]:
    """Sample alias lookup data for testing."""
    return {
        "microservice": {
            "canonical_term": "microservice",
            "classification": "concept",
        },
        "microservices": {
            "canonical_term": "microservice",
            "classification": "concept",
        },
        "api gateway": {
            "canonical_term": "api_gateway",
            "classification": "concept",
        },
        "api-gateway": {
            "canonical_term": "api_gateway",
            "classification": "concept",
        },
        "langchain": {
            "canonical_term": "langchain",
            "classification": "concept",
        },
        "llm": {
            "canonical_term": "large_language_model",
            "classification": "concept",
        },
        "large language model": {
            "canonical_term": "large_language_model",
            "classification": "concept",
        },
        "python": {
            "canonical_term": "python",
            "classification": "keyword",
        },
    }


@pytest.fixture
def temp_alias_file(tmp_path: Path, sample_alias_data: dict) -> Path:
    """Create a temporary alias_lookup.json file for testing."""
    alias_file = tmp_path / "alias_lookup.json"
    alias_file.write_text(json.dumps(sample_alias_data, indent=2))
    return alias_file


@pytest.fixture
def alias_lookup(temp_alias_file: Path) -> "AliasLookup":
    """Create an AliasLookup instance with test data."""
    from src.classifiers.alias_lookup import AliasLookup

    return AliasLookup(lookup_path=temp_alias_file)


# =============================================================================
# AC1.1: Test AliasLookupResult dataclass exists
# =============================================================================


class TestAliasLookupResultDataclass:
    """Tests for AC-1.1: AliasLookupResult dataclass structure."""

    def test_alias_lookup_result_dataclass_exists(self) -> None:
        """AC1.1: AliasLookupResult dataclass should exist."""
        from src.classifiers.alias_lookup import AliasLookupResult

        assert AliasLookupResult is not None

    def test_alias_lookup_result_has_required_fields(self) -> None:
        """AC1.1: AliasLookupResult should have required fields."""
        from src.classifiers.alias_lookup import AliasLookupResult

        result = AliasLookupResult(
            canonical_term="microservice",
            classification="concept",
            confidence=1.0,
            tier_used=1,
        )

        assert result.canonical_term == "microservice"
        assert result.classification == "concept"
        assert result.confidence == 1.0
        assert result.tier_used == 1

    def test_alias_lookup_result_is_frozen(self) -> None:
        """AC1.1: AliasLookupResult should be immutable (frozen)."""
        from src.classifiers.alias_lookup import AliasLookupResult

        result = AliasLookupResult(
            canonical_term="microservice",
            classification="concept",
            confidence=1.0,
            tier_used=1,
        )

        with pytest.raises(FrozenInstanceError):
            result.canonical_term = "other"  # type: ignore[misc]


# =============================================================================
# AC1.2: Test exact match returns result
# =============================================================================


class TestExactMatch:
    """Tests for AC-1.1: Exact match functionality."""

    def test_exact_match_returns_result(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.2: Exact match should return AliasLookupResult."""
        result = alias_lookup.get("microservice")

        assert result is not None
        assert result.canonical_term == "microservice"
        assert result.classification == "concept"

    def test_exact_match_confidence_is_one(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.3: Exact match should have confidence=1.0."""
        result = alias_lookup.get("microservice")

        assert result is not None
        assert result.confidence == 1.0

    def test_exact_match_tier_used_is_one(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.3: Exact match should have tier_used=1."""
        result = alias_lookup.get("microservice")

        assert result is not None
        assert result.tier_used == 1


# =============================================================================
# AC1.4: Test case-insensitive lookup
# =============================================================================


class TestCaseInsensitiveLookup:
    """Tests for AC-1.2: Case-insensitive lookup."""

    def test_lowercase_lookup(self, alias_lookup: "AliasLookup") -> None:
        """AC1.4: Lowercase term should match."""
        result = alias_lookup.get("microservice")
        assert result is not None
        assert result.canonical_term == "microservice"

    def test_uppercase_lookup(self, alias_lookup: "AliasLookup") -> None:
        """AC1.4: Uppercase term should match."""
        result = alias_lookup.get("MICROSERVICE")
        assert result is not None
        assert result.canonical_term == "microservice"

    def test_mixed_case_lookup(self, alias_lookup: "AliasLookup") -> None:
        """AC1.4: Mixed case term should match."""
        result = alias_lookup.get("MicroService")
        assert result is not None
        assert result.canonical_term == "microservice"

    def test_multi_word_case_insensitive(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.4: Multi-word terms should be case-insensitive."""
        result = alias_lookup.get("API Gateway")
        assert result is not None
        assert result.canonical_term == "api_gateway"

    def test_mixed_case_multi_word(self, alias_lookup: "AliasLookup") -> None:
        """AC1.4: Mixed case multi-word should match."""
        result = alias_lookup.get("Api GateWay")
        assert result is not None
        assert result.canonical_term == "api_gateway"


# =============================================================================
# AC1.5: Test unknown term returns None
# =============================================================================


class TestUnknownTerm:
    """Tests for AC-1.3: Unknown term handling."""

    def test_unknown_term_returns_none(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.5: Unknown term should return None."""
        result = alias_lookup.get("unknown_xyz_term")
        assert result is None

    def test_empty_string_returns_none(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.5: Empty string should return None."""
        result = alias_lookup.get("")
        assert result is None

    def test_whitespace_only_returns_none(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.5: Whitespace-only string should return None."""
        result = alias_lookup.get("   ")
        assert result is None


# =============================================================================
# AC1.6: Test alias resolves to canonical
# =============================================================================


class TestAliasResolution:
    """Tests for AC-1.4: Alias resolution to canonical term."""

    def test_plural_alias_resolves_to_canonical(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.6: Plural alias 'microservices' should resolve to 'microservice'."""
        result = alias_lookup.get("microservices")

        assert result is not None
        assert result.canonical_term == "microservice"

    def test_hyphenated_alias_resolves_to_canonical(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.6: Hyphenated alias 'api-gateway' should resolve to 'api_gateway'."""
        result = alias_lookup.get("api-gateway")

        assert result is not None
        assert result.canonical_term == "api_gateway"

    def test_acronym_alias_resolves_to_canonical(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.6: Acronym 'llm' should resolve to 'large_language_model'."""
        result = alias_lookup.get("llm")

        assert result is not None
        assert result.canonical_term == "large_language_model"

    def test_expanded_form_resolves_to_canonical(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC1.6: Expanded form should resolve to same canonical."""
        result = alias_lookup.get("large language model")

        assert result is not None
        assert result.canonical_term == "large_language_model"


# =============================================================================
# AC1.7: Test load from JSON file
# =============================================================================


class TestJsonLoading:
    """Tests for AC-1.5: Loading from JSON file."""

    def test_loads_from_json_file(self, temp_alias_file: Path) -> None:
        """AC1.7: Should load lookup data from JSON file."""
        from src.classifiers.alias_lookup import AliasLookup

        lookup = AliasLookup(lookup_path=temp_alias_file)

        # Verify data was loaded
        result = lookup.get("microservice")
        assert result is not None

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """AC1.7: Should raise FileNotFoundError for missing file."""
        from src.classifiers.alias_lookup import AliasLookup

        missing_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            AliasLookup(lookup_path=missing_file)

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """AC1.7: Should raise JSONDecodeError for invalid JSON."""
        from src.classifiers.alias_lookup import AliasLookup

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            AliasLookup(lookup_path=invalid_file)

    def test_empty_lookup_file(self, tmp_path: Path) -> None:
        """AC1.7: Should handle empty lookup gracefully."""
        from src.classifiers.alias_lookup import AliasLookup

        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")

        lookup = AliasLookup(lookup_path=empty_file)
        result = lookup.get("anything")
        assert result is None


# =============================================================================
# AC-1.6: O(1) Performance Test
# =============================================================================


class TestPerformance:
    """Tests for AC-1.6: O(1) lookup performance."""

    def test_lookup_performance_under_1ms(
        self, alias_lookup: "AliasLookup"
    ) -> None:
        """AC-1.6: Single lookup should complete in < 1ms."""
        start = time.perf_counter()
        _ = alias_lookup.get("microservice")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.001, f"Lookup took {elapsed*1000:.3f}ms, expected < 1ms"

    def test_o1_performance_scaling(self, tmp_path: Path) -> None:
        """AC-1.6: Lookup time should be constant regardless of dict size."""
        from src.classifiers.alias_lookup import AliasLookup

        # Create a large lookup table (10,000 entries)
        large_data = {
            f"term_{i}": {
                "canonical_term": f"canonical_{i}",
                "classification": "concept",
            }
            for i in range(10000)
        }
        large_file = tmp_path / "large_lookup.json"
        large_file.write_text(json.dumps(large_data))

        lookup = AliasLookup(lookup_path=large_file)

        # Time lookup of first, middle, and last entries
        times = []
        for key in ["term_0", "term_5000", "term_9999"]:
            start = time.perf_counter()
            _ = lookup.get(key)
            times.append(time.perf_counter() - start)

        # All lookups should be roughly the same time (within 10x)
        max_time = max(times)
        min_time = min(times)

        # O(1) means times should be relatively constant
        assert max_time < 0.001, "All lookups should be < 1ms"


# =============================================================================
# Classification Type Tests
# =============================================================================


class TestClassificationType:
    """Tests for classification type handling (concept vs keyword)."""

    def test_concept_classification(self, alias_lookup: "AliasLookup") -> None:
        """Should correctly identify concept classification."""
        result = alias_lookup.get("microservice")

        assert result is not None
        assert result.classification == "concept"

    def test_keyword_classification(self, alias_lookup: "AliasLookup") -> None:
        """Should correctly identify keyword classification."""
        result = alias_lookup.get("python")

        assert result is not None
        assert result.classification == "keyword"
