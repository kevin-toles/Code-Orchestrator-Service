"""
Tests for scripts/build_alias_lookup.py - WBS-AC7.5.

TDD RED Phase Tests for:
- AC-7.5: Alias lookup generation from taxonomy/aggregated results
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Import from existing build_alias_lookup.py
from scripts.build_alias_lookup import (
    CONCEPT,
    KEYWORD,
    build_alias_lookup,
    build_lookup_entry,
    generate_aliases,
    load_aggregated_results,
    normalize_to_canonical,
    save_lookup,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_aggregated_data() -> dict[str, Any]:
    """Sample validated_term_filter.json structure."""
    return {
        "timestamp": "2025-12-23",
        "description": "Test data",
        "summary": {
            "total_concepts": 3,
            "total_keywords": 3,
        },
        "concepts": [
            "api gateway",
            "machine learning",
            "microservices",
        ],
        "keywords": [
            "docker",
            "kubernetes",
            "python",
        ],
    }


@pytest.fixture
def temp_input_file(sample_aggregated_data: dict[str, Any]) -> Path:
    """Create a temporary input JSON file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        json.dump(sample_aggregated_data, f)
        return Path(f.name)


@pytest.fixture
def temp_output_file() -> Path:
    """Create a temporary output path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        return Path(f.name)


# =============================================================================
# AC-7.5: Test normalize_to_canonical
# =============================================================================


class TestNormalizeToCanonical:
    """Tests for normalize_to_canonical()."""

    def test_lowercase_conversion(self) -> None:
        """Converts to lowercase."""
        assert normalize_to_canonical("API Gateway") == "api_gateway"

    def test_space_to_underscore(self) -> None:
        """Replaces spaces with underscores."""
        assert normalize_to_canonical("machine learning") == "machine_learning"

    def test_hyphen_to_underscore(self) -> None:
        """Replaces hyphens with underscores."""
        assert normalize_to_canonical("ci-cd") == "ci_cd"

    def test_strips_whitespace(self) -> None:
        """Strips leading/trailing whitespace."""
        assert normalize_to_canonical("  docker  ") == "docker"

    def test_removes_special_characters(self) -> None:
        """Removes special characters except underscores."""
        assert normalize_to_canonical("c++") == "c"
        assert normalize_to_canonical("node.js") == "nodejs"

    def test_single_word(self) -> None:
        """Single words remain unchanged (except lowercase)."""
        assert normalize_to_canonical("Kubernetes") == "kubernetes"


# =============================================================================
# AC-7.5: Test generate_aliases
# =============================================================================


class TestGenerateAliases:
    """Tests for generate_aliases()."""

    def test_includes_original_lowercase(self) -> None:
        """Includes the original term in lowercase."""
        aliases = generate_aliases("API Gateway")
        assert "api gateway" in aliases

    def test_includes_hyphenated_version(self) -> None:
        """Multi-word terms include hyphenated version."""
        aliases = generate_aliases("api gateway")
        assert "api-gateway" in aliases

    def test_includes_underscored_version(self) -> None:
        """Multi-word terms include underscored version."""
        aliases = generate_aliases("api gateway")
        assert "api_gateway" in aliases

    def test_includes_concatenated_version(self) -> None:
        """Multi-word terms include concatenated version."""
        aliases = generate_aliases("api gateway")
        assert "apigateway" in aliases

    def test_hyphenated_generates_space_version(self) -> None:
        """Hyphenated terms generate space version."""
        aliases = generate_aliases("ci-cd")
        assert "ci cd" in aliases

    def test_underscored_generates_space_version(self) -> None:
        """Underscored terms generate space version."""
        aliases = generate_aliases("machine_learning")
        assert "machine learning" in aliases

    def test_single_word_returns_lowercase(self) -> None:
        """Single words return just lowercase."""
        aliases = generate_aliases("Docker")
        assert "docker" in aliases


# =============================================================================
# AC-7.5: Test build_lookup_entry
# =============================================================================


class TestBuildLookupEntry:
    """Tests for build_lookup_entry()."""

    def test_creates_entries_for_all_aliases(self) -> None:
        """Creates an entry for each alias."""
        entries = build_lookup_entry("api gateway", CONCEPT)

        assert "api gateway" in entries
        assert "api-gateway" in entries
        assert "api_gateway" in entries

    def test_entry_has_canonical_term(self) -> None:
        """Each entry has canonical_term field."""
        entries = build_lookup_entry("api gateway", CONCEPT)

        for alias, entry in entries.items():
            assert entry["canonical_term"] == "api_gateway"

    def test_entry_has_classification(self) -> None:
        """Each entry has classification field."""
        entries = build_lookup_entry("machine learning", CONCEPT)

        for alias, entry in entries.items():
            assert entry["classification"] == CONCEPT

    def test_keyword_classification(self) -> None:
        """Keywords have 'keyword' classification."""
        entries = build_lookup_entry("docker", KEYWORD)

        for alias, entry in entries.items():
            assert entry["classification"] == KEYWORD


# =============================================================================
# AC-7.5: Test load_aggregated_results
# =============================================================================


class TestLoadAggregatedResults:
    """Tests for load_aggregated_results()."""

    def test_loads_valid_json(self, temp_input_file: Path) -> None:
        """Loads valid JSON file."""
        data = load_aggregated_results(temp_input_file)

        assert "concepts" in data
        assert "keywords" in data

    def test_file_not_found_exits(self) -> None:
        """Missing file causes system exit."""
        with pytest.raises(SystemExit):
            load_aggregated_results(Path("/nonexistent/file.json"))

    def test_returns_concepts_list(self, temp_input_file: Path) -> None:
        """Returns concepts as a list."""
        data = load_aggregated_results(temp_input_file)

        assert isinstance(data["concepts"], list)
        assert "api gateway" in data["concepts"]

    def test_returns_keywords_list(self, temp_input_file: Path) -> None:
        """Returns keywords as a list."""
        data = load_aggregated_results(temp_input_file)

        assert isinstance(data["keywords"], list)
        assert "docker" in data["keywords"]


# =============================================================================
# AC-7.5: Test build_alias_lookup
# =============================================================================


class TestBuildAliasLookup:
    """Tests for build_alias_lookup()."""

    def test_builds_lookup_from_concepts_and_keywords(self) -> None:
        """Builds lookup from concepts and keywords lists."""
        concepts = ["microservices", "api gateway"]
        keywords = ["docker", "python"]

        lookup = build_alias_lookup(concepts, keywords)

        assert "microservices" in lookup
        assert "api gateway" in lookup
        assert "docker" in lookup
        assert "python" in lookup

    def test_concepts_classified_as_concept(self) -> None:
        """Concepts have 'concept' classification."""
        concepts = ["microservices"]
        keywords: list[str] = []

        lookup = build_alias_lookup(concepts, keywords)

        assert lookup["microservices"]["classification"] == CONCEPT

    def test_keywords_classified_as_keyword(self) -> None:
        """Keywords have 'keyword' classification."""
        concepts: list[str] = []
        keywords = ["docker"]

        lookup = build_alias_lookup(concepts, keywords)

        assert lookup["docker"]["classification"] == KEYWORD

    def test_concepts_take_priority_over_keywords(self) -> None:
        """When a term is both concept and keyword, concept wins."""
        # If same term appears in both, concept classification takes priority
        concepts = ["kubernetes"]
        keywords = ["kubernetes"]

        lookup = build_alias_lookup(concepts, keywords)

        assert lookup["kubernetes"]["classification"] == CONCEPT

    def test_generates_all_alias_variations(self) -> None:
        """Generates all alias variations for multi-word terms."""
        concepts = ["api gateway"]
        keywords: list[str] = []

        lookup = build_alias_lookup(concepts, keywords)

        assert "api gateway" in lookup
        assert "api-gateway" in lookup
        assert "api_gateway" in lookup
        assert "apigateway" in lookup


# =============================================================================
# AC-7.5: Test save_lookup
# =============================================================================


class TestSaveLookup:
    """Tests for save_lookup()."""

    def test_saves_to_json_file(self, temp_output_file: Path) -> None:
        """Saves lookup to JSON file."""
        lookup = {"test": {"canonical_term": "test", "classification": CONCEPT}}

        save_lookup(lookup, temp_output_file)

        assert temp_output_file.exists()

    def test_includes_metadata(self, temp_output_file: Path) -> None:
        """Output includes _metadata section."""
        lookup = {"test": {"canonical_term": "test", "classification": CONCEPT}}

        save_lookup(lookup, temp_output_file)

        with temp_output_file.open("r") as f:
            data = json.load(f)

        assert "_metadata" in data
        assert "generated" in data["_metadata"]
        assert "total_entries" in data["_metadata"]

    def test_creates_parent_directories(self) -> None:
        """Creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "config" / "nested" / "lookup.json"
            lookup = {"test": {"canonical_term": "test", "classification": CONCEPT}}

            save_lookup(lookup, nested_path)

            assert nested_path.exists()

    def test_preserves_all_entries(self, temp_output_file: Path) -> None:
        """All lookup entries preserved in output."""
        lookup = {
            "api gateway": {"canonical_term": "api_gateway", "classification": CONCEPT},
            "api-gateway": {"canonical_term": "api_gateway", "classification": CONCEPT},
            "docker": {"canonical_term": "docker", "classification": KEYWORD},
        }

        save_lookup(lookup, temp_output_file)

        with temp_output_file.open("r") as f:
            data = json.load(f)

        assert "api gateway" in data
        assert "api-gateway" in data
        assert "docker" in data


# =============================================================================
# AC-7.5: Integration Test
# =============================================================================


class TestAliasLookupGeneration:
    """Integration tests for alias lookup generation."""

    def test_full_generation_pipeline(
        self, temp_input_file: Path, temp_output_file: Path
    ) -> None:
        """Full pipeline: load -> build -> save."""
        # Load
        data = load_aggregated_results(temp_input_file)

        # Build
        lookup = build_alias_lookup(data["concepts"], data["keywords"])

        # Save
        save_lookup(lookup, temp_output_file)

        # Verify output
        with temp_output_file.open("r") as f:
            output_data = json.load(f)

        # Check concepts
        assert "api gateway" in output_data
        assert output_data["api gateway"]["classification"] == CONCEPT

        # Check keywords
        assert "docker" in output_data
        assert output_data["docker"]["classification"] == KEYWORD

        # Check aliases generated
        assert "api-gateway" in output_data
        assert "machine_learning" in output_data

    def test_lookup_usable_by_alias_lookup_class(
        self, temp_input_file: Path, temp_output_file: Path
    ) -> None:
        """Generated lookup can be used by AliasLookup class."""
        # Generate lookup
        data = load_aggregated_results(temp_input_file)
        lookup = build_alias_lookup(data["concepts"], data["keywords"])
        save_lookup(lookup, temp_output_file)

        # Use with AliasLookup class
        from src.classifiers.alias_lookup import AliasLookup

        alias_lookup = AliasLookup(lookup_path=temp_output_file)

        # Test lookups
        result = alias_lookup.get("api gateway")
        assert result is not None
        assert result.classification == CONCEPT
        assert result.canonical_term == "api_gateway"

        result = alias_lookup.get("docker")
        assert result is not None
        assert result.classification == KEYWORD
