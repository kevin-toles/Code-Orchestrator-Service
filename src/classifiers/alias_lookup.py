"""
Alias Lookup (Tier 1) - Hybrid Tiered Classifier.

This module provides O(1) hash-based lookup for known terms.
It serves as Tier 1 of the 4-tier classification pipeline.

Pattern: Hashed Feature (Machine Learning Design Patterns, Ch. 2)

AC-1.1: Exact match returns AliasLookupResult with confidence=1.0, tier_used=1
AC-1.2: Case-insensitive lookup
AC-1.3: Unknown term returns None
AC-1.4: Alias resolves to canonical term
AC-1.5: Loads from alias_lookup.json at startup
AC-1.6: O(1) lookup performance
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONFIDENCE: Final[float] = 1.0
TIER_ALIAS_LOOKUP: Final[int] = 1

# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class AliasLookupResult:
    """
    Result from Tier 1 alias lookup.

    Attributes:
        canonical_term: The normalized/canonical form of the term.
        classification: Either 'concept' or 'keyword'.
        confidence: Always 1.0 for exact alias matches.
        tier_used: Always 1 for alias lookup results.
    """

    canonical_term: str
    classification: str
    confidence: float
    tier_used: int


# =============================================================================
# Alias Lookup Class
# =============================================================================


class AliasLookup:
    """
    O(1) hash-based lookup for known terms (Tier 1).

    This class provides fast, constant-time lookup for terms that have been
    pre-mapped to their canonical forms. It supports:
    - Case-insensitive matching
    - Alias resolution (e.g., 'microservices' → 'microservice')
    - Classification type (concept vs keyword)

    Example:
        >>> lookup = AliasLookup(lookup_path=Path("config/alias_lookup.json"))
        >>> result = lookup.get("Microservices")
        >>> result.canonical_term
        'microservice'
        >>> result.confidence
        1.0

    Attributes:
        _lookup: Internal dictionary mapping normalized terms to their metadata.
    """

    __slots__ = ("_lookup",)

    def __init__(self, lookup_path: Path) -> None:
        """
        Initialize the alias lookup from a JSON file.

        Args:
            lookup_path: Path to the alias_lookup.json file.

        Raises:
            FileNotFoundError: If the lookup file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        self._lookup: dict[str, dict[str, str]] = self._load_lookup(lookup_path)

    def _load_lookup(self, lookup_path: Path) -> dict[str, dict[str, str]]:
        """
        Load the lookup dictionary from a JSON file.

        The JSON file should have the structure:
        {
            "term_key": {
                "canonical_term": "canonical_form",
                "classification": "concept" | "keyword"
            },
            ...
        }

        Args:
            lookup_path: Path to the JSON file.

        Returns:
            Dictionary with normalized keys (lowercase).

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file contains invalid JSON.
        """
        if not lookup_path.exists():
            raise FileNotFoundError(f"Alias lookup file not found: {lookup_path}")

        with lookup_path.open("r", encoding="utf-8") as f:
            raw_data: dict[str, dict[str, str]] = json.load(f)

        # Normalize all keys to lowercase for case-insensitive lookup
        return {key.lower(): value for key, value in raw_data.items()}

    def get(self, term: str) -> AliasLookupResult | None:
        """
        Look up a term and return its canonical form if found.

        This method provides O(1) lookup performance via dictionary hashing.
        Terms are normalized to lowercase before lookup.

        Args:
            term: The term to look up (case-insensitive).

        Returns:
            AliasLookupResult if found, None otherwise.
            When found, confidence is always 1.0 and tier_used is always 1.

        Example:
            >>> result = lookup.get("API Gateway")
            >>> result.canonical_term
            'api_gateway'
        """
        # Handle empty/whitespace-only strings
        normalized = term.strip().lower()
        if not normalized:
            return None

        # O(1) dictionary lookup
        entry = self._lookup.get(normalized)
        if entry is None:
            return None

        return AliasLookupResult(
            canonical_term=entry["canonical_term"],
            classification=entry["classification"],
            confidence=DEFAULT_CONFIDENCE,
            tier_used=TIER_ALIAS_LOOKUP,
        )

    def __len__(self) -> int:
        """Return the number of entries in the lookup table."""
        return len(self._lookup)

    def __contains__(self, term: str) -> bool:
        """Check if a term exists in the lookup table."""
        return self.get(term) is not None
