"""
Heuristic Filter (Tier 3) for noise term detection.

WBS: WBS-AC3 - Heuristic Filter
AC Block: AC-3.1 through AC-3.6

This module provides rule-based filtering to detect and reject noise terms
before they reach the LLM fallback tier. Implements 8 noise categories:

YAML-based categories (configurable):
1. watermarks - OCR watermarks (e.g., "oceanofpdf", "ebscohost")
2. url_fragments - Broken URL components (e.g., "www", "com")
3. generic_filler - Common words with no technical value (e.g., "using")
4. code_artifacts - Programming keywords (e.g., "self", "return")
5. page_markers - Document structure terms (e.g., "chapter", "section")

Regex-based categories (hardcoded):
6. broken_contraction - Contraction fragments (e.g., "'ll", "n't")
7. underscore_prefix - Internal identifiers (e.g., "_add", "__init__")
8. single_char - Single character terms
9. pure_number - Numeric-only strings
10. empty - Empty or whitespace-only strings

Pattern: Configuration-Driven Filter
Reference: CODING_PATTERNS_ANALYSIS.md - Anti-Pattern Compliance
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Mapping

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "noise_terms.yaml"

# Rejection reason prefixes (AC-8.1: no duplicated strings)
REJECTION_PREFIX = "noise_"

# Category to rejection reason mapping
CATEGORY_TO_REASON: dict[str, str] = {
    "watermarks": "noise_watermarks",
    "url_fragments": "noise_url_fragments",
    "generic_filler": "noise_generic_filler",
    "code_artifacts": "noise_code_artifacts",
    "page_markers": "noise_page_markers",
}

# Regex patterns for hardcoded rules
REGEX_BROKEN_CONTRACTION = re.compile(r"^(ll|nt|ve|re|s|d|m)$", re.IGNORECASE)
REGEX_UNDERSCORE_PREFIX = re.compile(r"^_+\w+")
REGEX_PURE_NUMBER = re.compile(r"^[\d.]+$")


# =============================================================================
# Exceptions
# =============================================================================


class HeuristicFilterConfigError(Exception):
    """Raised when config loading fails."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class HeuristicFilterResult:
    """Result from heuristic filter check.

    Attributes:
        rejection_reason: Reason code for rejection (e.g., "noise_watermarks")
        matched_term: The term or pattern that matched
        category: The category that triggered rejection
    """

    rejection_reason: str
    matched_term: str
    category: str


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class HeuristicFilterProtocol(Protocol):
    """Protocol for heuristic filter implementations.

    Enables dependency injection and test doubles (AC-8.4).
    """

    def check(self, term: str) -> HeuristicFilterResult | None:
        """Check if term is noise.

        Args:
            term: The term to check

        Returns:
            HeuristicFilterResult if noise, None if valid
        """
        ...

    def check_batch(
        self, terms: list[str]
    ) -> dict[str, HeuristicFilterResult | None]:
        """Check multiple terms for noise.

        Args:
            terms: List of terms to check

        Returns:
            Dict mapping each term to its result (None if valid)
        """
        ...

    def get_categories(self) -> list[str]:
        """Get list of YAML-based noise categories.

        Returns:
            List of category names
        """
        ...


# =============================================================================
# Main Implementation
# =============================================================================


class HeuristicFilter:
    """Rule-based filter for detecting noise terms.

    Implements HeuristicFilterProtocol for dependency injection.

    Usage:
        filter = HeuristicFilter()
        result = filter.check("oceanofpdf")
        if result:
            print(f"Rejected: {result.rejection_reason}")
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize HeuristicFilter with config.

        Args:
            config_path: Path to noise_terms.yaml. Uses default if None.

        Raises:
            HeuristicFilterConfigError: If config cannot be loaded
        """
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._noise_terms: dict[str, set[str]] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load noise terms from YAML config.

        Raises:
            HeuristicFilterConfigError: If file missing or invalid YAML
        """
        if not self._config_path.exists():
            msg = f"Config file not found: {self._config_path}"
            raise HeuristicFilterConfigError(msg)

        try:
            with open(self._config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in config: {e}"
            raise HeuristicFilterConfigError(msg) from e

        # Load each category as a lowercase set for O(1) lookup
        # Convert all terms to strings (YAML parses "true"/"false" as booleans)
        for category in CATEGORY_TO_REASON:
            terms = config.get(category, [])
            if terms:
                self._noise_terms[category] = {str(t).lower() for t in terms}
            else:
                self._noise_terms[category] = set()

    def check(self, term: str) -> HeuristicFilterResult | None:
        """Check if term is noise.

        Checks in order:
        1. Empty/whitespace
        2. Single character
        3. Pure number
        4. Underscore prefix
        5. Broken contraction
        6. YAML-based categories

        Args:
            term: The term to check

        Returns:
            HeuristicFilterResult if noise, None if valid
        """
        # Normalize for comparison
        normalized = term.strip().lower()

        # Check empty/whitespace (highest priority)
        if not normalized:
            return HeuristicFilterResult(
                rejection_reason="noise_empty",
                matched_term=term,
                category="empty",
            )

        # Check single character
        if len(normalized) == 1:
            return HeuristicFilterResult(
                rejection_reason="noise_single_char",
                matched_term=term,
                category="single_char",
            )

        # Check pure number (before underscore to catch "3.14")
        if REGEX_PURE_NUMBER.match(normalized):
            return HeuristicFilterResult(
                rejection_reason="noise_pure_number",
                matched_term=term,
                category="pure_number",
            )

        # Check underscore prefix (but allow underscores in middle, e.g., "api_gateway")
        if REGEX_UNDERSCORE_PREFIX.match(normalized):
            return HeuristicFilterResult(
                rejection_reason="noise_underscore_prefix",
                matched_term=term,
                category="underscore_prefix",
            )

        # Check broken contraction
        if REGEX_BROKEN_CONTRACTION.match(normalized):
            return HeuristicFilterResult(
                rejection_reason="noise_broken_contraction",
                matched_term=term,
                category="broken_contraction",
            )

        # Check YAML-based categories
        for category, terms_set in self._noise_terms.items():
            if normalized in terms_set:
                return HeuristicFilterResult(
                    rejection_reason=CATEGORY_TO_REASON[category],
                    matched_term=term,
                    category=category,
                )

        # Valid term - passes through
        return None

    def check_batch(
        self, terms: list[str]
    ) -> dict[str, HeuristicFilterResult | None]:
        """Check multiple terms for noise.

        Args:
            terms: List of terms to check

        Returns:
            Dict mapping each term to its result (None if valid)
        """
        return {term: self.check(term) for term in terms}

    def get_categories(self) -> list[str]:
        """Get list of YAML-based noise categories.

        Returns:
            List of category names loaded from config
        """
        return list(self._noise_terms.keys())


# =============================================================================
# Test Double (AC-8.4: Protocol-based fakes)
# =============================================================================


class FakeHeuristicFilter:
    """Fake HeuristicFilter for testing.

    Implements HeuristicFilterProtocol for use in unit tests.

    Usage:
        fake = FakeHeuristicFilter(responses={"noise": HeuristicFilterResult(...)})
        result = fake.check("noise")  # Returns configured response
    """

    def __init__(
        self,
        responses: Mapping[str, HeuristicFilterResult | None] | None = None,
    ) -> None:
        """Initialize with pre-configured responses.

        Args:
            responses: Dict mapping terms to expected results
        """
        self._responses: dict[str, HeuristicFilterResult | None] = (
            dict(responses) if responses else {}
        )

    def check(self, term: str) -> HeuristicFilterResult | None:
        """Return pre-configured response for term.

        Args:
            term: The term to check

        Returns:
            Configured result, or None if not configured
        """
        return self._responses.get(term)

    def check_batch(
        self, terms: list[str]
    ) -> dict[str, HeuristicFilterResult | None]:
        """Check multiple terms using configured responses.

        Args:
            terms: List of terms to check

        Returns:
            Dict mapping each term to its configured result
        """
        return {term: self.check(term) for term in terms}

    def get_categories(self) -> list[str]:
        """Return empty category list for fake.

        Returns:
            Empty list (fake has no real categories)
        """
        return []
