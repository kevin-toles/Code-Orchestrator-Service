"""Dictionary validator for metadata extraction.

Validates extracted terms against approved keywords/concepts lists
from validated_term_filter.json.

Only terms that exist in the approved lists are accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Rejection reason for dictionary validation
DICT_REASON_NOT_IN_KEYWORDS: Final[str] = "not_in_approved_keywords"
DICT_REASON_NOT_IN_CONCEPTS: Final[str] = "not_in_approved_concepts"

# Default path to validated term filter
DEFAULT_FILTER_PATH: Final[Path] = (
    Path(__file__).parent.parent.parent / "data" / "validated_term_filter.json"
)


@dataclass(slots=True)
class DictionaryValidationResult:
    """Result of dictionary validation for a batch of terms."""

    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)


class DictionaryValidator:
    """Validates terms against approved keywords/concepts dictionary.

    Only accepts terms that exist in the validated_term_filter.json
    approved lists. This ensures we only extract known, validated terms.
    """

    def __init__(self, filter_path: Path | None = None) -> None:
        """Initialize the dictionary validator.

        Args:
            filter_path: Path to validated_term_filter.json.
                         Defaults to data/validated_term_filter.json.
        """
        self._filter_path = filter_path or DEFAULT_FILTER_PATH
        self._keywords: frozenset[str] = frozenset()
        self._concepts: frozenset[str] = frozenset()
        self._loaded = False
        self._load_filter()

    def _load_filter(self) -> None:
        """Load the validated term filter from JSON."""
        if not self._filter_path.exists():
            raise FileNotFoundError(
                f"Validated term filter not found: {self._filter_path}"
            )

        with open(self._filter_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load keywords and concepts as lowercase frozensets for O(1) lookup
        keywords_list = data.get("keywords", [])
        concepts_list = data.get("concepts", [])

        self._keywords = frozenset(kw.lower().strip() for kw in keywords_list)
        self._concepts = frozenset(c.lower().strip() for c in concepts_list)
        self._loaded = True

    @property
    def keyword_count(self) -> int:
        """Return number of approved keywords."""
        return len(self._keywords)

    @property
    def concept_count(self) -> int:
        """Return number of approved concepts."""
        return len(self._concepts)

    def is_valid_keyword(self, term: str) -> bool:
        """Check if term is in approved keywords list.

        Args:
            term: Term to validate.

        Returns:
            True if term is in approved keywords.
        """
        return term.lower().strip() in self._keywords

    def is_valid_concept(self, term: str) -> bool:
        """Check if term is in approved concepts list.

        Args:
            term: Term to validate.

        Returns:
            True if term is in approved concepts.
        """
        return term.lower().strip() in self._concepts

    def validate_keywords(self, terms: list[str]) -> DictionaryValidationResult:
        """Validate a batch of terms against approved keywords.

        Args:
            terms: List of terms to validate.

        Returns:
            DictionaryValidationResult with accepted/rejected terms.
        """
        result = DictionaryValidationResult()

        for term in terms:
            if self.is_valid_keyword(term):
                result.accepted.append(term)
            else:
                result.rejected.append(term)
                result.rejection_reasons[term] = DICT_REASON_NOT_IN_KEYWORDS

        return result

    def validate_concepts(self, terms: list[str]) -> DictionaryValidationResult:
        """Validate a batch of terms against approved concepts.

        Args:
            terms: List of terms to validate.

        Returns:
            DictionaryValidationResult with accepted/rejected terms.
        """
        result = DictionaryValidationResult()

        for term in terms:
            if self.is_valid_concept(term):
                result.accepted.append(term)
            else:
                result.rejected.append(term)
                result.rejection_reasons[term] = DICT_REASON_NOT_IN_CONCEPTS

        return result
