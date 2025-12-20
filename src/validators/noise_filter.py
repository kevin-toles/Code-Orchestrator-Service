"""Noise filter for metadata extraction - WBS-1.2.

Filters noise terms from extracted keywords per AC-2.4.
Categories: watermarks, contractions, URL fragments, filler,
code artifacts, page markers, single chars, numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

# === Rejection Reason Constants (S1192 compliance) ===

NOISE_REASON_WATERMARK: Final[str] = "noise_watermark"
NOISE_REASON_CONTRACTION: Final[str] = "broken_contraction"
NOISE_REASON_URL_FRAGMENT: Final[str] = "noise_url_fragment"
NOISE_REASON_GENERIC_FILLER: Final[str] = "generic_filler"
NOISE_REASON_CODE_ARTIFACT: Final[str] = "code_artifact"
NOISE_REASON_PAGE_MARKER: Final[str] = "noise_page_marker"
NOISE_REASON_SINGLE_CHAR: Final[str] = "single_char"
NOISE_REASON_PURE_NUMBER: Final[str] = "pure_number"


# === Noise Term Lists ===

WATERMARK_TERMS: Final[frozenset[str]] = frozenset({
    "oceanofpdf",
    "ebscohost",
    "packt",
    "oreilly",
    "manning",
    "springer",
    "wiley",
    "apress",
    "pearson",
    "allitebooks",
    "foxebook",
    "pdfhive",
    "bookzz",
    "libgen",
})

URL_FRAGMENT_TERMS: Final[frozenset[str]] = frozenset({
    "www",
    "http",
    "https",
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "html",
    "htm",
    "php",
    "asp",
})

GENERIC_FILLER_TERMS: Final[frozenset[str]] = frozenset({
    "using",
    "used",
    "use",
    "one",
    "two",
    "three",
    "new",
    "first",
    "second",
    "third",
    "example",
    "examples",
    "also",
    "many",
    "much",
    "well",
    "way",
    "need",
    "make",
    "see",
    "get",
    "set",
    "can",
    "may",
    "like",
    "just",
    "even",
    "know",
    "take",
    "come",
    "want",
    "look",
    "give",
})

CODE_ARTIFACT_TERMS: Final[frozenset[str]] = frozenset({
    "self",
    "cls",
    "none",
    "true",
    "false",
    "return",
    "import",
    "from",
    "class",
    "def",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "with",
    "as",
    "lambda",
    "yield",
    "pass",
    "break",
    "continue",
    "raise",
    "assert",
    "global",
    "nonlocal",
    "del",
})

PAGE_MARKER_TERMS: Final[frozenset[str]] = frozenset({
    "chapter",
    "section",
    "figure",
    "table",
    "page",
    "appendix",
    "index",
    "contents",
    "bibliography",
    "references",
    "glossary",
    "preface",
    "introduction",
    "conclusion",
    "summary",
    "abstract",
    "acknowledgments",
})

# === Regex Patterns ===

# Pattern for broken contractions: 'll, n't, 's, 've, 'd, 're, 'm
CONTRACTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^['`]?(ll|nt|n't|s|ve|d|re|m)$",
    re.IGNORECASE,
)

# Pattern for underscore-prefixed code artifacts: _add, __init__, etc.
UNDERSCORE_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^_+\w*$",
)

# Pattern for pure numbers
PURE_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\d+$",
)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Result of filtering a single term."""

    is_rejected: bool
    reason: str | None = None


@dataclass(slots=True)
class BatchFilterResult:
    """Result of filtering a batch of terms."""

    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)


class NoiseFilter:
    """Filters noise terms from extracted keywords.

    Implements AC-2.4: Noise filtering with rejected.keywords
    and rejected.reasons.

    Noise categories:
    - OCR Watermarks: oceanofpdf, ebscohost, packt
    - Broken Contractions: 'll, n't, 's, 've
    - URL Fragments: www, http, https, com
    - Generic Filler: using, used, one, new, first
    - Code Artifacts: _add, __init__, self, cls
    - Page Markers: chapter, figure, page
    - Single Characters: a, b, c
    - Pure Numbers: 123, 2025
    """

    def __init__(self) -> None:
        """Initialize the noise filter."""
        # Pre-compute lowercase sets for O(1) lookup
        self._watermarks = WATERMARK_TERMS
        self._url_fragments = URL_FRAGMENT_TERMS
        self._filler_terms = GENERIC_FILLER_TERMS
        self._code_artifacts = CODE_ARTIFACT_TERMS
        self._page_markers = PAGE_MARKER_TERMS

    def filter_term(self, term: str) -> FilterResult:
        """Filter a single term and return rejection status.

        Args:
            term: The term to evaluate.

        Returns:
            FilterResult with is_rejected and reason if rejected.
        """
        if not term:
            return FilterResult(is_rejected=True, reason=NOISE_REASON_SINGLE_CHAR)

        term_lower = term.lower().strip()

        # Stage 1: Single character check
        if len(term_lower) == 1:
            return FilterResult(is_rejected=True, reason=NOISE_REASON_SINGLE_CHAR)

        # Stage 2: Pure number check
        if PURE_NUMBER_PATTERN.match(term_lower):
            return FilterResult(is_rejected=True, reason=NOISE_REASON_PURE_NUMBER)

        # Stage 3: Broken contraction check
        if CONTRACTION_PATTERN.match(term_lower):
            return FilterResult(is_rejected=True, reason=NOISE_REASON_CONTRACTION)

        # Stage 4: Underscore prefix check (code artifacts)
        if UNDERSCORE_PREFIX_PATTERN.match(term_lower):
            return FilterResult(is_rejected=True, reason=NOISE_REASON_CODE_ARTIFACT)

        # Stage 5: Watermark check
        if term_lower in self._watermarks:
            return FilterResult(is_rejected=True, reason=NOISE_REASON_WATERMARK)

        # Stage 6: URL fragment check
        if term_lower in self._url_fragments:
            return FilterResult(is_rejected=True, reason=NOISE_REASON_URL_FRAGMENT)

        # Stage 7: Generic filler check
        if term_lower in self._filler_terms:
            return FilterResult(is_rejected=True, reason=NOISE_REASON_GENERIC_FILLER)

        # Stage 8: Code artifact check
        if term_lower in self._code_artifacts:
            return FilterResult(is_rejected=True, reason=NOISE_REASON_CODE_ARTIFACT)

        # Stage 9: Page marker check
        if term_lower in self._page_markers:
            return FilterResult(is_rejected=True, reason=NOISE_REASON_PAGE_MARKER)

        # Term is valid
        return FilterResult(is_rejected=False, reason=None)

    def filter_batch(self, terms: list[str]) -> BatchFilterResult:
        """Filter a batch of terms and return categorized results.

        Args:
            terms: List of terms to filter.

        Returns:
            BatchFilterResult with accepted, rejected, and reasons.
        """
        result = BatchFilterResult()

        for term in terms:
            filter_result = self.filter_term(term)

            if filter_result.is_rejected:
                result.rejected.append(term)
                if filter_result.reason:
                    result.rejection_reasons[term] = filter_result.reason
            else:
                result.accepted.append(term)

        return result
