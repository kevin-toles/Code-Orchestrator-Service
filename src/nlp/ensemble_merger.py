"""EnsembleMerger - Merges YAKE and TextRank extraction results.

HCE-2.26 through HCE-2.27 GREEN Phase Implementation.
AC-2.7: EnsembleMerger.merge() unions results with source tracking.

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Low cognitive complexity through single-responsibility methods
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

DEFAULT_TEXTRANK_SCORE: Final[float] = 0.5
SOURCE_YAKE: Final[str] = "yake"
SOURCE_TEXTRANK: Final[str] = "textrank"
SOURCE_BOTH: Final[str] = "both"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExtractedTerm:
    """A term extracted by one or more extraction methods.

    Attributes:
        term: The extracted keyword or phrase.
        score: Relevance score (lower is better for YAKE, default 0.5 for TextRank).
        source: Origin of extraction ('yake', 'textrank', or 'both').
    """

    term: str
    score: float
    source: str


# =============================================================================
# EnsembleMerger
# =============================================================================


class EnsembleMerger:
    """Merges extraction results from YAKE and TextRank.

    AC-2.7: Unions results with source tracking.
    - YAKE terms retain their original scores
    - TextRank terms get default score of 0.5
    - Terms from both sources are marked as 'both' with YAKE score
    - Deduplication is case-insensitive
    """

    def merge(
        self,
        yake_terms: list[tuple[str, float]],
        textrank_terms: list[str],
    ) -> list[ExtractedTerm]:
        """Merge YAKE and TextRank extraction results.

        Args:
            yake_terms: List of (term, score) tuples from YAKE extractor.
            textrank_terms: List of term strings from TextRank extractor.

        Returns:
            List of ExtractedTerm objects with source tracking.
        """
        # Dictionary keyed by lowercase term for deduplication
        merged: dict[str, ExtractedTerm] = {}

        # Process YAKE terms first (they have scores)
        for term, score in yake_terms:
            key = term.lower()
            merged[key] = ExtractedTerm(term=term, score=score, source=SOURCE_YAKE)

        # Process TextRank terms
        for term in textrank_terms:
            key = term.lower()
            if key in merged:
                # Term already exists from YAKE - mark as 'both', keep YAKE score
                existing = merged[key]
                merged[key] = ExtractedTerm(
                    term=existing.term,  # Keep original case from YAKE
                    score=existing.score,  # Keep YAKE score
                    source=SOURCE_BOTH,
                )
            else:
                # New term from TextRank only
                merged[key] = ExtractedTerm(
                    term=term,
                    score=DEFAULT_TEXTRANK_SCORE,
                    source=SOURCE_TEXTRANK,
                )

        return list(merged.values())
