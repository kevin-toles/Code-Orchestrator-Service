"""TextRank Keyword Extractor - HCE-2.18 through HCE-2.20.

Wrapper around the summa library for TextRank-based keyword extraction.
TextRank uses graph-based ranking to identify important terms.

AC Reference:
- AC-2.2: summa>=1.2.0 dependency
- AC-2.4: extract() returns List[str]
- AC-2.6: Respects words config

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Low cognitive complexity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from summa import keywords as summa_keywords

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

DEFAULT_WORDS: Final[int] = 20
DEFAULT_SPLIT: Final[bool] = True


# =============================================================================
# Configuration Dataclass (AC-2.6)
# =============================================================================


@dataclass
class TextRankConfig:
    """Configuration for TextRank keyword extractor.

    Attributes:
        words: Maximum number of keywords to extract.
        split: Whether to split multi-word keywords into individual words.
    """

    words: int = DEFAULT_WORDS
    split: bool = DEFAULT_SPLIT


# =============================================================================
# TextRankExtractor Class (AC-2.4)
# =============================================================================


class TextRankExtractor:
    """TextRank keyword extractor wrapper.

    Extracts keywords using summa's TextRank implementation:
    - Builds a graph of word co-occurrences
    - Applies PageRank-like algorithm
    - Selects top-ranked words as keywords

    Note: TextRank does not provide scores in the same way YAKE does.
    Keywords are returned as strings only.
    """

    def __init__(self, config: TextRankConfig | None = None) -> None:
        """Initialize extractor with optional configuration.

        Args:
            config: Extraction configuration. Defaults to TextRankConfig().
        """
        self.config = config or TextRankConfig()

    def extract(self, text: str) -> list[str]:
        """Extract keywords from text using TextRank.

        Args:
            text: Input text to extract keywords from.

        Returns:
            List of keyword strings (no scores).
        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return []

        try:
            # Extract keywords using summa
            # summa.keywords returns a string of keywords separated by newlines
            # when split=True, or phrases when split=False
            result = summa_keywords.keywords(
                text,
                words=self.config.words,
                split=self.config.split,
            )

            # Handle None or empty result
            if not result:
                return []

            # If result is string, split into list
            if isinstance(result, str):
                keywords = [kw.strip() for kw in result.split("\n") if kw.strip()]
            else:
                keywords = list(result)

            # Limit to configured number of words
            return keywords[: self.config.words]

        except Exception:
            # TextRank may fail on very short or unusual text
            # Return empty list instead of raising
            return []
