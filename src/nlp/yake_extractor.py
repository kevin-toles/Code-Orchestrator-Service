"""YAKE Keyword Extractor - HCE-2.10 through HCE-2.12.

Wrapper around the YAKE library for unsupervised keyword extraction.
YAKE (Yet Another Keyword Extractor) uses statistical text features.

AC Reference:
- AC-2.1: yake>=0.4.8 dependency
- AC-2.3: extract() returns List[Tuple[str, float]]
- AC-2.5: Respects top_n, n_gram_size config

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Low cognitive complexity
- #12: YAKE instance cached (singleton pattern)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import yake

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

DEFAULT_TOP_N: Final[int] = 20
DEFAULT_N_GRAM_SIZE: Final[int] = 3
DEFAULT_DEDUP_THRESHOLD: Final[float] = 0.9
DEFAULT_LANGUAGE: Final[str] = "en"
DEFAULT_WINDOW_SIZE: Final[int] = 1


# =============================================================================
# Configuration Dataclass (AC-2.5)
# =============================================================================


@dataclass
class YAKEConfig:
    """Configuration for YAKE keyword extractor.

    Attributes:
        top_n: Maximum number of keywords to extract.
        n_gram_size: Maximum n-gram size (1=single words, 3=up to trigrams).
        dedup_threshold: Deduplication threshold (0.0-1.0, higher=stricter).
        language: Language code for extraction.
        window_size: Window size for feature computation.
    """

    top_n: int = DEFAULT_TOP_N
    n_gram_size: int = DEFAULT_N_GRAM_SIZE
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    language: str = DEFAULT_LANGUAGE
    window_size: int = DEFAULT_WINDOW_SIZE


# =============================================================================
# YAKEExtractor Class (AC-2.3)
# =============================================================================


class YAKEExtractor:
    """YAKE keyword extractor wrapper.

    Extracts keywords using YAKE's statistical approach:
    - Position of terms in the document
    - Word frequency and co-occurrence
    - Word relatedness to context

    Scores: Lower is better (0.0 = most relevant).

    Anti-Pattern Compliance:
    - #12: YAKE instance is cached after first extraction
    """

    def __init__(self, config: YAKEConfig | None = None) -> None:
        """Initialize extractor with optional configuration.

        Args:
            config: Extraction configuration. Defaults to YAKEConfig().
        """
        self.config = config or YAKEConfig()
        self._yake_extractor: yake.KeywordExtractor | None = None

    def extract(self, text: str) -> list[tuple[str, float]]:
        """Extract keywords from text using YAKE.

        Args:
            text: Input text to extract keywords from.

        Returns:
            List of (keyword, score) tuples sorted by score ascending.
            Lower scores indicate more relevant keywords.
        """
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return []

        # Initialize YAKE extractor lazily (Anti-Pattern #12)
        if self._yake_extractor is None:
            self._yake_extractor = yake.KeywordExtractor(
                lan=self.config.language,
                n=self.config.n_gram_size,
                dedupLim=self.config.dedup_threshold,
                dedupFunc="seqm",
                windowsSize=self.config.window_size,
                top=self.config.top_n,
            )

        # Extract keywords
        keywords = self._yake_extractor.extract_keywords(text)

        # YAKE returns List[Tuple[str, float]] already sorted by score
        return list(keywords)
