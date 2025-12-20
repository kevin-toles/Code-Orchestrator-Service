"""
Constants for Metadata Extraction Models.

WBS: CME-1.1.6 - Extract constants to separate module for S1192 compliance.

Reference:
- CODING_PATTERNS_ANALYSIS: S1192 - No duplicated string literals
- Anti-Pattern #7: Proper naming conventions

Usage:
    from src.models.constants import (
        DEFAULT_TOP_K_KEYWORDS,
        ERROR_TEXT_EMPTY,
    )
"""

from typing import Final

# =============================================================================
# Default Option Values
# =============================================================================

DEFAULT_TOP_K_KEYWORDS: Final[int] = 15
DEFAULT_TOP_K_CONCEPTS: Final[int] = 10
DEFAULT_MIN_KEYWORD_CONFIDENCE: Final[float] = 0.3
DEFAULT_MIN_CONCEPT_CONFIDENCE: Final[float] = 0.3
DEFAULT_SUMMARY_RATIO: Final[float] = 0.2


# =============================================================================
# Validation Bounds
# =============================================================================

MIN_TOP_K: Final[int] = 1
MAX_TOP_K: Final[int] = 100
MIN_CONFIDENCE: Final[float] = 0.0
MAX_CONFIDENCE: Final[float] = 1.0
MIN_SCORE: Final[float] = 0.0
MAX_SCORE: Final[float] = 1.0
MIN_TEXT_LENGTH: Final[int] = 1


# =============================================================================
# Error Messages
# =============================================================================

ERROR_TEXT_EMPTY: Final[str] = "text cannot be empty"
ERROR_TOP_K_RANGE: Final[str] = f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
ERROR_CONFIDENCE_RANGE: Final[str] = (
    f"confidence must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}"
)
ERROR_SCORE_RANGE: Final[str] = f"score must be between {MIN_SCORE} and {MAX_SCORE}"


# =============================================================================
# Export All
# =============================================================================

__all__ = [
    # Defaults
    "DEFAULT_TOP_K_KEYWORDS",
    "DEFAULT_TOP_K_CONCEPTS",
    "DEFAULT_MIN_KEYWORD_CONFIDENCE",
    "DEFAULT_MIN_CONCEPT_CONFIDENCE",
    "DEFAULT_SUMMARY_RATIO",
    # Bounds
    "MIN_TOP_K",
    "MAX_TOP_K",
    "MIN_CONFIDENCE",
    "MAX_CONFIDENCE",
    "MIN_SCORE",
    "MAX_SCORE",
    "MIN_TEXT_LENGTH",
    # Errors
    "ERROR_TEXT_EMPTY",
    "ERROR_TOP_K_RANGE",
    "ERROR_CONFIDENCE_RANGE",
    "ERROR_SCORE_RANGE",
]
