"""Stemmer module for morphological deduplication - HCE-3.0.

Provides suffix stripping and stem-based deduplication for concept terms.
Ported from llm-document-enhancer with protected technical terms support.

AC Reference:
- AC-3.1: Stemmer module exists with SUFFIX_RULES and PROTECTED_TERMS
- AC-3.2: get_word_stem() strips suffixes correctly
- AC-3.3: Technical terms preserved (microservices, kubernetes, APIs)
- AC-3.4: deduplicate_by_stem() keeps first occurrence, returns count

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Low cognitive complexity through single-responsibility functions
"""

from __future__ import annotations

from typing import Final, TypeVar, Union

from src.nlp.ensemble_merger import ExtractedTerm

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

# Type alias for term types
T = TypeVar("T", str, ExtractedTerm)

# Suffix rules: (suffix, replacement, min_stem_length)
# Ordered from longest to shortest for correct matching
SUFFIX_RULES: Final[tuple[tuple[str, str, int], ...]] = (
    ("ation", "", 5),   # implementation → implement
    ("ity", "", 4),     # complexity → complex
    ("ing", "", 4),     # processing → process
    ("ies", "y", 3),    # dependencies → dependency
    ("es", "", 3),      # services → servic
    ("s", "", 4),       # models → model
)

# Protected technical terms that should NOT be stemmed
# Uses frozenset for O(1) lookup and immutability
PROTECTED_TERMS: Final[frozenset[str]] = frozenset({
    # Cloud/Infrastructure
    "microservices",
    "kubernetes",
    "serverless",
    
    # APIs
    "APIs",
    "GraphQL",
    "REST",
    "gRPC",
    
    # Databases
    "Redis",
    "Postgres",
    "MongoDB",
    
    # ML/AI
    "PyTorch",
    "TensorFlow",
    "Keras",
    
    # Languages/Frameworks
    "JavaScript",
    "TypeScript",
    "FastAPI",
    "NumPy",
    "Pandas",
})

# Lowercase lookup set for case-insensitive matching
_PROTECTED_TERMS_LOWER: Final[frozenset[str]] = frozenset(
    t.lower() for t in PROTECTED_TERMS
)


# =============================================================================
# Stemming Functions (AC-3.2, AC-3.3)
# =============================================================================


def get_word_stem(word: str) -> str:
    """Get the stem of a word by stripping known suffixes.

    Applies suffix stripping rules in order (longest suffix first).
    Protects technical terms from stemming.

    Args:
        word: The word to stem.

    Returns:
        The stemmed word, or original if protected or too short.

    Examples:
        >>> get_word_stem("implementation")
        'implement'
        >>> get_word_stem("complexity")
        'complex'
        >>> get_word_stem("microservices")  # protected
        'microservices'
    """
    if not word:
        return ""

    # Check if word is protected (case-insensitive)
    if word.lower() in _PROTECTED_TERMS_LOWER:
        return word

    # Try each suffix rule in order
    word_lower = word.lower()
    for suffix, replacement, min_stem_len in SUFFIX_RULES:
        if word_lower.endswith(suffix):
            stem_len = len(word) - len(suffix) + len(replacement)
            if stem_len >= min_stem_len:
                # Apply the rule: strip suffix, add replacement
                return word[: len(word) - len(suffix)] + replacement

    return word


# =============================================================================
# Deduplication Functions (AC-3.4)
# =============================================================================


def deduplicate_by_stem(
    terms: list[T],
) -> tuple[list[T], int]:
    """Deduplicate terms by their morphological stem.

    Groups terms by stem and keeps only the first occurrence of each stem.
    Works with both string lists and ExtractedTerm lists.

    Args:
        terms: List of terms (strings or ExtractedTerm objects).

    Returns:
        Tuple of (deduplicated_terms, removed_count).

    Examples:
        >>> deduplicate_by_stem(["model", "models", "modeling"])
        (['model'], 2)
        >>> deduplicate_by_stem(["complex", "complexity"])
        (['complex'], 1)
    """
    if not terms:
        return [], 0

    seen_stems: dict[str, bool] = {}
    result: list[T] = []

    for term in terms:
        # Extract the term string
        term_str = term.term if isinstance(term, ExtractedTerm) else term
        
        # Get the stem
        stem = get_word_stem(term_str).lower()
        
        # Keep only first occurrence of each stem
        if stem not in seen_stems:
            seen_stems[stem] = True
            result.append(term)

    removed_count = len(terms) - len(result)
    return result, removed_count
