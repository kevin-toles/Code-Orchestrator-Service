#!/usr/bin/env python3
"""
Build Alias Lookup JSON from validated_term_filter.json.

This script generates the alias_lookup.json file used by the Tier 1
Alias Lookup component of the Hybrid Tiered Classifier.

Usage:
    python scripts/build_alias_lookup.py

Output:
    config/alias_lookup.json

Data Source:
    data/validated_term_filter.json

The script:
1. Loads concepts and keywords from the aggregated results
2. Creates canonical forms (lowercase, underscores for spaces)
3. Generates aliases for common variations (plurals, hyphenated, etc.)
4. Outputs a JSON file for O(1) lookup
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Final

# =============================================================================
# Constants
# =============================================================================

# Path constants
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent

INPUT_FILE: Final[Path] = (
    PROJECT_ROOT / "data" / "validated_term_filter.json"
)
OUTPUT_FILE: Final[Path] = PROJECT_ROOT / "config" / "alias_lookup.json"

# Classification types
CONCEPT: Final[str] = "concept"
KEYWORD: Final[str] = "keyword"


# =============================================================================
# Alias Generation Functions
# =============================================================================


def normalize_to_canonical(term: str) -> str:
    """
    Convert a term to its canonical form.

    - Lowercase
    - Replace spaces with underscores
    - Remove special characters except underscores

    Args:
        term: The raw term.

    Returns:
        Canonical form of the term.
    """
    # Lowercase and strip whitespace
    canonical = term.lower().strip()

    # Replace spaces and hyphens with underscores
    canonical = re.sub(r"[\s-]+", "_", canonical)

    # Remove any remaining special characters except underscores
    canonical = re.sub(r"[^\w]", "", canonical)

    return canonical


def generate_aliases(term: str) -> list[str]:
    """
    Generate common alias variations for a term.

    Variations include:
    - Original term (lowercase)
    - Hyphenated version (e.g., "api gateway" -> "api-gateway")
    - Underscored version (e.g., "api gateway" -> "api_gateway")
    - No-space version (e.g., "api gateway" -> "apigateway")

    Args:
        term: The original term.

    Returns:
        List of alias variations.
    """
    aliases = set()
    lower_term = term.lower().strip()

    # Original lowercase
    aliases.add(lower_term)

    # If multi-word, add variations
    if " " in lower_term:
        # Hyphenated
        aliases.add(lower_term.replace(" ", "-"))
        # Underscored
        aliases.add(lower_term.replace(" ", "_"))
        # Concatenated (no spaces)
        aliases.add(lower_term.replace(" ", ""))

    # If hyphenated, add space version
    if "-" in lower_term:
        aliases.add(lower_term.replace("-", " "))
        aliases.add(lower_term.replace("-", "_"))

    # If underscored, add space version
    if "_" in lower_term:
        aliases.add(lower_term.replace("_", " "))
        aliases.add(lower_term.replace("_", "-"))

    return list(aliases)


def build_lookup_entry(term: str, classification: str) -> dict[str, dict[str, str]]:
    """
    Build lookup entries for a term and its aliases.

    Args:
        term: The term to process.
        classification: Either 'concept' or 'keyword'.

    Returns:
        Dictionary mapping aliases to their canonical entry.
    """
    canonical = normalize_to_canonical(term)
    aliases = generate_aliases(term)

    entry = {
        "canonical_term": canonical,
        "classification": classification,
    }

    return {alias: entry for alias in aliases}


# =============================================================================
# Main Script
# =============================================================================


def load_aggregated_results(input_path: Path) -> dict[str, list[str] | str | dict[str, int]]:
    """Load the validated_term_filter.json file."""
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        result: dict[str, list[str] | str | dict[str, int]] = json.load(f)
        return result


def build_alias_lookup(concepts: list[str], keywords: list[str]) -> dict[str, dict[str, str]]:
    """
    Build the complete alias lookup dictionary.

    Args:
        concepts: List of concept terms.
        keywords: List of keyword terms.

    Returns:
        Complete lookup dictionary.
    """
    lookup: dict[str, dict[str, str]] = {}

    # Process concepts
    for term in concepts:
        entries = build_lookup_entry(term, CONCEPT)
        lookup.update(entries)

    # Process keywords
    for term in keywords:
        entries = build_lookup_entry(term, KEYWORD)
        # Only add if not already present (concepts take priority)
        for alias, entry in entries.items():
            if alias not in lookup:
                lookup[alias] = entry

    return lookup


def save_lookup(lookup: dict[str, dict[str, str]], output_path: Path) -> None:
    """Save the lookup dictionary to JSON file."""
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    output_data = {
        "_metadata": {
            "generated": datetime.now(tz=None).isoformat(),
            "source": str(INPUT_FILE),
            "total_entries": len(lookup),
            "generator": "build_alias_lookup.py",
        },
        **lookup,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    print(f"✓ Saved alias lookup to: {output_path}")
    print(f"  Total entries: {len(lookup):,}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Building Alias Lookup for Hybrid Tiered Classifier")
    print("=" * 60)

    # Load input data
    print(f"\nLoading: {INPUT_FILE}")
    data = load_aggregated_results(INPUT_FILE)

    concepts_raw = data.get("concepts", [])
    keywords_raw = data.get("keywords", [])

    # Type assertion for list[str]
    concepts: list[str] = concepts_raw if isinstance(concepts_raw, list) else []
    keywords: list[str] = keywords_raw if isinstance(keywords_raw, list) else []

    print(f"  Concepts: {len(concepts):,}")
    print(f"  Keywords: {len(keywords):,}")

    # Build lookup
    print("\nBuilding lookup table...")
    lookup = build_alias_lookup(concepts, keywords)

    # Save output
    print(f"\nSaving to: {OUTPUT_FILE}")
    save_lookup(lookup, OUTPUT_FILE)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
