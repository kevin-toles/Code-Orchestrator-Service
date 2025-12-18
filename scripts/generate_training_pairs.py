#!/usr/bin/env python3
"""
Generate Training Pairs CLI

WBS: EEP-1.5.1 - Training Data Generation
AC-1.5.1.1: CLI wrapper for training pair generation

Usage:
    python scripts/generate_training_pairs.py --input-dir data/books --output pairs.jsonl
    python scripts/generate_training_pairs.py --input-dir data/books --output pairs.jsonl --pair-type text
    python scripts/generate_training_pairs.py --input-dir data/code --output code_pairs.jsonl --pair-type code
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embedding.training_data import (
    PairType,
    TrainingDataConfig,
    TrainingPairGenerator,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate training pairs for embedding fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate text pairs from book data
    python scripts/generate_training_pairs.py \\
        --input-dir data/books \\
        --output training_pairs.jsonl \\
        --pair-type text

    # Generate code pairs with custom config
    python scripts/generate_training_pairs.py \\
        --input-dir data/code \\
        --output code_pairs.jsonl \\
        --pair-type code \\
        --min-positive-score 0.8 \\
        --use-bm25-negatives
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Input directory containing source files",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSONL file for training pairs",
    )

    parser.add_argument(
        "--pair-type",
        "-t",
        type=str,
        choices=["text", "code", "both"],
        default="text",
        help="Type of pairs to generate (default: text)",
    )

    parser.add_argument(
        "--min-positive-score",
        type=float,
        default=0.7,
        help="Minimum similarity score for positive pairs (default: 0.7)",
    )

    parser.add_argument(
        "--use-bm25-negatives",
        action="store_true",
        help="Use BM25 for hard negative selection (default: False)",
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to generate (default: no limit)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        return 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create configuration
    config = TrainingDataConfig(
        min_positive_score=args.min_positive_score,
        use_bm25_negatives=args.use_bm25_negatives,
        max_pairs=args.max_pairs,
    )

    # Determine pair type
    pair_type = PairType(args.pair_type)

    if args.verbose:
        print(f"Input directory: {args.input_dir}")
        print(f"Output file: {args.output}")
        print(f"Pair type: {pair_type.value}")
        print(f"Config: {config}")

    # Generate pairs
    generator = TrainingPairGenerator(args.input_dir)

    try:
        pairs = generator.generate(pair_type=pair_type, config=config)
    except Exception as e:
        print(f"Error generating pairs: {e}")
        return 1

    # Write output
    with open(args.output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    if args.verbose:
        print(f"Generated {len(pairs)} pairs")
        print(f"Output written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
