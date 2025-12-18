#!/usr/bin/env python3
"""
Fine-tune UniXcoder Model CLI

WBS: EEP-1.5.3 - Fine-tune UniXcoder for code similarity
AC-1.5.3.1: CLI wrapper for UniXcoder fine-tuning

Usage:
    python scripts/finetune_unixcoder.py --input code_pairs.jsonl --output models/unixcoder-finetuned
    python scripts/finetune_unixcoder.py --input code_pairs.jsonl --output models/unixcoder-finetuned --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embedding.unixcoder_trainer import UniXcoderTrainer
from src.models.embedding.config import (
    UniXcoderTrainingConfig,
    MODEL_UNIXCODER,
)


# UniXcoder-specific defaults per WBS AC-1.5.3.4
DEFAULT_UNIXCODER_EPOCHS = 3
DEFAULT_UNIXCODER_LEARNING_RATE = 1e-5
DEFAULT_UNIXCODER_BATCH_SIZE = 16
DEFAULT_UNIXCODER_MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune UniXcoder model on code pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic fine-tuning with defaults (3 epochs, lr=1e-5)
    python scripts/finetune_unixcoder.py \\
        --input code_pairs.jsonl \\
        --output models/unixcoder-book-corpus

    # Custom training parameters
    python scripts/finetune_unixcoder.py \\
        --input code_pairs.jsonl \\
        --output models/unixcoder-custom \\
        --epochs 5 \\
        --batch-size 8 \\
        --learning-rate 5e-6

    # Use specific base model
    python scripts/finetune_unixcoder.py \\
        --input code_pairs.jsonl \\
        --output models/unixcoder-custom \\
        --model microsoft/unixcoder-base-nine
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input JSONL file containing code pairs",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("models/unixcoder-book-corpus"),
        help="Output directory for fine-tuned model (default: models/unixcoder-book-corpus)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_UNIXCODER,
        help=f"Base model to fine-tune (default: {MODEL_UNIXCODER})",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=DEFAULT_UNIXCODER_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_UNIXCODER_EPOCHS})",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_UNIXCODER_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_UNIXCODER_BATCH_SIZE})",
    )

    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=DEFAULT_UNIXCODER_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_UNIXCODER_LEARNING_RATE})",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_UNIXCODER_MAX_LENGTH,
        help=f"Maximum sequence length for code (default: {DEFAULT_UNIXCODER_MAX_LENGTH})",
    )

    parser.add_argument(
        "--eval-input",
        type=Path,
        default=None,
        help="Optional evaluation pairs JSONL file",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_pairs(filepath: Path) -> list[dict]:
    """Load code pairs from JSONL file.

    Expected format per line:
    {"anchor": "code_snippet", "positive": "similar_code", "negative": "different_code", "concept": "optional"}

    Args:
        filepath: Path to JSONL file

    Returns:
        List of pair dictionaries
    """
    pairs = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                pair = json.loads(line)
                # Validate required fields
                if "anchor" in pair and "positive" in pair:
                    pairs.append(pair)
    return pairs


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file does not exist: {args.input}")
        return 1

    if not args.input.is_file():
        print(f"Error: Input path is not a file: {args.input}")
        return 1

    # Create configuration
    config = UniXcoderTrainingConfig(
        model_name=args.model,
        output_dir=str(args.output),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    if args.verbose:
        print("=" * 60)
        print("UniXcoder Fine-Tuning (EEP-1.5.3)")
        print("=" * 60)
        print(f"Input file: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Model: {args.model}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max length: {args.max_length}")
        print("=" * 60)

    # Load training pairs
    try:
        train_pairs = load_pairs(args.input)
        if args.verbose:
            print(f"Loaded {len(train_pairs)} code pairs")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return 1
    except Exception as e:
        print(f"Error loading code pairs: {e}")
        return 1

    if not train_pairs:
        print("Error: No valid code pairs found in input file")
        print("Expected format: {\"anchor\": \"code\", \"positive\": \"similar_code\"}")
        return 1

    # Load evaluation pairs if provided
    eval_pairs = None
    if args.eval_input:
        if not args.eval_input.exists():
            print(f"Warning: Evaluation file does not exist: {args.eval_input}")
        else:
            try:
                eval_pairs = load_pairs(args.eval_input)
                if args.verbose:
                    print(f"Loaded {len(eval_pairs)} evaluation pairs")
            except Exception as e:
                print(f"Warning: Error loading evaluation pairs: {e}")

    # Create trainer and train
    trainer = UniXcoderTrainer(config)

    if args.verbose:
        print("Starting training...")

    try:
        trainer.train(train_pairs, eval_pairs=eval_pairs)
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    if args.verbose:
        print("=" * 60)
        print("Training complete!")
        print(f"Model saved to: {args.output}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
