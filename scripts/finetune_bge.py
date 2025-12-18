#!/usr/bin/env python3
"""
Fine-tune BGE Model CLI

WBS: EEP-1.5.2 - Fine-tune BGE-large for text similarity
AC-1.5.2.1: CLI wrapper for BGE fine-tuning

Usage:
    python scripts/finetune_bge.py --input pairs.jsonl --output models/bge-finetuned
    python scripts/finetune_bge.py --input pairs.jsonl --output models/bge-finetuned --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embedding.bge_trainer import BGETrainer
from src.models.embedding.config import (
    BGETrainingConfig,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    MODEL_BGE_LARGE,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune BGE model on training pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic fine-tuning
    python scripts/finetune_bge.py \\
        --input training_pairs.jsonl \\
        --output models/bge-finetuned

    # Custom training parameters
    python scripts/finetune_bge.py \\
        --input training_pairs.jsonl \\
        --output models/bge-custom \\
        --epochs 10 \\
        --batch-size 16 \\
        --learning-rate 1e-5

    # Use specific base model
    python scripts/finetune_bge.py \\
        --input training_pairs.jsonl \\
        --output models/bge-custom \\
        --model BAAI/bge-base-en-v1.5
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input JSONL file containing training pairs",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("models/bge-finetuned"),
        help="Output directory for fine-tuned model (default: models/bge-finetuned)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_BGE_LARGE,
        help=f"Base model to fine-tune (default: {MODEL_BGE_LARGE})",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup steps ratio (default: 0.1)",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 mixed precision training (default: True)",
    )

    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 mixed precision training",
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
    """Load training pairs from JSONL file.

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
                pairs.append(json.loads(line))
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

    # Handle fp16 flag
    use_fp16 = args.fp16 and not args.no_fp16

    # Create configuration
    config = BGETrainingConfig(
        model_name=args.model,
        output_dir=str(args.output),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=use_fp16,
    )

    if args.verbose:
        print(f"Input file: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Model: {args.model}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"FP16: {use_fp16}")

    # Load training pairs
    try:
        train_pairs = load_pairs(args.input)
        if args.verbose:
            print(f"Loaded {len(train_pairs)} training pairs")
    except Exception as e:
        print(f"Error loading training pairs: {e}")
        return 1

    if not train_pairs:
        print("Error: No training pairs found in input file")
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
    trainer = BGETrainer(config)

    if args.verbose:
        print(f"Using loss function: {trainer.loss_function_name}")
        print("Starting training...")

    try:
        trainer.train(train_pairs, eval_pairs=eval_pairs)
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    if args.verbose:
        print(f"Training complete!")
        print(f"Model saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
