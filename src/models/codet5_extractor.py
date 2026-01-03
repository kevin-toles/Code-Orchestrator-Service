"""
Code-Orchestrator-Service - CodeT5+ Term Extractor

WBS 2.2: Term Extractor (Model Wrapper)
Extracts technical terms from text using CodeT5+ seq2seq model.

Uses locally hosted Salesforce/codet5p-220m for keyword generation.

Architecture Role: GENERATOR
- Encoder-decoder architecture enables text generation
- Trained on NL↔Code pairs
- Generates primary_terms, related_terms, code_patterns

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- Generative extraction: Use T5 model to generate relevant terms
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (caches tokenizer and model)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.core.logging import get_logger

# Get logger
logger = get_logger(__name__)

# Local model path
_LOCAL_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "codet5"

# Singleton model instances
_tokenizer: Any | None = None
_model: T5ForConditionalGeneration | None = None


def _get_codet5() -> tuple[Any, T5ForConditionalGeneration]:
    """Get or create singleton CodeT5+ model and tokenizer from local path."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_path = str(_LOCAL_MODEL_PATH)
        if not _LOCAL_MODEL_PATH.exists():
            # Fallback to HuggingFace if local not found
            model_path = "Salesforce/codet5p-220m"
            logger.warning("local_codet5_not_found", path=str(_LOCAL_MODEL_PATH), using=model_path)
        else:
            logger.info("loading_codet5_from_local", path=model_path)

        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _model = T5ForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        logger.info("codet5_model_loaded", device=device)

    return _tokenizer, _model


class ExtractionResult(BaseModel):
    """Result of term extraction from text.

    WBS 2.2.2: Output structure for extract_terms()
    """

    primary_terms: list[str]
    """Primary technical terms identified in the text."""

    related_terms: list[str]
    """Related/supporting terms identified in the text."""


class CodeT5Extractor:
    """CodeT5+ term extractor using generative approach.

    WBS 2.2: Extracts technical terms from chapter text using CodeT5+ seq2seq model.
    Uses locally hosted Salesforce/codet5p-220m.

    Architecture Role: GENERATOR (STATE 1: EXTRACTION)
    - Input: "Extract technical search terms for: <query>"
    - Output: primary_terms[], related_terms[], code_patterns[]

    Approach:
    1. Prompt CodeT5+ with extraction task
    2. Generate keyword sequences using beam search
    3. Parse and deduplicate extracted terms
    4. Return top terms as primary, remaining as related
    """

    # Common stopwords to filter from generated output
    STOPWORDS = frozenset([
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "it", "its", "this", "that", "these", "those", "such", "very", "so",
        "just", "also", "than", "more", "most", "less", "least", "any", "some",
    ])

    def __init__(self) -> None:
        """Initialize CodeT5+ extractor with local model."""
        self._tokenizer, self._model = _get_codet5()
        self._device = next(self._model.parameters()).device
        logger.info("codet5_extractor_initialized", device=str(self._device))

    def _generate_candidates(self, text: str) -> list[str]:
        """Generate n-gram candidates from text as fallback.

        Args:
            text: Input text

        Returns:
            List of candidate terms (1-grams, 2-grams, 3-grams)
        """
        # Handle empty text
        if not text or not text.strip():
            return []

        # Clean text and split into words
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", text)

        # Filter short words and stopwords for base terms
        words = [w for w in words if len(w) > 1 and w.lower() not in self.STOPWORDS]

        if not words:
            return []

        candidates = []

        # 1-grams (individual words)
        candidates.extend(words)

        # 2-grams
        for i in range(len(words) - 1):
            candidates.append(f"{words[i]} {words[i+1]}")

        # 3-grams
        for i in range(len(words) - 2):
            candidates.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            lower_c = c.lower()
            if lower_c not in seen:
                seen.add(lower_c)
                unique.append(c)

        return unique

    def _parse_generated_terms(self, generated_text: str) -> list[str]:
        """Parse individual terms from generated text.

        Args:
            generated_text: Raw output from CodeT5+

        Returns:
            List of parsed individual terms
        """
        terms = []

        # Split on common delimiters
        for delimiter in [",", ";", "\n", "|", "•", "-", ":"]:
            if delimiter in generated_text:
                parts = generated_text.split(delimiter)
                for p in parts:
                    cleaned = p.strip()
                    if cleaned and len(cleaned) > 1:
                        # Filter stopwords
                        if cleaned.lower() not in self.STOPWORDS:
                            terms.append(cleaned)
                return terms

        # No delimiter found - extract words/phrases
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", generated_text)
        for w in words:
            if len(w) > 2 and w.lower() not in self.STOPWORDS:
                terms.append(w)

        return terms

    def extract_terms(
        self,
        text: str,
        top_k: int = 5,
        related_k: int = 5,
        max_length: int = 64,
        num_beams: int = 4,
    ) -> ExtractionResult:
        """Extract technical terms from text using CodeT5+ generation.

        WBS 2.2.2: Input text → Output primary_terms[], related_terms[]

        Architecture: STATE 1 EXTRACTION using CodeT5+ seq2seq

        Args:
            text: Input text to extract terms from
            top_k: Number of primary terms to return
            related_k: Number of related terms to return
            max_length: Maximum generation length
            num_beams: Number of beams for beam search

        Returns:
            ExtractionResult with primary and related terms
        """
        logger.debug("extracting_terms_codet5", text_length=len(text))

        # Handle empty or whitespace-only input
        if not text or not text.strip():
            logger.info("empty_text_input")
            return ExtractionResult(primary_terms=[], related_terms=[])

        # Truncate input if too long (CodeT5+ has 512 token limit)
        truncated_text = text[:1500] if len(text) > 1500 else text

        # Format as code-like prompt (CodeT5+ works better with code-style input)
        prompt = f"# Extract keywords from: {truncated_text}"

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self._device)

        # Generate with beam search
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=min(num_beams, 3),
                early_stopping=True,
                do_sample=False,
            )

        # Decode generated sequences
        generated_texts = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Parse terms from all generated sequences
        all_terms: list[str] = []
        for gen_text in generated_texts:
            parsed = self._parse_generated_terms(gen_text)
            all_terms.extend(parsed)

        # If CodeT5+ output is poor, fall back to n-gram extraction
        if len(all_terms) < 3:
            logger.info("codet5_fallback_to_ngrams", generated_count=len(all_terms))
            all_terms = self._generate_candidates(text)

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in all_terms:
            lower_term = term.lower().strip()
            if lower_term and lower_term not in seen and len(lower_term) > 1:
                seen.add(lower_term)
                unique_terms.append(term.strip())

        # Split into primary and related
        primary_terms = unique_terms[:top_k]
        related_terms = unique_terms[top_k:top_k + related_k]

        logger.info(
            "codet5_terms_extracted",
            primary_count=len(primary_terms),
            related_count=len(related_terms),
        )

        return ExtractionResult(
            primary_terms=primary_terms,
            related_terms=related_terms,
        )

    def extract_terms_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Batch process multiple texts.

        WBS 2.2.3: Process multiple chapters efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of ExtractionResult, one per input text
        """
        if not texts:
            return []

        logger.info("codet5_batch_extraction_started", count=len(texts))

        results = []
        for text in texts:
            result = self.extract_terms(text)
            results.append(result)

        logger.info("codet5_batch_extraction_completed", count=len(results))
        return results
