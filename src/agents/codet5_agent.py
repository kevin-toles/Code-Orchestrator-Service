"""
Code-Orchestrator-Service - CodeT5+ Generator Agent

WBS 2.2: CodeT5+ Generator Agent
Extracts technical terms from text using CodeT5+ model.

Patterns Applied:
- Agent Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- FakeModelRegistry for testing (no real HuggingFace model in unit tests)
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached model from registry)
"""

from __future__ import annotations

import signal
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.agents.registry import ModelRegistry
from src.core.exceptions import ModelNotReadyError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.agents.registry import ModelRegistryProtocol

# Get logger
logger = get_logger(__name__)


class ExtractionResult(BaseModel):
    """Result of term extraction from text.

    WBS 2.2.2: Output structure for extract_terms()
    """

    primary_terms: list[str]
    """Primary technical terms identified in the text."""

    related_terms: list[str]
    """Related/supporting terms identified in the text."""


class CodeT5Agent:
    """CodeT5+ Generator Agent for term extraction.

    WBS 2.2: Extracts technical terms from chapter text using CodeT5+ model.

    Pattern: Agent Pattern per Kitchen Brigade model
    - Generator role: Creates/extracts candidate terms
    - Uses model from registry (Anti-Pattern #12 prevention)

    Attributes:
        DEFAULT_TIMEOUT_SECONDS: Default inference timeout (30s per WBS spec)
    """

    DEFAULT_TIMEOUT_SECONDS: float = 30.0

    def __init__(self, registry: ModelRegistryProtocol | None = None) -> None:
        """Initialize CodeT5 agent.

        Args:
            registry: ModelRegistry instance (or FakeModelRegistry for testing)

        Raises:
            ModelNotReadyError: If codet5 model not loaded in registry
        """
        self._registry = registry or ModelRegistry.get_registry()

        # Get model from registry
        model_tuple = self._registry.get_model("codet5")
        if model_tuple is None:
            raise ModelNotReadyError("CodeT5 model not loaded in registry")

        self._model, self._tokenizer = model_tuple
        logger.info("codet5_agent_initialized")

    @contextmanager
    def _timeout_context(self, seconds: float) -> Generator[None, None, None]:
        """Context manager for inference timeout.

        Args:
            seconds: Timeout in seconds

        Yields:
            None

        Raises:
            TimeoutError: If code in context exceeds timeout
        """
        def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            raise TimeoutError(f"Inference exceeded {seconds}s timeout")

        # Only use SIGALRM on Unix-like systems
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def extract_terms(
        self,
        text: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> ExtractionResult:
        """Extract technical terms from text.

        WBS 2.2.2: Input text → Output primary_terms[], related_terms[]

        Args:
            text: Input text to extract terms from
            timeout_seconds: Maximum time for inference (default 30s)

        Returns:
            ExtractionResult with primary and related terms

        Raises:
            TimeoutError: If inference exceeds timeout
        """
        logger.debug("extracting_terms", text_length=len(text))

        # Build prompt for term extraction
        prompt = f"Extract technical terms from: {text}"

        try:
            with self._timeout_context(timeout_seconds):
                # Tokenize
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                )

                # Generate
                outputs = self._model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                )

                # Decode
                generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        except TimeoutError:
            logger.warning("extraction_timeout", timeout=timeout_seconds)
            raise

        # Parse generated output
        primary_terms, related_terms = self._parse_output(generated)

        logger.info(
            "terms_extracted",
            primary_count=len(primary_terms),
            related_count=len(related_terms),
        )

        return ExtractionResult(
            primary_terms=primary_terms,
            related_terms=related_terms,
        )

    def _parse_output(self, generated: str) -> tuple[list[str], list[str]]:
        """Parse model output into primary and related terms.

        Args:
            generated: Raw model output string

        Returns:
            Tuple of (primary_terms, related_terms)
        """
        primary_terms: list[str] = []
        related_terms: list[str] = []

        # Look for "primary:" and "related:" markers
        lower_gen = generated.lower()

        if "primary:" in lower_gen:
            parts = generated.split("primary:", 1)
            if len(parts) > 1:
                primary_part = parts[1]
                if "related:" in primary_part.lower():
                    primary_part = primary_part.lower().split("related:")[0]
                primary_terms = self._extract_terms_from_text(primary_part)

        if "related:" in lower_gen:
            parts = lower_gen.split("related:", 1)
            if len(parts) > 1:
                related_terms = self._extract_terms_from_text(parts[1])

        # Fallback: if no markers, split by comma/semicolon
        if not primary_terms and not related_terms:
            all_terms = self._extract_terms_from_text(generated)
            # First half as primary, second half as related
            mid = len(all_terms) // 2 if len(all_terms) > 1 else len(all_terms)
            primary_terms = all_terms[:mid] if mid > 0 else all_terms
            related_terms = all_terms[mid:] if mid < len(all_terms) else []

        return primary_terms, related_terms

    def _extract_terms_from_text(self, text: str) -> list[str]:
        """Extract individual terms from text.

        Args:
            text: Text containing terms separated by commas or semicolons

        Returns:
            List of cleaned term strings
        """
        # Split by common delimiters
        import re

        terms = re.split(r"[,;]", text)

        # Clean and filter
        cleaned = []
        for term in terms:
            t = term.strip()
            # Remove quotes and trailing punctuation
            t = t.strip("\"'")
            if t and len(t) > 1:
                cleaned.append(t)

        return cleaned

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

        logger.info("batch_extraction_started", count=len(texts))

        results = []
        for text in texts:
            result = self.extract_terms(text)
            results.append(result)

        logger.info("batch_extraction_completed", count=len(results))
        return results
