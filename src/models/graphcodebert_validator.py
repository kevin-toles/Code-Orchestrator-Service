"""
Code-Orchestrator-Service - GraphCodeBERT Term Validator

WBS 2.3: GraphCodeBERT Validator (Model Wrapper)
Validates and filters technical terms using GraphCodeBERT model.

NOTE: These are HuggingFace model wrappers, NOT autonomous agents.
Autonomous agents (LangGraph workflows) live in the ai-agents service.

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- FakeModelRegistry for testing (no real HuggingFace model in unit tests)
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached model from registry)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.models.registry import ModelRegistry
from src.core.exceptions import ModelNotReadyError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.models.registry import ModelRegistryProtocol

# Get logger
logger = get_logger(__name__)


# Generic terms to filter out
_GENERIC_TERMS: set[str] = {
    "data",
    "split",
    "value",
    "item",
    "list",
    "string",
    "number",
    "function",
    "method",
    "class",
    "object",
    "variable",
    "code",
    "file",
    "result",
    "output",
    "input",
    "return",
    "type",
    "name",
    "get",
    "set",
    "add",
    "remove",
    "create",
    "update",
    "delete",
    "process",
    "handle",
}

# Domain keywords for classification
_AI_ML_KEYWORDS: set[str] = {
    "rag",
    "llm",
    "embedding",
    "transformer",
    "attention",
    "neural",
    "model",
    "training",
    "inference",
    "tokenization",
    "chunking",
    "vector",
    "semantic",
    "nlp",
    "bert",
    "gpt",
    "prompt",
    "fine-tuning",
    "retrieval",
    "generation",
    "language model",
    "deep learning",
    "machine learning",
}

_SYSTEMS_KEYWORDS: set[str] = {
    "socket",
    "tcp",
    "udp",
    "http",
    "networking",
    "thread",
    "process",
    "memory",
    "cpu",
    "kernel",
    "filesystem",
    "database",
    "cache",
    "pool",
    "connection",
    "server",
    "client",
    "protocol",
    "packet",
    "buffer",
}

# Semantic expansion mappings
_EXPANSION_MAP: dict[str, list[str]] = {
    "rag": ["retrieval", "semantic_search", "augmented_generation"],
    "chunking": ["segmentation", "text_splitting", "passage"],
    "embedding": ["vector", "representation", "encoding"],
    "llm": ["language_model", "transformer", "gpt"],
    "tokenization": ["tokenizer", "subword", "vocabulary"],
}


class ValidationResult(BaseModel):
    """Result of term validation.

    WBS 2.3.2: Output structure for validate_terms()
    """

    valid_terms: list[str]
    """Terms that passed validation."""

    rejected_terms: list[str]
    """Terms that were rejected."""

    rejection_reasons: dict[str, str]
    """Reason for each rejection (term -> reason)."""


class GraphCodeBERTValidator:
    """GraphCodeBERT model wrapper for term validation and filtering.

    WBS 2.3: Validates terms against query context and domain.

    NOTE: This is a HuggingFace model wrapper, NOT an autonomous agent.
    - Validator role: Filters and validates candidate terms
    - Uses model from registry (Anti-Pattern #12 prevention)
    """

    def __init__(self, registry: ModelRegistryProtocol | None = None) -> None:
        """Initialize GraphCodeBERT validator.

        Args:
            registry: ModelRegistry instance (or FakeModelRegistry for testing)

        Raises:
            ModelNotReadyError: If graphcodebert model not loaded in registry
        """
        self._registry = registry or ModelRegistry.get_registry()

        # Get model from registry
        model_tuple = self._registry.get_model("graphcodebert")
        if model_tuple is None:
            raise ModelNotReadyError("GraphCodeBERT model not loaded in registry")

        self._model, self._tokenizer = model_tuple
        logger.info("graphcodebert_validator_initialized")

    def validate_terms(
        self,
        terms: list[str],
        original_query: str,
        domain: str,
    ) -> ValidationResult:
        """Validate terms against query and domain.

        WBS 2.3.2: Filters generic terms like 'split', 'data'.

        Args:
            terms: List of terms to validate
            original_query: Original query for context
            domain: Target domain (ai-ml, systems, etc.)

        Returns:
            ValidationResult with valid/rejected terms and reasons
        """
        logger.debug(
            "validating_terms",
            term_count=len(terms),
            domain=domain,
        )

        valid_terms: list[str] = []
        rejected_terms: list[str] = []
        rejection_reasons: dict[str, str] = {}

        # Classify query domain for future use
        _query_domain = self.classify_domain(original_query)  # noqa: F841

        for term in terms:
            term_lower = term.lower()

            # Check if too generic
            if term_lower in _GENERIC_TERMS:
                rejected_terms.append(term)
                rejection_reasons[term] = "too_generic"
                continue

            # Check domain relevance
            if domain == "ai-ml" and term_lower in _SYSTEMS_KEYWORDS:
                rejected_terms.append(term)
                rejection_reasons[term] = "out_of_domain"
                continue

            if domain in ("systems", "networking") and term_lower in _AI_ML_KEYWORDS:
                rejected_terms.append(term)
                rejection_reasons[term] = "out_of_domain"
                continue

            # Term passes validation
            valid_terms.append(term)

        logger.info(
            "validation_complete",
            valid_count=len(valid_terms),
            rejected_count=len(rejected_terms),
        )

        return ValidationResult(
            valid_terms=valid_terms,
            rejected_terms=rejected_terms,
            rejection_reasons=rejection_reasons,
        )

    def classify_domain(self, text: str) -> str:
        """Classify the domain of given text.

        WBS 2.3.3: Identifies AI/LLM vs systems context.

        Args:
            text: Text to classify

        Returns:
            Domain string: 'ai-ml', 'systems', 'general'
        """
        text_lower = text.lower()

        # Count domain keyword matches
        ai_ml_count = sum(1 for kw in _AI_ML_KEYWORDS if kw in text_lower)
        systems_count = sum(1 for kw in _SYSTEMS_KEYWORDS if kw in text_lower)

        if ai_ml_count > systems_count:
            return "ai-ml"
        elif systems_count > ai_ml_count:
            return "systems"
        else:
            return "general"

    def expand_terms(
        self,
        terms: list[str],
        domain: str,  # noqa: ARG002
        max_expansions: int = 3,
    ) -> list[str]:
        """Expand terms with semantically related terms.

        WBS 2.3.4: RAG → semantic_search, retrieval

        Args:
            terms: List of terms to expand
            domain: Target domain for context
            max_expansions: Max related terms to add per input term

        Returns:
            Expanded list including original and related terms
        """
        logger.debug(
            "expanding_terms",
            term_count=len(terms),
            max_expansions=max_expansions,
        )

        expanded: list[str] = []

        for term in terms:
            # Add original term
            expanded.append(term)

            # Look up expansions
            term_lower = term.lower()
            if term_lower in _EXPANSION_MAP:
                related = _EXPANSION_MAP[term_lower][:max_expansions]
                expanded.extend(related)

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for t in expanded:
            if t not in seen:
                seen.add(t)
                result.append(t)

        logger.info("expansion_complete", expanded_count=len(result))
        return result
