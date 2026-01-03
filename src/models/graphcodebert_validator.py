"""
Code-Orchestrator-Service - GraphCodeBERT Term Validator

WBS 2.3: Term Validator (Model Wrapper)
Validates and filters technical terms using GraphCodeBERT model.

Uses locally hosted microsoft/graphcodebert-base for semantic validation.

Architecture Role: VALIDATOR (STATE 2: VALIDATION)
- Understands code structure via data flow graphs
- Catches semantic mismatches
- Filters generic terms ("split", "data")
- Domain classification

GraphCodeBERT Capabilities:
- 768-dimensional embeddings (RoBERTa-based)
- Code structure awareness from pre-training
- Semantic validation via cosine similarity

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- Singleton model loading for efficiency
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached singleton)
- #12: Embedding cache to avoid recomputation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from transformers import AutoModel, AutoTokenizer

from src.core.logging import get_logger

# Get logger
logger = get_logger(__name__)

# Local model path
_LOCAL_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "graphcodebert"

# Singleton model instances
_tokenizer: Any | None = None
_model: Any | None = None


def _get_graphcodebert() -> tuple[Any, Any]:
    """Get or create singleton GraphCodeBERT model and tokenizer from local path."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_path = str(_LOCAL_MODEL_PATH)
        if not _LOCAL_MODEL_PATH.exists():
            # Fallback to HuggingFace if local not found
            model_path = "microsoft/graphcodebert-base"
            logger.warning("local_graphcodebert_not_found", path=str(_LOCAL_MODEL_PATH), using=model_path)
        else:
            logger.info("loading_graphcodebert_from_local", path=model_path)

        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModel.from_pretrained(model_path)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()  # Set to evaluation mode
        logger.info("graphcodebert_model_loaded", device=device)

    return _tokenizer, _model

# Validation thresholds
MIN_SEMANTIC_SIMILARITY: float = 0.15  # Minimum cosine similarity to query
MIN_TERM_LENGTH: int = 2  # Minimum characters for valid term

# Generic terms to filter out (pre-filter before model inference)
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

# Domain reference texts for embedding-based classification
_DOMAIN_REFERENCES: dict[str, str] = {
    "ai-ml": (
        "machine learning neural network deep learning transformer attention "
        "embedding vector RAG retrieval augmented generation LLM large language model "
        "tokenization chunking semantic search NLP natural language processing "
        "fine-tuning training inference prompt engineering"
    ),
    "systems": (
        "socket TCP UDP HTTP networking thread process memory CPU kernel "
        "filesystem database cache connection pool server client protocol "
        "packet buffer operating system distributed systems"
    ),
    "web": (
        "REST API endpoint HTTP request response JSON HTML CSS JavaScript "
        "frontend backend server client authentication authorization "
        "routing middleware session cookie"
    ),
    "data": (
        "database SQL query table schema index transaction ORM migration "
        "ETL pipeline data warehouse analytics aggregation"
    ),
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

    similarity_scores: dict[str, float] = {}
    """Semantic similarity scores for valid terms (term -> score)."""


class GraphCodeBERTValidator:
    """GraphCodeBERT-based term validator and filter.

    WBS 2.3: Validates terms using GraphCodeBERT embeddings (768-dim).
    Uses locally hosted microsoft/graphcodebert-base.

    Architecture Role: VALIDATOR (STATE 2: VALIDATION)
    - Filters generic terms ("split", "data")
    - Computes semantic similarity to query context
    - Domain classification via embedding space

    Uses actual model inference for:
    - 768-dim semantic embeddings for each term
    - Cosine similarity scoring between terms and query
    - Domain classification via reference embedding comparison
    """

    def __init__(self) -> None:
        """Initialize GraphCodeBERT validator with local model."""
        self._tokenizer, self._model = _get_graphcodebert()
        self._device = next(self._model.parameters()).device

        # Embedding cache (Anti-Pattern #12: avoid recomputation)
        self._embedding_cache: dict[str, npt.NDArray[np.floating[Any]]] = {}

        # Pre-compute domain reference embeddings
        self._domain_embeddings: dict[str, npt.NDArray[np.floating[Any]]] = {}

        logger.info("graphcodebert_validator_initialized", device=str(self._device))

    def _get_embedding(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Generate 768-dim embedding for text using GraphCodeBERT.

        Uses mean pooling of last hidden states.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector (768-dim for GraphCodeBERT)
        """
        # Check cache first (Anti-Pattern #12)
        cache_key = text[:200]  # Truncate for cache key
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self._device)

        # Get embeddings from GraphCodeBERT
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pooling of last hidden states
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]

        # Cache the result
        self._embedding_cache[cache_key] = embedding

        return embedding

    def _cosine_similarity(
        self,
        vec1: npt.NDArray[np.floating[Any]],
        vec2: npt.NDArray[np.floating[Any]],
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        similarity = sklearn_cosine([vec1], [vec2])[0][0]
        return max(0.0, float(similarity))

    def validate_terms(
        self,
        terms: list[str],
        original_query: str,
        domain: str,
        min_similarity: float = MIN_SEMANTIC_SIMILARITY,
    ) -> ValidationResult:
        """Validate terms using GraphCodeBERT 768-dim embeddings.

        WBS 2.3.2: Filters terms based on semantic relevance to query.

        Architecture Role: STATE 2 VALIDATION
        - Filters generic terms ("split", "data")
        - Computes similarity using GraphCodeBERT embeddings
        - Returns validation result with scores

        Validation Pipeline:
        1. Pre-filter generic terms (fast, no model inference)
        2. Compute query embedding once (768-dim)
        3. For each term, compute embedding and cosine similarity
        4. Filter terms below similarity threshold

        Args:
            terms: List of terms to validate
            original_query: Original query for semantic context
            domain: Target domain (ai-ml, systems, etc.) - used for logging
            min_similarity: Minimum cosine similarity threshold (default 0.15)

        Returns:
            ValidationResult with valid/rejected terms, reasons, and similarity scores
        """
        logger.debug(
            "validating_terms_with_embeddings",
            term_count=len(terms),
            domain=domain,
        )

        valid_terms: list[str] = []
        rejected_terms: list[str] = []
        rejection_reasons: dict[str, str] = {}
        similarity_scores: dict[str, float] = {}

        # Step 1: Pre-filter generic terms (no model inference needed)
        terms_to_check: list[str] = []
        for term in terms:
            term_lower = term.lower().strip()

            # Length check
            if len(term_lower) < MIN_TERM_LENGTH:
                rejected_terms.append(term)
                rejection_reasons[term] = "too_short"
                continue

            # Generic term check
            if term_lower in _GENERIC_TERMS:
                rejected_terms.append(term)
                rejection_reasons[term] = "too_generic"
                continue

            terms_to_check.append(term)

        # If no terms to check, return early
        if not terms_to_check:
            return ValidationResult(
                valid_terms=valid_terms,
                rejected_terms=rejected_terms,
                rejection_reasons=rejection_reasons,
                similarity_scores=similarity_scores,
            )

        # Step 2: Compute query embedding (once)
        query_embedding = self._get_embedding(original_query)

        # Step 3: Compute term embeddings and similarities
        for term in terms_to_check:
            term_embedding = self._get_embedding(term)
            similarity = self._cosine_similarity(term_embedding, query_embedding)

            # Step 4: Filter by similarity threshold
            if similarity >= min_similarity:
                valid_terms.append(term)
                similarity_scores[term] = round(similarity, 4)
            else:
                rejected_terms.append(term)
                rejection_reasons[term] = f"low_similarity:{similarity:.3f}"

        logger.info(
            "validation_complete",
            valid_count=len(valid_terms),
            rejected_count=len(rejected_terms),
            avg_similarity=round(
                sum(similarity_scores.values()) / len(similarity_scores), 3
            ) if similarity_scores else 0,
        )

        return ValidationResult(
            valid_terms=valid_terms,
            rejected_terms=rejected_terms,
            rejection_reasons=rejection_reasons,
            similarity_scores=similarity_scores,
        )

    def classify_domain(self, text: str) -> str:
        """Classify the domain of given text using GraphCodeBERT embeddings.

        WBS 2.3.3: Identifies domain via semantic similarity to reference texts.

        Uses cosine similarity between text embedding and pre-defined domain
        reference embeddings. Returns the domain with highest similarity.

        Args:
            text: Text to classify

        Returns:
            Domain string: 'ai-ml', 'systems', 'web', 'data', or 'general'
        """
        # Compute text embedding
        text_embedding = self._get_embedding(text)

        # Compute domain similarities
        domain_scores: dict[str, float] = {}
        for domain, reference_text in _DOMAIN_REFERENCES.items():
            # Get or compute domain reference embedding
            if domain not in self._domain_embeddings:
                self._domain_embeddings[domain] = self._get_embedding(reference_text)

            similarity = self._cosine_similarity(
                text_embedding,
                self._domain_embeddings[domain],
            )
            domain_scores[domain] = similarity

        # Find best matching domain
        best_domain = max(domain_scores, key=domain_scores.get)  # type: ignore[arg-type]
        best_score = domain_scores[best_domain]

        logger.debug(
            "domain_classified",
            domain=best_domain,
            score=round(best_score, 3),
            all_scores={k: round(v, 3) for k, v in domain_scores.items()},
        )

        # Return 'general' if no strong match
        if best_score < 0.3:
            return "general"

        return best_domain

    def expand_terms(
        self,
        terms: list[str],
        domain: str,
        max_expansions: int = 3,
        expansion_candidates: list[str] | None = None,
    ) -> list[str]:
        """Expand terms with semantically related terms using GraphCodeBERT.

        WBS 2.3.4: Finds semantically similar terms via embedding similarity.

        If expansion_candidates provided, finds nearest neighbors from that list.
        Otherwise, returns original terms (no expansion without candidates).

        Args:
            terms: List of terms to expand
            domain: Target domain for context (used in logging)
            max_expansions: Max related terms to add per input term
            expansion_candidates: Optional list of candidate terms to search

        Returns:
            Expanded list including original and related terms
        """
        logger.debug(
            "expanding_terms",
            term_count=len(terms),
            max_expansions=max_expansions,
            has_candidates=expansion_candidates is not None,
        )

        # Without candidates, return original terms
        if not expansion_candidates:
            logger.info("expansion_skipped_no_candidates")
            return terms

        expanded: list[str] = list(terms)  # Start with original terms

        # Compute embeddings for all candidates
        candidate_embeddings: list[tuple[str, npt.NDArray[np.floating[Any]]]] = []
        for candidate in expansion_candidates:
            if candidate not in terms:  # Don't include original terms
                candidate_embeddings.append(
                    (candidate, self._get_embedding(candidate))
                )

        # For each term, find nearest neighbors in candidates
        for term in terms:
            term_embedding = self._get_embedding(term)

            # Calculate similarities to all candidates
            similarities: list[tuple[str, float]] = []
            for candidate, cand_emb in candidate_embeddings:
                sim = self._cosine_similarity(term_embedding, cand_emb)
                similarities.append((candidate, sim))

            # Sort by similarity (descending) and take top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            for candidate, sim in similarities[:max_expansions]:
                if sim >= MIN_SEMANTIC_SIMILARITY and candidate not in expanded:
                    expanded.append(candidate)
                    logger.debug(
                        "term_expanded",
                        original=term,
                        expansion=candidate,
                        similarity=round(sim, 3),
                    )

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for t in expanded:
            if t not in seen:
                seen.add(t)
                result.append(t)

        logger.info(
            "expansion_complete",
            original_count=len(terms),
            expanded_count=len(result),
        )
        return result

    def get_term_embeddings(
        self,
        terms: list[str],
    ) -> dict[str, npt.NDArray[np.floating[Any]]]:
        """Get embeddings for multiple terms (utility method).

        Useful for downstream tasks like clustering or visualization.

        Args:
            terms: List of terms to embed

        Returns:
            Dictionary mapping term -> embedding vector
        """
        return {term: self._get_embedding(term) for term in terms}

    def batch_similarity(
        self,
        terms: list[str],
        query: str,
    ) -> dict[str, float]:
        """Calculate similarity scores for multiple terms against a query.

        More efficient than calling validate_terms when you just need scores.

        Args:
            terms: List of terms to score
            query: Query text for comparison

        Returns:
            Dictionary mapping term -> similarity score
        """
        query_embedding = self._get_embedding(query)
        return {
            term: self._cosine_similarity(self._get_embedding(term), query_embedding)
            for term in terms
        }
