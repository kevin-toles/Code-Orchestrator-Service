"""
Code-Orchestrator-Service - GraphCodeBERT Term Validator

WBS 2.3: GraphCodeBERT Validator (Model Wrapper)
Validates and filters technical terms using GraphCodeBERT model embeddings.

NOTE: These are HuggingFace model wrappers, NOT autonomous agents.
Autonomous agents (LangGraph workflows) live in the ai-agents service.

GraphCodeBERT Capabilities Used:
- Semantic embeddings for term-query relevance scoring
- Cosine similarity for term validation (threshold-based filtering)
- Domain classification via embedding space clustering

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- FakeModelRegistry for testing (no real HuggingFace model in unit tests)
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached model from registry)
- #12: Embedding cache to avoid recomputation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from src.models.registry import ModelRegistry
from src.core.exceptions import ModelNotReadyError
from src.core.logging import get_logger

if TYPE_CHECKING:
    from src.models.registry import ModelRegistryProtocol

# Get logger
logger = get_logger(__name__)

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
    """GraphCodeBERT model wrapper for term validation and filtering.

    WBS 2.3: Validates terms using GraphCodeBERT embeddings for semantic relevance.

    Uses actual model inference for:
    - Semantic similarity scoring between terms and query context
    - Embedding-based domain classification
    - Term expansion via nearest neighbors in embedding space

    NOTE: This is a HuggingFace model wrapper, NOT an autonomous agent.
    - Validator role: Filters and validates candidate terms
    - Uses model from registry (Anti-Pattern #12 prevention)
    - Caches embeddings to avoid recomputation (Anti-Pattern #12)
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

        # Embedding cache (Anti-Pattern #12: avoid recomputation)
        self._embedding_cache: dict[str, npt.NDArray[np.floating[Any]]] = {}

        # Pre-compute domain reference embeddings
        self._domain_embeddings: dict[str, npt.NDArray[np.floating[Any]]] = {}

        logger.info("graphcodebert_validator_initialized")

    def _get_embedding(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding for text using GraphCodeBERT.

        Uses mean pooling over the last hidden state.

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
        )

        # Get model output
        outputs = self._model(**inputs)

        # Mean pooling over sequence dimension
        # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim=768)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        # Convert to numpy
        embedding_np: npt.NDArray[np.floating[Any]] = embedding.detach().numpy()

        # Cache the result
        self._embedding_cache[cache_key] = embedding_np

        return embedding_np

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
            Cosine similarity score (-1 to 1, normalized to 0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(dot_product / (norm1 * norm2))

        # Normalize to 0-1 range (cosine can be negative)
        return max(0.0, similarity)

    def validate_terms(
        self,
        terms: list[str],
        original_query: str,
        domain: str,
        min_similarity: float = MIN_SEMANTIC_SIMILARITY,
    ) -> ValidationResult:
        """Validate terms using GraphCodeBERT semantic embeddings.

        WBS 2.3.2: Filters terms based on semantic relevance to query.

        Validation Pipeline:
        1. Pre-filter generic terms (fast, no model inference)
        2. Compute query embedding once
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
