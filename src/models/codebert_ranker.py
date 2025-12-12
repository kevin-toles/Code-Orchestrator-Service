"""
Code-Orchestrator-Service - CodeBERT Similarity Ranker

WBS 2.4: CodeBERT Ranker (Model Wrapper)
Generates embeddings and ranks terms by semantic relevance using CodeBERT model.

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


class RankedTerm(BaseModel):
    """A term with its relevance score.

    WBS 2.4.4: Output structure for ranked terms
    """

    term: str
    """The term string."""

    score: float
    """Relevance score (0.0 to 1.0)."""


class RankingResult(BaseModel):
    """Result of term ranking.

    WBS 2.4.4: Output structure for rank_terms()
    """

    ranked_terms: list[RankedTerm]
    """Terms sorted by relevance score descending."""


class CodeBERTRanker:
    """CodeBERT Ranker for embedding generation and term ranking.

    WBS 2.4: Generates 768-dim embeddings and ranks terms by cosine similarity.

    Pattern: Model Wrapper Pattern per HuggingFace integration
    - Ranker role: Scores and ranks candidate terms
    - Uses model from registry (Anti-Pattern #12 prevention)
    """

    def __init__(self, registry: ModelRegistryProtocol | None = None) -> None:
        """Initialize CodeBERT ranker.

        Args:
            registry: ModelRegistry instance (or FakeModelRegistry for testing)

        Raises:
            ModelNotReadyError: If codebert model not loaded in registry
        """
        self._registry = registry or ModelRegistry.get_registry()

        # Get model from registry
        model_tuple = self._registry.get_model("codebert")
        if model_tuple is None:
            raise ModelNotReadyError("CodeBERT model not loaded in registry")

        self._model, self._tokenizer = model_tuple
        logger.info("codebert_ranker_initialized")

    def get_embedding(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding vector for text.

        WBS 2.4.2: Returns 768-dimensional embedding.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of shape (768,) or (1, 768)
        """
        logger.debug("generating_embedding", text_length=len(text))

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
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # Convert to numpy
        embedding: npt.NDArray[np.floating[Any]] = embeddings.detach().numpy()

        return embedding

    def get_embeddings_batch(self, texts: list[str]) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding arrays
        """
        if not texts:
            return []

        logger.debug("batch_embedding_started", count=len(texts))

        # Tokenize all at once
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )

        # Get model output
        outputs = self._model(**inputs)

        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # Convert to list of numpy arrays
        result = [embeddings[i].detach().numpy() for i in range(len(texts))]

        logger.info("batch_embedding_complete", count=len(result))
        return result

    def calculate_similarity(self, term: str, query: str) -> float:
        """Calculate cosine similarity between term and query.

        WBS 2.4.3: Returns score between 0.0 and 1.0.

        Args:
            term: Term to compare
            query: Query text to compare against

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Get embeddings
        term_emb = self.get_embedding(term)
        query_emb = self.get_embedding(query)

        # Flatten to 1D if needed
        term_emb = term_emb.flatten()
        query_emb = query_emb.flatten()

        # Cosine similarity
        dot_product = np.dot(term_emb, query_emb)
        norm_term = np.linalg.norm(term_emb)
        norm_query = np.linalg.norm(query_emb)

        if norm_term == 0 or norm_query == 0:
            return 0.0

        similarity = dot_product / (norm_term * norm_query)

        # Clamp to [0, 1] range (cosine can be negative)
        similarity = max(0.0, min(1.0, float(similarity)))

        return similarity

    def rank_terms(
        self,
        terms: list[str],
        query: str,
    ) -> RankingResult:
        """Rank terms by relevance to query.

        WBS 2.4.4: Sort by relevance score descending.

        Args:
            terms: List of terms to rank
            query: Query text for comparison

        Returns:
            RankingResult with terms sorted by score descending
        """
        logger.debug("ranking_terms", term_count=len(terms))

        # Calculate similarity for each term
        scored_terms: list[RankedTerm] = []
        for term in terms:
            score = self.calculate_similarity(term, query)
            scored_terms.append(RankedTerm(term=term, score=score))

        # Sort by score descending
        scored_terms.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "ranking_complete",
            term_count=len(scored_terms),
            top_score=scored_terms[0].score if scored_terms else 0,
        )

        return RankingResult(ranked_terms=scored_terms)
