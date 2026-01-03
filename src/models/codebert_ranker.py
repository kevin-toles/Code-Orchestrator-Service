"""
Code-Orchestrator-Service - CodeBERT Term Ranker

WBS 2.4: Term Ranker (Model Wrapper)
Generates embeddings and ranks terms by semantic relevance using CodeBERT model.

Uses locally hosted microsoft/codebert-base for NL↔Code bimodal similarity.

Architecture Role: RANKER (STATE 3: RANKING)
- 768-dimensional embeddings for code/text
- Fast similarity scoring
- Well-established baseline for ranking

CodeBERT Capabilities:
- 768-dim embeddings (RoBERTa-based, bimodal NL↔Code)
- Pre-trained on code-text pairs
- Optimized for code search and ranking

Patterns Applied:
- Model Wrapper Pattern with Protocol typing (CODING_PATTERNS_ANALYSIS.md)
- Singleton model loading for efficiency
- Pydantic response models for structured output

Anti-Patterns Avoided:
- #12: New model per request (uses cached singleton)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from src.core.logging import get_logger

# Get logger
logger = get_logger(__name__)

# Local model path
_LOCAL_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "codebert"

# Singleton model instances
_tokenizer: Any | None = None
_model: Any | None = None


def _get_codebert() -> tuple[Any, Any]:
    """Get or create singleton CodeBERT model and tokenizer from local path."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_path = str(_LOCAL_MODEL_PATH)
        if not _LOCAL_MODEL_PATH.exists():
            # Fallback to HuggingFace if local not found
            model_path = "microsoft/codebert-base"
            logger.warning("local_codebert_not_found", path=str(_LOCAL_MODEL_PATH), using=model_path)
        else:
            logger.info("loading_codebert_from_local", path=model_path)

        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModel.from_pretrained(model_path)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()  # Set to evaluation mode
        logger.info("codebert_model_loaded", device=device)

    return _tokenizer, _model


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
    """CodeBERT-based Ranker for embedding generation and term ranking.

    WBS 2.4: Generates 768-dim embeddings and ranks terms by cosine similarity.
    Uses locally hosted microsoft/codebert-base.

    Architecture Role: RANKER (STATE 3: RANKING)
    - Computes 768-dim embeddings for terms and query
    - Calculates cosine similarity
    - Sorts terms by relevance score descending
    """

    def __init__(self) -> None:
        """Initialize CodeBERT ranker with local model."""
        self._tokenizer, self._model = _get_codebert()
        self._device = next(self._model.parameters()).device
        logger.info("codebert_ranker_initialized", device=str(self._device))

    def _encode(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Generate 768-dim embedding using CodeBERT with mean pooling.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of shape (768,)
        """
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self._device)

        # Get embeddings from CodeBERT
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pooling of last hidden states
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]

        return embedding

    def get_embedding(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding vector for text.

        WBS 2.4.2: Returns 768-dimensional embedding.

        Args:
            text: Input text to embed

        Returns:
            Numpy array of shape (768,)
        """
        logger.debug("generating_codebert_embedding", text_length=len(text))
        return self._encode(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding arrays (each 768-dim)
        """
        if not texts:
            return []

        logger.debug("codebert_batch_embedding_started", count=len(texts))
        result = [self._encode(text) for text in texts]
        logger.info("codebert_batch_embedding_complete", count=len(result))
        return result

    def calculate_similarity(self, term: str, query: str) -> float:
        """Calculate cosine similarity between term and query using CodeBERT.

        WBS 2.4.3: Returns score between 0.0 and 1.0.

        Args:
            term: Term to compare
            query: Query text to compare against

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Get embeddings
        term_emb = self._encode(term)
        query_emb = self._encode(query)

        similarity = cosine_similarity([term_emb], [query_emb])[0][0]
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, float(similarity)))

    def rank_terms(
        self,
        terms: list[str],
        query: str,
    ) -> RankingResult:
        """Rank terms by relevance to query using CodeBERT embeddings.

        WBS 2.4.4: Sort by relevance score descending.

        Architecture Role: STATE 3 RANKING
        - Computes 768-dim embeddings for all terms
        - Calculates cosine similarity to query
        - Returns sorted list by score

        Args:
            terms: List of terms to rank
            query: Query text for comparison

        Returns:
            RankingResult with terms sorted by score descending
        """
        logger.debug("codebert_ranking_terms", term_count=len(terms))

        # Calculate similarity for each term
        scored_terms: list[RankedTerm] = []
        for term in terms:
            score = self.calculate_similarity(term, query)
            scored_terms.append(RankedTerm(term=term, score=score))

        # Sort by score descending
        scored_terms.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "codebert_ranking_complete",
            term_count=len(scored_terms),
            top_score=scored_terms[0].score if scored_terms else 0,
        )

        return RankingResult(ranked_terms=scored_terms)
