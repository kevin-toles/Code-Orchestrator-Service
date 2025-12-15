"""
WBS 5.2 M2.2-M2.4: Similarity API Endpoints

POST /api/v1/similarity - Compute cosine similarity between two texts
POST /api/v1/embeddings - Generate embeddings for batch of texts
POST /api/v1/similarity/batch - Compute similarity for multiple pairs (M2.3)
POST /api/v1/similar-chapters - Find top-k similar chapters (M2.4)

Patterns Applied:
- FastAPI router (Anti-Pattern #9 compliance)
- SBERTModelLoader singleton (Anti-Pattern #6, #12 compliance)
- Pydantic request/response models
- Proper error responses (Anti-Pattern #7 compliance)
- NumPy vectorization for batch processing (GUIDELINES Line 722)
"""

import time
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, HTTPException
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from src.models.sbert.model_loader import SBERTModelLoader, get_sbert_model

# =============================================================================
# Constants (S1192 compliance - no magic values)
# =============================================================================

DEFAULT_TOP_K: int = 5
DEFAULT_SIMILARITY_THRESHOLD: float = 0.0
MIN_THRESHOLD: float = 0.0
MAX_THRESHOLD: float = 1.0

# Method identifiers per llm-document-enhancer ARCHITECTURE
SIMILARITY_METHOD_SBERT: str = "sentence_transformers"
SIMILARITY_METHOD_TFIDF: str = "tfidf"

# =============================================================================
# Request/Response Models - Similarity
# =============================================================================


class SimilarityRequest(BaseModel):
    """Request body for similarity endpoint."""

    text1: Annotated[str, Field(min_length=1, description="First text to compare")]
    text2: Annotated[str, Field(min_length=1, description="Second text to compare")]


class SimilarityResponse(BaseModel):
    """Response from similarity endpoint."""

    score: float = Field(..., description="Cosine similarity score (-1 to 1)")
    model: str = Field(..., description="Model name used for computation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Request/Response Models - Embeddings
# =============================================================================


class EmbeddingsRequest(BaseModel):
    """Request body for embeddings endpoint."""

    texts: list[str] = Field(..., min_length=1, description="Texts to embed")

    @field_validator("texts")
    @classmethod
    def validate_texts_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that no text in the list is empty."""
        for i, text in enumerate(v):
            if not text or text.strip() == "":
                raise ValueError(f"Text at index {i} cannot be empty")
        return v


class EmbeddingsResponse(BaseModel):
    """Response from embeddings endpoint."""

    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model name used for computation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Request/Response Models - Batch Similarity (M2.3)
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.3.5-M2.3.6
# =============================================================================


class SimilarityPair(BaseModel):
    """A pair of texts to compare for similarity."""

    text1: Annotated[str, Field(min_length=1, description="First text")]
    text2: Annotated[str, Field(min_length=1, description="Second text")]


class BatchSimilarityRequest(BaseModel):
    """Request body for batch similarity endpoint.

    Per GUIDELINES Line 722: NumPy vectorization provides efficient batch processing.
    """

    pairs: list[SimilarityPair] = Field(
        ...,
        min_length=1,
        description="List of text pairs to compute similarity for",
    )


class BatchSimilarityResponse(BaseModel):
    """Response from batch similarity endpoint."""

    scores: list[float] = Field(..., description="Similarity scores for each pair")
    model: str = Field(..., description="Model name used for computation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Request/Response Models - Similar Chapters (M2.4)
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.4.1-M2.4.7
# Per TIER_RELATIONSHIP_DIAGRAM.md - Similar chapters with tier-aware relevance
# =============================================================================


class ChapterInput(BaseModel):
    """Input chapter to compare against.

    Per llm-document-enhancer ARCHITECTURE: Chapters have id, title, and content.
    """

    id: str = Field(..., min_length=1, description="Unique chapter identifier")
    title: str = Field(..., min_length=1, description="Chapter title")
    content: str = Field(..., min_length=1, description="Chapter content text")


class SimilarChapterResult(BaseModel):
    """A similar chapter result with similarity score.

    Per llm-document-enhancer ARCHITECTURE: Results include id, title, and score.
    """

    id: str = Field(..., description="Chapter identifier")
    title: str = Field(..., description="Chapter title")
    score: float = Field(..., description="Similarity score (0.0 to 1.0)")


class SimilarChaptersRequest(BaseModel):
    """Request body for similar-chapters endpoint.

    Per llm-document-enhancer ARCHITECTURE:
    - similarity_top_k: int = 10
    - similarity_threshold: float = 0.7

    Per engine defaults (for flexibility):
    - top_k: int = 5
    - threshold: float = 0.0
    """

    query: str = Field(..., min_length=1, description="Query text to find similar chapters for")
    chapters: list[ChapterInput] = Field(
        ...,
        min_length=1,
        description="List of chapters to compare against",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        description="Maximum number of similar chapters to return",
    )
    threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=MIN_THRESHOLD,
        le=MAX_THRESHOLD,
        description="Minimum similarity threshold (0.0 to 1.0)",
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold_range(cls, v: float) -> float:
        """Validate threshold is in valid range."""
        if v < MIN_THRESHOLD or v > MAX_THRESHOLD:
            raise ValueError(f"Threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}")
        return v


class SimilarChaptersResponse(BaseModel):
    """Response from similar-chapters endpoint.

    Per llm-document-enhancer ARCHITECTURE:
    - Method tracked: "sentence_transformers" or "tfidf"
    """

    chapters: list[SimilarChapterResult] = Field(
        ...,
        description="Similar chapters sorted by score (descending)",
    )
    method: str = Field(
        ...,
        description='Method used: "sentence_transformers" or "tfidf"',
    )
    model: str = Field(..., description="Model name used for computation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =============================================================================
# Router
# =============================================================================

similarity_router = APIRouter(prefix="/v1", tags=["similarity"])


@similarity_router.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """Compute cosine similarity between two texts using SBERT embeddings.

    Args:
        request: SimilarityRequest with text1 and text2

    Returns:
        SimilarityResponse with similarity score, model info, and timing

    Raises:
        HTTPException: If model fails to compute similarity
    """
    start_time = time.perf_counter()

    try:
        # Get singleton model loader
        model_loader = get_sbert_model()

        # Compute similarity directly (handles embedding internally)
        score = model_loader.compute_similarity(request.text1, request.text2)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return SimilarityResponse(
            score=float(score),
            model=model_loader.get_status()["model_name"],
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute similarity: {str(e)}",
        ) from e


@similarity_router.post("/embeddings", response_model=EmbeddingsResponse)
async def compute_embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    """Generate embeddings for a batch of texts using SBERT.

    Args:
        request: EmbeddingsRequest with list of texts

    Returns:
        EmbeddingsResponse with embedding vectors, model info, and timing

    Raises:
        HTTPException: If model fails to compute embeddings
    """
    start_time = time.perf_counter()

    try:
        # Get singleton model loader
        model_loader = get_sbert_model()

        # Compute embeddings for all texts
        embeddings = model_loader.compute_embeddings(request.texts)

        # Convert numpy arrays to lists for JSON serialization
        embeddings_list = [emb.tolist() for emb in embeddings]

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return EmbeddingsResponse(
            embeddings=embeddings_list,
            model=model_loader.get_status()["model_name"],
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute embeddings: {str(e)}",
        ) from e


# =============================================================================
# M2.3.6: Batch Similarity Endpoint
# Per SBERT_EXTRACTION_MIGRATION_WBS.md - "Process multiple pairs efficiently"
# Per GUIDELINES Line 722 - NumPy vectorization for batch processing
# =============================================================================


@similarity_router.post("/similarity/batch", response_model=BatchSimilarityResponse)
async def compute_batch_similarity(
    request: BatchSimilarityRequest,
) -> BatchSimilarityResponse:
    """Compute cosine similarity for multiple text pairs using vectorized operations.

    This endpoint is optimized for batch processing by:
    1. Computing embeddings for all unique texts at once
    2. Using vectorized cosine similarity computation

    Per GUIDELINES Line 722: "NumPy vectorized operations are well over an order
    of magnitude faster than Python's built-in functions for large-scale sampling."

    Args:
        request: BatchSimilarityRequest with list of text pairs

    Returns:
        BatchSimilarityResponse with scores for each pair

    Raises:
        HTTPException: If model fails to compute similarities
    """
    start_time = time.perf_counter()

    try:
        model_loader = get_sbert_model()

        # Collect all unique texts for efficient batch embedding
        # Using dict to preserve order and ensure uniqueness
        unique_texts: dict[str, int] = {}
        for pair in request.pairs:
            if pair.text1 not in unique_texts:
                unique_texts[pair.text1] = len(unique_texts)
            if pair.text2 not in unique_texts:
                unique_texts[pair.text2] = len(unique_texts)

        # Compute embeddings once for all unique texts (vectorized)
        text_list = list(unique_texts.keys())
        embeddings = model_loader.compute_embeddings(text_list)

        # Compute similarity scores using vectorized operations
        scores = _compute_pairwise_similarities(
            embeddings=embeddings,
            text_to_idx=unique_texts,
            pairs=request.pairs,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return BatchSimilarityResponse(
            scores=scores,
            model=model_loader.get_status()["model_name"],
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute batch similarity: {str(e)}",
        ) from e


def _compute_pairwise_similarities(
    embeddings: NDArray[np.floating[Any]],
    text_to_idx: dict[str, int],
    pairs: list[SimilarityPair],
) -> list[float]:
    """Compute cosine similarities for pairs using pre-computed embeddings.

    This helper function keeps the main endpoint under S3776 cognitive complexity limit.

    Args:
        embeddings: Pre-computed embeddings array (n_texts x embedding_dim)
        text_to_idx: Mapping from text to embedding index
        pairs: List of text pairs to compute similarity for

    Returns:
        List of similarity scores in the same order as input pairs
    """
    scores: list[float] = []

    for pair in pairs:
        idx1 = text_to_idx[pair.text1]
        idx2 = text_to_idx[pair.text2]

        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]

        # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
        # Embeddings are already L2-normalized, so dot product = cosine similarity
        similarity = float(np.dot(emb1, emb2))
        scores.append(similarity)

    return scores


# =============================================================================
# M2.4: Similar Chapters Endpoint
# Per SBERT_EXTRACTION_MIGRATION_WBS.md M2.4.1-M2.4.7
# Per llm-document-enhancer ARCHITECTURE: Chapter similarity with threshold/top_k
# =============================================================================


@similarity_router.post("/similar-chapters", response_model=SimilarChaptersResponse)
async def compute_similar_chapters(
    request: SimilarChaptersRequest,
) -> SimilarChaptersResponse:
    """Find chapters most similar to a query text.

    This endpoint computes semantic similarity between a query and a set of chapters,
    returning the top-k most similar chapters above the threshold.

    Per llm-document-enhancer ARCHITECTURE:
    - Supports configurable top_k (default: 5)
    - Supports configurable threshold (default: 0.0)
    - Returns method field ("sentence_transformers" or "tfidf")

    Args:
        request: SimilarChaptersRequest with query, chapters, top_k, and threshold

    Returns:
        SimilarChaptersResponse with similar chapters sorted by score (descending)

    Raises:
        HTTPException: If model fails to compute similarities
    """
    start_time = time.perf_counter()

    try:
        model_loader = get_sbert_model()

        # Find similar chapters using helper to keep cognitive complexity low
        similar_chapters = _find_similar_chapters(
            model_loader=model_loader,
            query=request.query,
            chapters=request.chapters,
            top_k=request.top_k,
            threshold=request.threshold,
        )

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return SimilarChaptersResponse(
            chapters=similar_chapters,
            method=SIMILARITY_METHOD_SBERT,
            model=model_loader.get_status()["model_name"],
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find similar chapters: {str(e)}",
        ) from e


def _find_similar_chapters(
    model_loader: SBERTModelLoader,
    query: str,
    chapters: list[ChapterInput],
    top_k: int,
    threshold: float,
) -> list[SimilarChapterResult]:
    """Find chapters similar to query using SBERT embeddings.

    This helper function keeps the main endpoint under S3776 cognitive complexity limit.
    Per GUIDELINES Line 722: Uses vectorized numpy operations for efficiency.

    Args:
        model_loader: The SBERT model loader instance
        query: Query text to find similar chapters for
        chapters: List of chapters to search
        top_k: Maximum number of results to return
        threshold: Minimum similarity threshold

    Returns:
        List of SimilarChapterResult sorted by score (descending)
    """
    # Compute query embedding
    query_embedding = model_loader.compute_embeddings([query])[0]

    # Compute chapter embeddings (vectorized batch computation)
    chapter_contents = [chapter.content for chapter in chapters]
    chapter_embeddings = model_loader.compute_embeddings(chapter_contents)

    # Compute cosine similarities using vectorized dot product
    # Embeddings are L2-normalized, so dot product = cosine similarity
    similarities = np.dot(chapter_embeddings, query_embedding)

    # Build results with threshold filtering
    results: list[tuple[float, ChapterInput]] = []
    for i, chapter in enumerate(chapters):
        score = float(similarities[i])
        if score >= threshold:
            results.append((score, chapter))

    # Sort by score descending and take top_k
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    # Convert to response model
    return [
        SimilarChapterResult(
            id=chapter.id,
            title=chapter.title,
            score=score,
        )
        for score, chapter in top_results
    ]
