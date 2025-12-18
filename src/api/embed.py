"""
Embedding API Endpoints

WBS: EEP-1.5.7 - API Endpoints
AC-1.5.7.1: /api/v1/embed endpoint
AC-1.5.7.2: /api/v1/embed/batch endpoint
AC-1.5.7.3: Response includes all three embeddings + fused

FastAPI router for multi-modal embedding endpoints.

Anti-Patterns Avoided:
- S3776: Helper functions for cognitive complexity < 15
- S6903: No exception shadowing
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.models.embedding.config import (
    DIM_BGE,
    DIM_FUSION_OUTPUT,
    DIM_INSTRUCTOR,
    DIM_UNIXCODER,
    FusionModelConfig,
)

logger = logging.getLogger(__name__)

# Router for embedding endpoints
router = APIRouter(prefix="/api/v1", tags=["embeddings"])


# Field description constants (SonarQube S1192 compliance)
DESC_INSTRUCTION = "Instruction for Instructor model"
DESC_DOMAIN = "Domain for instruction lookup"
DESC_INCLUDE_INDIVIDUAL = "Include individual model embeddings"


# Request/Response Models
class EmbedRequest(BaseModel):
    """Single embedding request."""

    text: str = Field(..., description="Text to embed")
    instruction: str | None = Field(None, description=DESC_INSTRUCTION)
    domain: str | None = Field(None, description=DESC_DOMAIN)
    include_individual: bool = Field(True, description=DESC_INCLUDE_INDIVIDUAL)


class EmbedBatchRequest(BaseModel):
    """Batch embedding request.

    AC-1.5.7.2: /api/v1/embed/batch endpoint
    """

    texts: list[str] = Field(..., description="Texts to embed")
    instruction: str | None = Field(None, description=DESC_INSTRUCTION)
    domain: str | None = Field(None, description=DESC_DOMAIN)
    include_individual: bool = Field(True, description=DESC_INCLUDE_INDIVIDUAL)


class EmbeddingResponse(BaseModel):
    """Single embedding response.

    AC-1.5.7.3: Response includes all three embeddings + fused
    """

    fused: list[float] = Field(..., description=f"Fused embedding [{DIM_FUSION_OUTPUT}]")
    embedding: list[float] | None = Field(None, description="Alias for fused embedding")
    bge: list[float] | None = Field(None, description=f"BGE embedding [{DIM_BGE}]")
    unixcoder: list[float] | None = Field(None, description=f"UniXcoder embedding [{DIM_UNIXCODER}]")
    instructor: list[float] | None = Field(None, description=f"Instructor embedding [{DIM_INSTRUCTOR}]")
    model_versions: dict[str, str] = Field(default_factory=dict, description="Model versions used")


class BatchEmbeddingResponse(BaseModel):
    """Batch embedding response."""

    embeddings: list[EmbeddingResponse] = Field(..., description="Embeddings for each text")
    count: int = Field(..., description="Number of embeddings")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models: dict[str, Any]


# -----------------------------------------------------------------------------
# AC-1.5.7.3: Similarity Endpoint Models
# -----------------------------------------------------------------------------


class SimilarityItemRequest(BaseModel):
    """Single item for similarity computation."""

    text: str = Field(..., description="Text content")
    instruction: str | None = Field(None, description=DESC_INSTRUCTION)
    domain: str | None = Field(None, description=DESC_DOMAIN)


class SimilarityRequest(BaseModel):
    """Request for unified similarity computation.

    AC-1.5.7.3: POST /api/v1/similarity/unified endpoint
    """

    source: SimilarityItemRequest = Field(..., description="Source item to compare")
    targets: list[SimilarityItemRequest] = Field(..., description="Target items to compare against")


class SimilarityResult(BaseModel):
    """Single similarity result with modality scores."""

    score: float = Field(..., description="Unified (fused) similarity score")
    bge_score: float = Field(..., description="BGE text similarity score")
    unixcoder_score: float = Field(..., description="UniXcoder code similarity score")
    instructor_score: float = Field(..., description="Instructor concept similarity score")
    target_index: int = Field(..., description="Index of target in request")


class SimilarityResponse(BaseModel):
    """Response with similarity scores.

    AC-1.5.7.3: Returns unified similarity with individual modality contributions
    """

    similarities: list[SimilarityResult] = Field(..., description="Similarity results for each target")
    source_embedding_dim: int = Field(..., description="Dimension of source embedding")


# Global embedder instances (lazy loaded)
_embedders: dict[str, Any] = {}
_fusion_layer: Any = None
_config: FusionModelConfig | None = None


def _get_config() -> FusionModelConfig:
    """Get or create configuration."""
    global _config
    if _config is None:
        _config = FusionModelConfig()
    return _config


def _get_embedders() -> dict[str, Any]:
    """Lazy load embedders."""
    global _embedders

    if not _embedders:
        from src.models.embedding.bge_embedder import BGEEmbedder
        from src.models.embedding.instructor_embedder import InstructorEmbedder
        from src.models.embedding.unixcoder_embedder import UniXcoderEmbedder

        config = _get_config()

        _embedders["bge"] = BGEEmbedder(model_name=config.bge_model_id)
        _embedders["unixcoder"] = UniXcoderEmbedder(model_name=config.unixcoder_model_id)
        _embedders["instructor"] = InstructorEmbedder(model_name=config.instructor_model_id)

    return _embedders


def _get_fusion_layer() -> Any:
    """Lazy load fusion layer."""
    global _fusion_layer

    if _fusion_layer is None:
        from src.models.embedding.fusion import FusionLayer

        config = _get_config()
        _fusion_layer = FusionLayer(output_dim=config.fusion_output_dim)

    return _fusion_layer


def _compute_fused_embedding(
    bge_emb: np.ndarray,
    unixcoder_emb: np.ndarray,
    instructor_emb: np.ndarray,
) -> np.ndarray:
    """Compute fused embedding from three embeddings.

    Args:
        bge_emb: BGE embedding
        unixcoder_emb: UniXcoder embedding
        instructor_emb: Instructor embedding

    Returns:
        Fused embedding
    """
    import torch

    fusion = _get_fusion_layer()
    fusion.eval()

    with torch.no_grad():
        bge_t = torch.tensor(bge_emb, dtype=torch.float32).unsqueeze(0)
        unixcoder_t = torch.tensor(unixcoder_emb, dtype=torch.float32).unsqueeze(0)
        instructor_t = torch.tensor(instructor_emb, dtype=torch.float32).unsqueeze(0)

        fused = fusion(bge_t, unixcoder_t, instructor_t)

    return fused.squeeze(0).numpy()


@router.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbedRequest) -> EmbeddingResponse:
    """Generate embeddings for a single text.

    AC-1.5.7.1: /api/v1/embed endpoint
    """
    try:
        embedders = _get_embedders()
        config = _get_config()

        # Generate embeddings from each model
        bge_emb = embedders["bge"].embed(request.text)
        unixcoder_emb = embedders["unixcoder"].embed(request.text)
        instructor_emb = embedders["instructor"].embed(
            request.text,
            instruction=request.instruction,
            domain=request.domain,
        )

        # Compute fused embedding
        fused_emb = _compute_fused_embedding(bge_emb, unixcoder_emb, instructor_emb)
        fused_list = fused_emb.tolist()

        # Build response
        response = EmbeddingResponse(
            fused=fused_list,
            embedding=fused_list,  # Alias for compatibility
            model_versions={
                "bge": config.bge_model_id,
                "unixcoder": config.unixcoder_model_id,
                "instructor": config.instructor_model_id,
            },
        )

        # Include individual embeddings if requested
        if request.include_individual:
            response.bge = bge_emb.tolist()
            response.unixcoder = unixcoder_emb.tolist()
            response.instructor = instructor_emb.tolist()

        return response

    except Exception as e:
        logger.exception("Error generating embedding")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}",
        ) from e


@router.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_batch(request: EmbedBatchRequest) -> BatchEmbeddingResponse:
    """Generate embeddings for multiple texts.

    AC-1.5.7.2: /api/v1/embed/batch endpoint
    """
    try:
        embedders = _get_embedders()
        config = _get_config()

        # Generate batch embeddings
        bge_embs = embedders["bge"].batch_embed(request.texts)
        unixcoder_embs = embedders["unixcoder"].batch_embed(request.texts)
        instructor_embs = embedders["instructor"].batch_embed(
            request.texts,
            instruction=request.instruction,
            domain=request.domain,
        )

        # Build responses
        embeddings = []
        for idx in range(len(request.texts)):
            fused_emb = _compute_fused_embedding(
                bge_embs[idx],
                unixcoder_embs[idx],
                instructor_embs[idx],
            )

            response = EmbeddingResponse(
                fused=fused_emb.tolist(),
                model_versions={
                    "bge": config.bge_model_id,
                    "unixcoder": config.unixcoder_model_id,
                    "instructor": config.instructor_model_id,
                },
            )

            if request.include_individual:
                response.bge = bge_embs[idx].tolist()
                response.unixcoder = unixcoder_embs[idx].tolist()
                response.instructor = instructor_embs[idx].tolist()

            embeddings.append(response)

        return BatchEmbeddingResponse(
            embeddings=embeddings,
            count=len(embeddings),
        )

    except Exception as e:
        logger.exception("Error generating batch embeddings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding generation failed: {str(e)}",
        ) from e


@router.get("/embed/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check health of embedding models."""
    try:
        embedders = _get_embedders()

        models_health = {}
        for name, embedder in embedders.items():
            models_health[name] = embedder.check_health()

        # Check fusion layer
        fusion = _get_fusion_layer()
        models_health["fusion"] = {
            "status": "healthy",
            "output_dim": fusion.output_dim,
        }

        all_healthy = all(
            m.get("status") == "healthy"
            for m in models_health.values()
        )

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            models=models_health,
        )

    except Exception as e:
        logger.exception("Health check failed")
        return HealthResponse(
            status="unhealthy",
            models={"error": str(e)},
        )


# -----------------------------------------------------------------------------
# AC-1.5.7.3: Similarity Endpoint
# -----------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def _compute_similarity_result(
    source_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
    target_index: int,
) -> SimilarityResult:
    """Compute similarity result between source and target.

    Args:
        source_embeddings: Dict with 'fused', 'bge', 'unixcoder', 'instructor' keys
        target_embeddings: Dict with same keys
        target_index: Index of target in original request

    Returns:
        SimilarityResult with all modality scores
    """
    return SimilarityResult(
        score=_cosine_similarity(source_embeddings["fused"], target_embeddings["fused"]),
        bge_score=_cosine_similarity(source_embeddings["bge"], target_embeddings["bge"]),
        unixcoder_score=_cosine_similarity(source_embeddings["unixcoder"], target_embeddings["unixcoder"]),
        instructor_score=_cosine_similarity(source_embeddings["instructor"], target_embeddings["instructor"]),
        target_index=target_index,
    )


def _embed_item(item: SimilarityItemRequest, embedders: dict[str, Any]) -> dict[str, np.ndarray]:
    """Generate all embeddings for a single item.

    Args:
        item: Request item with text
        embedders: Dict of embedder instances

    Returns:
        Dict with 'bge', 'unixcoder', 'instructor', 'fused' embeddings
    """
    bge_emb = embedders["bge"].embed(item.text)
    unixcoder_emb = embedders["unixcoder"].embed(item.text)
    instructor_emb = embedders["instructor"].embed(
        item.text,
        instruction=item.instruction,
        domain=item.domain,
    )
    fused_emb = _compute_fused_embedding(bge_emb, unixcoder_emb, instructor_emb)

    return {
        "bge": bge_emb,
        "unixcoder": unixcoder_emb,
        "instructor": instructor_emb,
        "fused": fused_emb,
    }


@router.post("/similarity/unified", response_model=SimilarityResponse)
async def compute_unified_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """Compute unified similarity between source and targets.

    AC-1.5.7.3: POST /api/v1/similarity/unified endpoint

    Returns similarity scores using the fused multi-modal embedding,
    along with individual modality similarity scores (BGE, UniXcoder, Instructor).
    """
    try:
        embedders = _get_embedders()

        # Embed source
        source_embeddings = _embed_item(request.source, embedders)

        # Compute similarity for each target
        similarities = []
        for idx, target in enumerate(request.targets):
            target_embeddings = _embed_item(target, embedders)
            result = _compute_similarity_result(source_embeddings, target_embeddings, idx)
            similarities.append(result)

        return SimilarityResponse(
            similarities=similarities,
            source_embedding_dim=len(source_embeddings["fused"]),
        )

    except Exception as e:
        logger.exception("Error computing similarity")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity computation failed: {str(e)}",
        ) from e
