"""
WBS B2.1-B2.2: Topic Discovery and Clustering Endpoints

POST /api/v1/topics - Discover topics from a corpus
POST /api/v1/cluster - Cluster documents and return topic assignments

Patterns Applied:
- FastAPI router (Anti-Pattern #9 compliance)
- Pydantic request/response models
- TopicClusterer from bertopic_clusterer.py
- Proper error responses (Anti-Pattern #7 compliance)
- Constants for magic values (S1192 compliance)

Anti-Patterns Avoided:
- S1172: Unused parameters (mark with underscore prefix)
- S3776: Cognitive complexity (keep methods < 15)
- S1192: Duplicated string literals (use constants)
"""

import time
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from src.core.logging import get_logger
from src.models.bertopic_clusterer import TopicClusterer

# =============================================================================
# Constants (S1192 compliance - no magic values)
# =============================================================================

DEFAULT_MIN_TOPIC_SIZE: int = 2
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE_LOWER_BOUND: int = 1

# =============================================================================
# Request/Response Models
# =============================================================================


class TopicsRequest(BaseModel):
    """Request body for topics endpoint.

    Schema per BERTOPIC_INTEGRATION_WBS.md B2.1:
    {
      "corpus": ["Chapter 1 text...", "Chapter 2 text..."],
      "min_topic_size": 2,
      "embedding_model": "all-MiniLM-L6-v2"
    }
    """

    corpus: Annotated[
        list[str],
        Field(
            min_length=1,
            description="List of documents to discover topics from",
        ),
    ]
    min_topic_size: int = Field(
        default=DEFAULT_MIN_TOPIC_SIZE,
        ge=MIN_TOPIC_SIZE_LOWER_BOUND,
        description="Minimum number of documents to form a topic",
    )
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Sentence transformer model for embeddings",
    )

    @field_validator("corpus")
    @classmethod
    def validate_corpus_not_empty_strings(cls, v: list[str]) -> list[str]:
        """Validate that no document in corpus is empty or whitespace-only."""
        for i, doc in enumerate(v):
            if not doc or doc.strip() == "":
                raise ValueError(f"Document at index {i} cannot be empty or whitespace")
        return v


class TopicItem(BaseModel):
    """A single topic in the response.

    Schema per BERTOPIC_INTEGRATION_WBS.md B2.1.
    """

    topic_id: int = Field(..., description="Topic identifier (-1 for outliers)")
    name: str = Field(..., description="Topic name/label")
    keywords: list[str] = Field(..., description="Top keywords for this topic")
    size: int = Field(..., ge=0, description="Number of documents in this topic")


class ModelInfo(BaseModel):
    """Model information in the response."""

    embedding_model: str = Field(..., description="Embedding model used")
    bertopic_version: str = Field(..., description="BERTopic version or 'kmeans-fallback'")


class TopicsResponse(BaseModel):
    """Response from topics endpoint.

    Schema per BERTOPIC_INTEGRATION_WBS.md B2.1:
    {
      "topics": [...],
      "topic_count": 15,
      "model_info": {...},
      "processing_time_ms": 1234.5
    }
    """

    topics: list[TopicItem] = Field(..., description="List of discovered topics")
    topic_count: int = Field(..., ge=0, description="Number of topics discovered")
    model_info: ModelInfo = Field(..., description="Model information")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["topics"])

logger = get_logger(__name__)


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/topics",
    response_model=TopicsResponse,
    status_code=status.HTTP_200_OK,
    summary="Discover topics from corpus",
    description="Uses BERTopic to discover topics from a corpus of documents.",
)
async def discover_topics(request: TopicsRequest) -> TopicsResponse:
    """Discover topics from a corpus of documents.

    WBS B2.1: POST /api/v1/topics

    Args:
        request: TopicsRequest with corpus and configuration

    Returns:
        TopicsResponse with discovered topics and metadata

    Raises:
        HTTPException: 500 if topic discovery fails
    """
    start_time = time.perf_counter()

    try:
        # Initialize clusterer with request parameters
        clusterer = TopicClusterer(
            embedding_model=request.embedding_model,
            min_topic_size=request.min_topic_size,
        )

        # Cluster the corpus
        results = clusterer.cluster(request.corpus)

        # Build response topics
        response_topics = _build_topic_items(results.topics)

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "topics_discovered",
            topic_count=len(results.topics),
            document_count=len(request.corpus),
            processing_time_ms=processing_time_ms,
        )

        return TopicsResponse(
            topics=response_topics,
            topic_count=len(results.topics),
            model_info=ModelInfo(
                embedding_model=results.model_info.get("embedding_model", request.embedding_model),
                bertopic_version=results.model_info.get("bertopic_version", "unknown"),
            ),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(
            "topics_discovery_failed",
            error=str(e),
            document_count=len(request.corpus),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Topic discovery failed: {e!s}",
        ) from e


def _build_topic_items(topics: list[Any]) -> list[TopicItem]:
    """Build TopicItem list from TopicInfo objects.

    Extracted to keep discover_topics() complexity low (S3776 compliance).

    Args:
        topics: List of TopicInfo dataclass instances

    Returns:
        List of TopicItem Pydantic models
    """
    return [
        TopicItem(
            topic_id=topic.topic_id,
            name=topic.name,
            keywords=topic.keywords,
            size=topic.size,
        )
        for topic in topics
    ]


# =============================================================================
# B2.2: Request/Response Models - Cluster Endpoint
# =============================================================================


class ChapterIndexItem(BaseModel):
    """Chapter metadata for clustering.

    Schema per BERTOPIC_INTEGRATION_WBS.md B2.2.
    """

    book: str = Field(..., min_length=1, description="Book filename/identifier")
    chapter: int = Field(..., ge=1, description="Chapter number")
    title: str = Field(..., min_length=1, description="Chapter title")


class ClusterRequest(BaseModel):
    """Request body for cluster endpoint.

    Schema per BERTOPIC_INTEGRATION_WBS.md B2.2:
    {
      "corpus": ["Chapter 1 text...", "Chapter 2 text..."],
      "chapter_index": [{"book": "...", "chapter": 1, "title": "..."}],
      "embeddings": null,
      "embedding_model": "all-MiniLM-L6-v2"
    }
    """

    corpus: Annotated[
        list[str],
        Field(
            min_length=1,
            description="List of documents to cluster",
        ),
    ]
    chapter_index: Annotated[
        list[ChapterIndexItem],
        Field(
            min_length=1,
            description="Metadata for each document (must match corpus length)",
        ),
    ]
    embeddings: list[list[float]] | None = Field(
        default=None,
        description="Optional precomputed embeddings (must match corpus length)",
    )
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        min_length=1,
        description="Sentence transformer model for embeddings",
    )

    @field_validator("corpus")
    @classmethod
    def validate_corpus_not_empty_strings(cls, v: list[str]) -> list[str]:
        """Validate that no document in corpus is empty or whitespace-only."""
        for i, doc in enumerate(v):
            if not doc or doc.strip() == "":
                raise ValueError(f"Document at index {i} cannot be empty or whitespace")
        return v

    @model_validator(mode="after")
    def validate_lengths_match(self) -> "ClusterRequest":
        """Validate that corpus and chapter_index have same length."""
        if len(self.corpus) != len(self.chapter_index):
            raise ValueError(
                f"corpus length ({len(self.corpus)}) must match "
                f"chapter_index length ({len(self.chapter_index)})"
            )
        if self.embeddings is not None and len(self.embeddings) != len(self.corpus):
            raise ValueError(
                f"embeddings length ({len(self.embeddings)}) must match "
                f"corpus length ({len(self.corpus)})"
            )
        return self


class ClusterAssignment(BaseModel):
    """Topic assignment for a single document.

    Schema per BERTOPIC_INTEGRATION_WBS.md B2.2.
    """

    book: str = Field(..., description="Book filename/identifier")
    chapter: int = Field(..., description="Chapter number")
    title: str = Field(..., description="Chapter title")
    topic_id: int = Field(..., description="Assigned topic ID (-1 for outliers)")
    topic_name: str = Field(..., description="Assigned topic name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assignment confidence")


class ClusterResponse(BaseModel):
    """Response from cluster endpoint.

    Schema per BERTOPIC_INTEGRATION_WBS.md B2.2:
    {
      "assignments": [...],
      "topics": [...],
      "topic_count": 15,
      "processing_time_ms": 1234.5
    }
    """

    assignments: list[ClusterAssignment] = Field(
        ..., description="Topic assignment for each document"
    )
    topics: list[TopicItem] = Field(..., description="List of discovered topics")
    topic_count: int = Field(..., ge=0, description="Number of topics discovered")
    processing_time_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )


# =============================================================================
# B2.2: Cluster Endpoint
# =============================================================================


@router.post(
    "/cluster",
    response_model=ClusterResponse,
    status_code=status.HTTP_200_OK,
    summary="Cluster documents and return topic assignments",
    description="Clusters documents using BERTopic and returns topic assignment per document.",
)
async def cluster_documents(request: ClusterRequest) -> ClusterResponse:
    """Cluster documents and return topic assignments.

    WBS B2.2: POST /api/v1/cluster

    Args:
        request: ClusterRequest with corpus, chapter_index, and configuration

    Returns:
        ClusterResponse with assignments, topics, and metadata

    Raises:
        HTTPException: 500 if clustering fails
    """
    start_time = time.perf_counter()

    try:
        # Initialize clusterer with request parameters
        clusterer = TopicClusterer(
            embedding_model=request.embedding_model,
            min_topic_size=DEFAULT_MIN_TOPIC_SIZE,
        )

        # Cluster the corpus (precomputed embeddings not yet supported by TopicClusterer)
        results = clusterer.cluster(request.corpus)

        # Build response assignments
        assignments = _build_cluster_assignments(
            request.chapter_index, results.assignments
        )

        # Build response topics
        response_topics = _build_topic_items(results.topics)

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "documents_clustered",
            topic_count=len(results.topics),
            document_count=len(request.corpus),
            processing_time_ms=processing_time_ms,
        )

        return ClusterResponse(
            assignments=assignments,
            topics=response_topics,
            topic_count=len(results.topics),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(
            "clustering_failed",
            error=str(e),
            document_count=len(request.corpus),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering failed: {e!s}",
        ) from e


def _build_cluster_assignments(
    chapter_index: list[ChapterIndexItem],
    topic_assignments: list[Any],
) -> list[ClusterAssignment]:
    """Build ClusterAssignment list from chapter index and topic assignments.

    Extracted to keep cluster_documents() complexity low (S3776 compliance).

    Args:
        chapter_index: List of ChapterIndexItem from request
        topic_assignments: List of TopicAssignment from clusterer

    Returns:
        List of ClusterAssignment Pydantic models
    """
    return [
        ClusterAssignment(
            book=chapter.book,
            chapter=chapter.chapter,
            title=chapter.title,
            topic_id=assignment.topic_id,
            topic_name=assignment.topic_name,
            confidence=assignment.confidence,
        )
        for chapter, assignment in zip(chapter_index, topic_assignments, strict=True)
    ]
