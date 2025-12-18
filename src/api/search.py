"""
WBS 4.3: Full Search Endpoint

POST /v1/search - Full pipeline: extract → search → curate

Patterns Applied:
- FastAPI router pattern
- Pydantic request/response models
- Dependency injection for services
"""

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.extract import ExtractOptions, ExtractRequest, extract_terms
from src.curation.curator import ResultCurator
from src.curation.models import SearchResult as CurationSearchResult

# =============================================================================
# Request/Response Models
# =============================================================================


class SearchOptions(BaseModel):
    """Options for search endpoint."""

    top_k: int = Field(default=10, ge=1, le=100, description="Maximum results")


class SearchRequest(BaseModel):
    """Request body for search endpoint."""

    query: str = Field(..., min_length=1, description="Search query")
    domain: str | None = Field(default=None, description="Domain context (optional)")
    options: SearchOptions | None = None


class SearchResultItem(BaseModel):
    """A search result item."""

    book: str
    chapter: int | None = None
    relevance_score: float
    content: str | None = None


class PipelineMetadata(BaseModel):
    """Pipeline execution metadata."""

    stages_completed: int


class SearchMetadata(BaseModel):
    """Metadata about the search execution."""

    processing_time_ms: float
    pipeline: PipelineMetadata
    total_results: int = 0


class SearchResponse(BaseModel):
    """Response from search endpoint."""

    results: list[SearchResultItem]
    metadata: SearchMetadata


# =============================================================================
# Router
# =============================================================================

search_router = APIRouter(prefix="/v1", tags=["search"])


@search_router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Execute full search pipeline: extract → search → curate.

    The pipeline has 4 stages:
    1. Extract: Generate search terms from query
    2. Search: Query semantic-search-service (simulated)
    3. Curate: Filter, dedupe, rank results
    4. Format: Prepare response

    Args:
        request: SearchRequest with query, domain, and options

    Returns:
        SearchResponse with curated results and metadata
    """
    start_time = time.perf_counter()
    stages_completed = 0

    try:
        # Get options
        options = request.options or SearchOptions()

        # Stage 1: Extract terms from query
        extract_request = ExtractRequest(
            query=request.query,
            domain=request.domain,
            options=ExtractOptions(max_terms=10),
        )
        extract_response = await extract_terms(extract_request)
        stages_completed = 1

        # Get extracted search terms
        search_terms = [t.term for t in extract_response.search_terms]

        # Stage 2: Search semantic-search-service (simulated)
        # In production, this would call the actual semantic-search-service
        raw_results = await _simulate_search(search_terms, options.top_k)
        stages_completed = 2

        # Stage 3: Curate results (filter, dedupe, rank)
        curator = ResultCurator()
        curated = curator.curate(
            results=raw_results,
            query=request.query,
            domain=request.domain,
        )
        stages_completed = 3

        # Stage 4: Format response
        result_items = [
            SearchResultItem(
                book=r.book,
                chapter=r.chapter,
                relevance_score=r.relevance_score,
                content=r.content,
            )
            for r in curated[:options.top_k]
        ]
        stages_completed = 4

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            results=result_items,
            metadata=SearchMetadata(
                processing_time_ms=elapsed_ms,
                pipeline=PipelineMetadata(stages_completed=stages_completed),
                total_results=len(result_items),
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Helper Functions (will be replaced with real semantic-search integration)
# =============================================================================


async def _simulate_search(
    _search_terms: list[str],
    top_k: int,
) -> list[CurationSearchResult]:
    """Simulate search results from semantic-search-service.

    TODO: Replace with real SemanticSearchClient integration.

    Args:
        _search_terms: Extracted search terms (prefixed with _ as placeholder)
        top_k: Maximum results

    Returns:
        Simulated search results
    """
    # Simulate results based on search terms
    simulated_results: list[CurationSearchResult] = []

    # AI/ML book results (high relevance)
    ai_books = [
        ("AI Engineering", 5, 0.91, "LLM document chunking strategies"),
        ("Building LLM Apps", 8, 0.88, "RAG pipeline implementation"),
        ("AI Engineering", 3, 0.85, "embedding generation techniques"),
        ("Building LLM Powered Applications", 12, 0.82, "semantic search patterns"),
        ("Architecture Patterns with Python", 7, 0.78, "domain-driven design"),
    ]

    # Systems book results (should be filtered for ai-ml domain)
    systems_books = [
        ("C++ Concurrency", 3, 0.45, "memory chunk allocation"),
        ("C++ Concurrency in Action", 5, 0.42, "thread-safe chunking"),
    ]

    # Add AI/ML results
    for book, chapter, score, content in ai_books:
        if len(simulated_results) >= top_k:
            break
        simulated_results.append(
            CurationSearchResult(
                book=book,
                chapter=chapter,
                score=score,
                content=content,
            )
        )

    # Add some systems results (will be filtered by curator)
    for book, chapter, score, content in systems_books:
        if len(simulated_results) >= top_k + 2:  # Add a few extras to test filtering
            break
        simulated_results.append(
            CurationSearchResult(
                book=book,
                chapter=chapter,
                score=score,
                content=content,
            )
        )

    return simulated_results


# =============================================================================
# Helper to convert between result types
# =============================================================================


def _search_result_to_item(result: dict[str, Any]) -> SearchResultItem:
    """Convert dict to SearchResultItem."""
    return SearchResultItem(
        book=result.get("book", ""),
        chapter=result.get("chapter"),
        relevance_score=result.get("score", 0.0),
        content=result.get("content"),
    )
