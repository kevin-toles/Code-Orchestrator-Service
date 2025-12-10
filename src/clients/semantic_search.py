"""
WBS 4.1: Semantic Search Client

HTTP client for semantic-search-service with retry, timeout, and connection pooling.
"""

from src.clients import (
    FakeSearchClient,
    SearchClientError,
    SearchResult,
    SemanticSearchClient,
    SemanticSearchClientProtocol,
)

__all__ = [
    "FakeSearchClient",
    "SearchClientError",
    "SearchResult",
    "SemanticSearchClient",
    "SemanticSearchClientProtocol",
]
