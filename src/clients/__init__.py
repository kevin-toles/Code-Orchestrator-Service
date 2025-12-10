"""
WBS 4.1: Semantic Search Client

HTTP client for semantic-search-service with retry, timeout, and connection pooling.

Patterns Applied (from CODING_PATTERNS_ANALYSIS.md):
- Anti-Pattern #12: Connection pooling (reuse httpx.AsyncClient)
- Anti-Pattern #2.3: Retry with exponential backoff
- Repository Pattern: Protocol for duck typing
- Anti-Pattern #7/#13: Custom namespaced exceptions
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

# =============================================================================
# Custom Exceptions (Anti-Pattern #7/#13: Namespaced)
# =============================================================================


class SearchClientError(Exception):
    """Custom exception for SemanticSearchClient errors.

    Namespaced to avoid shadowing builtins like ConnectionError or TimeoutError.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SearchResult:
    """A search result from semantic-search-service."""

    book: str
    chapter: int
    score: float
    content: str = ""


# =============================================================================
# Protocol for Duck Typing (Repository Pattern)
# =============================================================================


class SemanticSearchClientProtocol(Protocol):
    """Protocol for SemanticSearchClient duck typing.

    Enables FakeSearchClient for testing without real HTTP calls.
    """

    async def search(
        self,
        keywords: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for documents matching keywords."""
        ...


# =============================================================================
# SemanticSearchClient Implementation
# =============================================================================


class SemanticSearchClient:
    """HTTP client for semantic-search-service.

    Uses connection pooling (Anti-Pattern #12) and exponential backoff retry
    (Anti-Pattern #2.3).

    Attributes:
        base_url: Base URL of semantic-search-service (e.g., http://localhost:8081)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the semantic search client.

        Args:
            base_url: Base URL of semantic-search-service
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Connection pooling: single client instance (Anti-Pattern #12)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def search(
        self,
        keywords: list[str],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search for documents matching keywords.

        Args:
            keywords: List of search keywords
            top_k: Maximum number of results

        Returns:
            List of SearchResult objects

        Raises:
            SearchClientError: On 4xx client errors
        """
        try:
            response = await self._execute_request(
                keywords=keywords,
                top_k=top_k,
            )
            return self._parse_results(response)
        except SearchClientError as e:
            # Return empty on 5xx, raise on 4xx
            if e.status_code and e.status_code >= 500:
                return []
            raise

    async def _execute_request(
        self,
        keywords: list[str],
        top_k: int,
    ) -> dict[str, Any]:
        """Execute HTTP request with retry logic.

        Uses exponential backoff (Anti-Pattern #2.3).

        Args:
            keywords: Search keywords
            top_k: Maximum results

        Returns:
            Response JSON dict

        Raises:
            SearchClientError: On HTTP errors after retries exhausted
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    "/v1/search",
                    json={"keywords": keywords, "top_k": top_k},
                )

                # Check for HTTP errors
                if response.status_code >= 400:
                    error_type = self._classify_error(response.status_code)
                    if error_type == "client_error":
                        raise SearchClientError(
                            f"Client error: {response.text}",
                            status_code=response.status_code,
                        )
                    # Server error - retry
                    raise SearchClientError(
                        f"Server error: {response.text}",
                        status_code=response.status_code,
                    )

                result: dict[str, Any] = response.json()
                return result

            except httpx.TimeoutException:
                last_error = TimeoutError("Connection timed out")
            except SearchClientError:
                raise
            except Exception as e:
                last_error = e

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        # Exhausted retries
        raise SearchClientError(
            f"Request failed after {self.max_retries} attempts: {last_error}",
            status_code=500,
        )

    def _classify_error(self, status_code: int) -> str:
        """Classify HTTP error type.

        Extracted to reduce cognitive complexity (Anti-Pattern #2.3).

        Args:
            status_code: HTTP status code

        Returns:
            Error type: 'client_error' or 'server_error'
        """
        if 400 <= status_code < 500:
            return "client_error"
        return "server_error"

    def _parse_results(self, response: dict[str, Any]) -> list[SearchResult]:
        """Parse response JSON into SearchResult objects.

        Args:
            response: Response dict from API

        Returns:
            List of SearchResult objects
        """
        results: list[SearchResult] = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    book=item.get("book", ""),
                    chapter=item.get("chapter", 0),
                    score=item.get("score", 0.0),
                    content=item.get("content", ""),
                )
            )
        return results

    async def close(self) -> None:
        """Close the HTTP client connection pool."""
        await self._client.aclose()


# =============================================================================
# FakeSearchClient for Testing
# =============================================================================


class FakeSearchClient:
    """Fake client for unit testing without real HTTP.

    Implements SemanticSearchClientProtocol for duck typing.
    """

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        """Initialize fake client with optional preset results.

        Args:
            results: Preset results to return from search()
        """
        self._results = results or []

    async def search(
        self,
        keywords: list[str],  # noqa: ARG002
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Return preset results.

        Args:
            keywords: Ignored in fake
            top_k: Limit results

        Returns:
            Preset results limited by top_k
        """
        await asyncio.sleep(0)  # Proper async behavior per SonarLint
        return self._results[:top_k]

    def set_results(self, results: list[SearchResult]) -> None:
        """Set results to return from search().

        Args:
            results: Results to return
        """
        self._results = results
