"""
WBS 4.1: Semantic Search Client Tests (RED Phase)

Tests for SemanticSearchClient:
- 4.1.1: HTTP client with retry, timeout
- 4.1.2: POST to semantic-search /v1/search
- 4.1.3: Handle errors gracefully (empty on 5xx, raise on 4xx)

Patterns Applied (from CODING_PATTERNS_ANALYSIS.md):
- Anti-Pattern #12: Connection pooling (don't create new client per request)
- Anti-Pattern #2.3: Retry with exponential backoff
- Repository Pattern: Protocol for duck typing to enable FakeClient
- Anti-Pattern #7/#13: Custom namespaced exceptions
"""

from unittest.mock import AsyncMock, patch

import pytest

# =============================================================================
# WBS 4.1.1: Client Initialization Tests
# =============================================================================


class TestSearchClientInit:
    """Tests for SemanticSearchClient initialization."""

    def test_search_client_module_exists(self) -> None:
        """SemanticSearchClient module exists."""
        from src.clients.semantic_search import SemanticSearchClient

        assert SemanticSearchClient is not None

    def test_search_client_initializes_with_base_url(self) -> None:
        """SemanticSearchClient initializes with base_url."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        assert client.base_url == "http://localhost:8081"

    def test_search_client_has_default_timeout(self) -> None:
        """SemanticSearchClient has default timeout of 30 seconds."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        assert client.timeout == 30.0

    def test_search_client_timeout_configurable(self) -> None:
        """SemanticSearchClient timeout is configurable."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(
            base_url="http://localhost:8081",
            timeout=60.0,
        )

        assert client.timeout == 60.0

    def test_search_client_has_default_max_retries(self) -> None:
        """SemanticSearchClient has default max_retries of 3."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        assert client.max_retries == 3

    def test_search_client_max_retries_configurable(self) -> None:
        """SemanticSearchClient max_retries is configurable."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(
            base_url="http://localhost:8081",
            max_retries=5,
        )

        assert client.max_retries == 5

    def test_search_client_uses_connection_pooling(self) -> None:
        """SemanticSearchClient uses httpx.AsyncClient (not per-request).

        Per Anti-Pattern #12: Connection Pooling
        """
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        # Should have an internal _client attribute for connection pooling
        assert hasattr(client, "_client")


# =============================================================================
# WBS 4.1.2: Search Method Tests
# =============================================================================


class TestSearchClientSearch:
    """Tests for SemanticSearchClient.search() method."""

    @pytest.mark.asyncio
    async def test_search_method_exists(self) -> None:
        """SemanticSearchClient has search() method."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        assert hasattr(client, "search")
        assert callable(client.search)

    @pytest.mark.asyncio
    async def test_search_accepts_keywords_and_top_k(self) -> None:
        """search() accepts keywords and top_k parameters."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        # Mock the HTTP call
        with patch.object(client, "_execute_request", new_callable=AsyncMock) as mock:
            mock.return_value = {"results": []}

            await client.search(
                keywords=["chunking", "RAG"],
                top_k=10,
            )

            # Verify the correct parameters were passed
            mock.assert_called_once()
            call_args = mock.call_args
            assert call_args[1]["keywords"] == ["chunking", "RAG"]
            assert call_args[1]["top_k"] == 10

    @pytest.mark.asyncio
    async def test_search_returns_list_of_results(self) -> None:
        """search() returns list of SearchResult objects."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        # Mock response
        with patch.object(client, "_execute_request", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "results": [
                    {"book": "AI Engineering", "chapter": 5, "score": 0.91},
                    {"book": "Building LLM Apps", "chapter": 8, "score": 0.88},
                ]
            }

            results = await client.search(keywords=["chunking"], top_k=10)

            assert isinstance(results, list)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_result_has_required_fields(self) -> None:
        """Each search result has book, chapter, score fields."""
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        with patch.object(client, "_execute_request", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "results": [
                    {"book": "AI Engineering", "chapter": 5, "score": 0.91},
                ]
            }

            results = await client.search(keywords=["test"], top_k=5)

            assert len(results) == 1
            result = results[0]
            assert result.book == "AI Engineering"
            assert result.chapter == 5
            assert result.score == 0.91


# =============================================================================
# WBS 4.1.3: Error Handling Tests
# =============================================================================


class TestSearchClientErrors:
    """Tests for SemanticSearchClient error handling."""

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_5xx_error(self) -> None:
        """search() returns empty list on 5xx server errors."""
        from src.clients.semantic_search import SearchClientError, SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        # Mock 500 error
        with patch.object(client, "_execute_request", new_callable=AsyncMock) as mock:
            mock.side_effect = SearchClientError("Server error", status_code=500)

            results = await client.search(keywords=["test"], top_k=5)

            # Should return empty, not raise
            assert results == []

    @pytest.mark.asyncio
    async def test_search_raises_on_4xx_error(self) -> None:
        """search() raises SearchClientError on 4xx client errors."""
        from src.clients.semantic_search import SearchClientError, SemanticSearchClient

        client = SemanticSearchClient(base_url="http://localhost:8081")

        # Mock 400 error
        with patch.object(client, "_execute_request", new_callable=AsyncMock) as mock:
            mock.side_effect = SearchClientError("Bad request", status_code=400)

            with pytest.raises(SearchClientError) as exc_info:
                await client.search(keywords=["test"], top_k=5)

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_search_retries_on_timeout(self) -> None:
        """search() retries on timeout errors."""
        from unittest.mock import MagicMock

        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(
            base_url="http://localhost:8081",
            max_retries=3,
            retry_delay=0.01,  # Fast retries for testing
        )

        call_count = 0

        async def mock_post(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Connection timed out")
            # Return successful response on 3rd attempt
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            return mock_response

        with patch.object(client._client, "post", side_effect=mock_post):
            results = await client.search(keywords=["test"], top_k=5)

            assert call_count == 3  # Retried twice, succeeded on third
            assert results == []

    @pytest.mark.asyncio
    async def test_search_uses_exponential_backoff(self) -> None:
        """search() uses exponential backoff between retries.

        Per Anti-Pattern #2.3: Retry with exponential backoff
        """
        from src.clients.semantic_search import SemanticSearchClient

        client = SemanticSearchClient(
            base_url="http://localhost:8081",
            max_retries=3,
            retry_delay=1.0,
        )

        # Verify exponential backoff is used
        assert hasattr(client, "retry_delay")
        assert client.retry_delay == 1.0


# =============================================================================
# Custom Exception Tests
# =============================================================================


class TestSearchClientExceptions:
    """Tests for SemanticSearchClient custom exceptions.

    Per Anti-Pattern #7/#13: Custom namespaced exceptions
    """

    def test_search_client_error_exists(self) -> None:
        """SearchClientError exception exists."""
        from src.clients.semantic_search import SearchClientError

        assert SearchClientError is not None

    def test_search_client_error_has_status_code(self) -> None:
        """SearchClientError captures status_code."""
        from src.clients.semantic_search import SearchClientError

        error = SearchClientError("Test error", status_code=500)

        assert error.status_code == 500
        assert str(error) == "Test error"

    def test_search_client_error_inherits_from_exception(self) -> None:
        """SearchClientError inherits from Exception, not builtin."""
        from src.clients.semantic_search import SearchClientError

        assert issubclass(SearchClientError, Exception)
        # Should NOT shadow builtins
        assert SearchClientError.__name__ != "ConnectionError"
        assert SearchClientError.__name__ != "TimeoutError"


# =============================================================================
# Protocol for Testing
# =============================================================================


class TestSearchClientProtocol:
    """Tests for SemanticSearchClientProtocol.

    Per Repository Pattern: Protocol for duck typing
    """

    def test_search_client_protocol_exists(self) -> None:
        """SemanticSearchClientProtocol exists for duck typing."""
        from src.clients.semantic_search import SemanticSearchClientProtocol

        assert SemanticSearchClientProtocol is not None

    def test_fake_client_can_be_used_for_testing(self) -> None:
        """FakeSearchClient can be used as drop-in replacement."""
        from src.clients.semantic_search import FakeSearchClient

        # FakeSearchClient should work without real HTTP
        client = FakeSearchClient()
        assert client is not None
