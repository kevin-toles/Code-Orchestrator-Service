"""
LLM Fallback (Tier 4) for concept classification.

WBS: WBS-AC4 - LLM Fallback
AC Block: AC-4.1 through AC-4.6

This module provides LLM-based classification for terms that couldn't be
classified by Tiers 1-3. It calls the ai-agents service to validate concepts.

Pattern: Tool Proxy (llm-gateway ARCHITECTURE.md)
- Uses httpx.AsyncClient for async HTTP requests
- Caches high-confidence results in Tier 1 (AliasLookup)
- Implements proper timeout and error handling

Endpoint: POST ai-agents:8082/v1/agents/validate-concept
Request:  {"term": "string"}
Response: {"classification": "concept|keyword", "confidence": float, "canonical_term": str}

Reference: CODING_PATTERNS_ANALYSIS.md - Anti-Pattern Compliance
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from src.classifiers.exceptions import LLMFallbackError

# =============================================================================
# Constants (AC-8.1: No duplicated strings)
# =============================================================================

DEFAULT_BASE_URL = "http://ai-agents:8082"
ENDPOINT_PATH = "/v1/agents/validate-concept"
DEFAULT_TIMEOUT = 10.0
CACHE_CONFIDENCE_THRESHOLD = 0.9
TIER_LLM_FALLBACK = 4

# Response field names
FIELD_CLASSIFICATION = "classification"
FIELD_CONFIDENCE = "confidence"
FIELD_CANONICAL_TERM = "canonical_term"

# Default values
DEFAULT_CLASSIFICATION = "unknown"
DEFAULT_CONFIDENCE = 0.5


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class LLMFallbackResult:
    """Result from LLM fallback classification.

    Attributes:
        classification: Either "concept" or "keyword"
        confidence: Confidence score from LLM (0.0 to 1.0)
        canonical_term: Normalized canonical term
        tier_used: Always 4 for LLM fallback tier
    """

    classification: str
    confidence: float
    canonical_term: str
    tier_used: int


# =============================================================================
# Protocol (AC-8.4: Protocol-based fakes)
# =============================================================================


@runtime_checkable
class AliasCacheProtocol(Protocol):
    """Protocol for alias cache implementations.

    Defines the interface required for caching high-confidence results.
    This allows LLMFallback to work with any cache implementation
    that provides an add() method.
    """

    def add(
        self, *, term: str, canonical_term: str, classification: str
    ) -> None:
        """Add a term to the alias cache.

        Args:
            term: The original term to cache
            canonical_term: The canonical/normalized form
            classification: Either "concept" or "keyword"
        """
        ...


@runtime_checkable
class LLMFallbackProtocol(Protocol):
    """Protocol for LLM fallback implementations.

    Enables dependency injection and test doubles.
    """

    async def classify(self, term: str) -> LLMFallbackResult:
        """Classify a term using LLM.

        Args:
            term: The term to classify

        Returns:
            LLMFallbackResult with classification details
        """
        ...

    async def classify_batch(self, terms: list[str]) -> list[LLMFallbackResult]:
        """Classify multiple terms.

        Args:
            terms: List of terms to classify

        Returns:
            List of LLMFallbackResult objects
        """
        ...


# =============================================================================
# Main Implementation
# =============================================================================


class LLMFallback:
    """LLM-based classification for Tier 4 fallback.

    Calls ai-agents service to classify terms that couldn't be handled
    by earlier tiers. High-confidence results are cached in Tier 1.

    Usage:
        fallback = LLMFallback(alias_lookup=alias_lookup)
        result = await fallback.classify("unknown term")
    """

    def __init__(
        self,
        alias_lookup: AliasCacheProtocol | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize LLMFallback.

        Args:
            alias_lookup: Optional cache for caching high-confidence results
            base_url: Base URL for ai-agents service
            timeout: Request timeout in seconds
        """
        self._alias_lookup = alias_lookup
        self._base_url = base_url
        self.timeout = timeout

    async def classify(self, term: str) -> LLMFallbackResult:
        """Classify a term using LLM fallback.

        Makes POST request to ai-agents service, parses response,
        and optionally caches high-confidence results.

        Args:
            term: The term to classify

        Returns:
            LLMFallbackResult with classification details

        Raises:
            LLMFallbackError: On timeout, connection error, or invalid response
        """
        url = f"{self._base_url}{ENDPOINT_PATH}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json={"term": term})
        except httpx.TimeoutException as e:
            msg = f"Timeout calling ai-agents for term '{term}': {e}"
            raise LLMFallbackError(msg) from e
        except httpx.ConnectError as e:
            msg = f"Connection error calling ai-agents for term '{term}': {e}"
            raise LLMFallbackError(msg) from e
        except httpx.HTTPError as e:
            msg = f"HTTP error calling ai-agents for term '{term}': {e}"
            raise LLMFallbackError(msg) from e

        if response.status_code != 200:
            msg = f"ai-agents returned status {response.status_code} for term '{term}'"
            raise LLMFallbackError(msg)

        result = self._parse_response(response.json(), term)

        # Cache high-confidence results
        if self._alias_lookup and result.confidence >= CACHE_CONFIDENCE_THRESHOLD:
            self._alias_lookup.add(
                term=term,
                canonical_term=result.canonical_term,
                classification=result.classification,
            )

        return result

    def _parse_response(
        self, data: dict[str, Any], original_term: str
    ) -> LLMFallbackResult:
        """Parse JSON response from ai-agents.

        Args:
            data: JSON response data
            original_term: Original term (used if canonical_term missing)

        Returns:
            LLMFallbackResult

        Raises:
            LLMFallbackError: If required fields are missing
        """
        if FIELD_CLASSIFICATION not in data or FIELD_CONFIDENCE not in data:
            msg = f"Invalid response from ai-agents: missing required fields in {data}"
            raise LLMFallbackError(msg)

        classification = data[FIELD_CLASSIFICATION]
        confidence = float(data[FIELD_CONFIDENCE])
        canonical_term = data.get(FIELD_CANONICAL_TERM, original_term)

        return LLMFallbackResult(
            classification=classification,
            confidence=confidence,
            canonical_term=canonical_term,
            tier_used=TIER_LLM_FALLBACK,
        )

    async def classify_batch(self, terms: list[str]) -> list[LLMFallbackResult]:
        """Classify multiple terms.

        Currently processes sequentially. Could be optimized with
        asyncio.gather for parallel requests if needed.

        Args:
            terms: List of terms to classify

        Returns:
            List of LLMFallbackResult objects
        """
        if not terms:
            return []

        results: list[LLMFallbackResult] = []
        for term in terms:
            result = await self.classify(term)
            results.append(result)
        return results


# =============================================================================
# Test Double (AC-8.4: Protocol-based fakes)
# =============================================================================


class FakeLLMFallback:
    """Fake LLMFallback for testing.

    Implements LLMFallbackProtocol for use in unit tests.
    Returns pre-configured responses without making HTTP requests.

    Usage:
        fake = FakeLLMFallback(responses={"ML": result})
        result = await fake.classify("ML")
    """

    def __init__(
        self,
        responses: Mapping[str, LLMFallbackResult] | None = None,
        error: LLMFallbackError | None = None,
    ) -> None:
        """Initialize with pre-configured responses.

        Args:
            responses: Dict mapping terms to expected results
            error: Optional error to raise on any classify call
        """
        self._responses: dict[str, LLMFallbackResult] = (
            dict(responses) if responses else {}
        )
        self._error = error

    async def classify(self, term: str) -> LLMFallbackResult:
        """Return pre-configured response for term.

        Args:
            term: The term to classify

        Returns:
            Configured result, or default "unknown" result

        Raises:
            LLMFallbackError: If error was configured
        """
        if self._error:
            raise self._error

        if term in self._responses:
            return self._responses[term]

        # Default response for unconfigured terms
        return LLMFallbackResult(
            classification=DEFAULT_CLASSIFICATION,
            confidence=DEFAULT_CONFIDENCE,
            canonical_term=term,
            tier_used=TIER_LLM_FALLBACK,
        )

    async def classify_batch(self, terms: list[str]) -> list[LLMFallbackResult]:
        """Classify multiple terms using configured responses.

        Args:
            terms: List of terms to classify

        Returns:
            List of results
        """
        return [await self.classify(term) for term in terms]
