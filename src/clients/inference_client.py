"""Inference Service Client - WBS Summary Generation.

HTTP client for inference-service to generate chapter summaries using local LLM.

Architecture:
- Code-Orchestrator-Service calls inference-service DIRECTLY (internal communication)
- External applications call through LLM Gateway
- Per Kitchen Brigade: internal services talk directly, external goes through Gateway

Patterns Applied (from CODING_PATTERNS_ANALYSIS.md):
- Anti-Pattern #12: Connection pooling (reuse httpx.AsyncClient)
- Anti-Pattern #2.3: Retry with exponential backoff
- Anti-Pattern #7/#13: Custom namespaced exceptions
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Final

import httpx


# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

DEFAULT_INFERENCE_URL: Final[str] = os.getenv(
    "INFERENCE_SERVICE_URL", "http://localhost:8085"
)
DEFAULT_TIMEOUT: Final[float] = 60.0  # LLM generation can take time
DEFAULT_MAX_RETRIES: Final[int] = 2
DEFAULT_RETRY_DELAY: Final[float] = 1.0

# Chat completions endpoint
ENDPOINT_CHAT: Final[str] = "/v1/chat/completions"
ENDPOINT_MODELS: Final[str] = "/v1/models"

# Default model - None means let inference-service decide via graceful degradation
# The inference-service will use whatever model is currently loaded
DEFAULT_MODEL: Final[str | None] = os.getenv("INFERENCE_SUMMARY_MODEL") or None

# Summary generation prompt template
SUMMARY_PROMPT_TEMPLATE: Final[str] = """Generate a concise summary of the following chapter content.
The summary should:
1. Capture the main topics and key concepts
2. Be 2-3 paragraphs (150-300 words)
3. Use clear, technical language appropriate for software engineering documentation
4. Focus on actionable insights and patterns

Chapter Title: {title}

Content:
{text}

Summary:"""


# =============================================================================
# Custom Exceptions (Anti-Pattern #7/#13: Namespaced)
# =============================================================================


class InferenceClientError(Exception):
    """Base exception for InferenceClient errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class InferenceTimeoutError(InferenceClientError):
    """Raised when inference request times out."""

    pass


class InferenceConnectionError(InferenceClientError):
    """Raised when unable to connect to inference-service."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SummaryResult:
    """Result from summary generation."""

    summary: str
    model: str
    tokens_used: int = 0
    generation_time_ms: float = 0.0


# =============================================================================
# InferenceClient Implementation
# =============================================================================


class InferenceClient:
    """HTTP client for inference-service.

    Uses connection pooling (Anti-Pattern #12) and exponential backoff retry.

    Attributes:
        base_url: Base URL of inference-service (default: http://localhost:8085)
        timeout: Request timeout in seconds (default: 60)
        max_retries: Maximum retry attempts (default: 2)
        model: Model ID to use for generation
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        model: str | None = None,
    ) -> None:
        """Initialize the inference client.

        Args:
            base_url: Base URL of inference-service
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            model: Model ID for generation
        """
        self.base_url = base_url or DEFAULT_INFERENCE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model or DEFAULT_MODEL
        self._retry_delay = DEFAULT_RETRY_DELAY

        # Connection pooling: single client instance (Anti-Pattern #12)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "InferenceClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded models from inference-service.
        
        Returns:
            List of model IDs that are currently loaded and ready.
        """
        if self._client is None:
            raise InferenceClientError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._client.get(ENDPOINT_MODELS)
            response.raise_for_status()
            data = response.json()
            # Return only models with status "loaded"
            return [m["id"] for m in data.get("data", []) if m.get("status") == "loaded"]
        except Exception:
            return []

    async def _resolve_model(self) -> str:
        """Resolve which model to use for generation.
        
        Strategy (graceful degradation):
        1. Query inference-service for loaded models
        2. If preferred model (self.model) is loaded, use it
        3. Otherwise, use first available loaded model
        4. If nothing loaded, raise clear error
        
        Returns:
            Model ID to use for generation.
            
        Raises:
            InferenceClientError: If no models are loaded.
        """
        loaded = await self.get_loaded_models()
        
        if not loaded:
            raise InferenceClientError("No models loaded in inference-service")
        
        if self.model and self.model in loaded:
            # Preferred model is loaded - use it
            return self.model
        
        # Use first loaded model (graceful degradation)
        return loaded[0]

    async def generate_summary(
        self,
        text: str,
        title: str | None = None,
        max_tokens: int = 1500,
    ) -> SummaryResult:
        """Generate a summary for the given text using LLM.

        Args:
            text: Chapter/document text to summarize
            title: Optional title for context
            max_tokens: Maximum tokens in response (1500 to accommodate thinking models)

        Returns:
            SummaryResult with generated summary

        Raises:
            InferenceClientError: On API errors
            InferenceTimeoutError: On timeout
            InferenceConnectionError: On connection failure
        """
        if self._client is None:
            raise InferenceClientError("Client not initialized. Use async context manager.")

        # Resolve which model to use (prefer loaded models)
        model_to_use = await self._resolve_model()

        # Build prompt
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            title=title or "Untitled",
            text=text[:8000],  # Truncate to fit context window
        )

        # Build request payload (OpenAI-compatible format)
        payload = {
            "model": model_to_use,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,  # Lower temperature for more focused summaries
        }

        # Execute with retry
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(ENDPOINT_CHAT, json=payload)
                response.raise_for_status()
                
                data = response.json()
                return self._parse_response(data)

            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise InferenceTimeoutError(
                        f"Timeout after {self.max_retries + 1} attempts: {e}"
                    ) from e
                await asyncio.sleep(self._retry_delay * (2 ** attempt))

            except httpx.ConnectError as e:
                raise InferenceConnectionError(
                    f"Failed to connect to inference-service at {self.base_url}: {e}"
                ) from e

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries:
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
                    continue
                raise InferenceClientError(
                    f"Inference API error: {e}",
                    status_code=e.response.status_code,
                ) from e

        # Should not reach here
        raise InferenceClientError("Unexpected error in retry loop")

    def _parse_response(self, data: dict[str, Any]) -> SummaryResult:
        """Parse inference API response.

        Args:
            data: JSON response from inference-service

        Returns:
            SummaryResult with extracted summary
        """
        choices = data.get("choices", [])
        if not choices:
            return SummaryResult(summary="", model=self.model or "unknown")

        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        
        # Strip chain-of-thought <think> tags from deepseek models
        content = self._strip_think_tags(content)

        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        return SummaryResult(
            summary=content,
            model=data.get("model", self.model or "unknown"),
            tokens_used=tokens_used,
        )

    @staticmethod
    def _strip_think_tags(content: str) -> str:
        """Strip <think> tags from model output.
        
        DeepSeek models include chain-of-thought reasoning in <think> tags.
        This removes those tags and returns only the final response.
        
        Handles:
        - Complete <think>...</think> blocks
        - Incomplete <think>... blocks (no closing tag - hit max_tokens)
        
        Args:
            content: Raw model output
            
        Returns:
            Content with thinking removed, or empty string if only thinking
        """
        import re
        
        # First try to remove complete <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        
        # If still has unclosed <think> tag, remove everything from <think> onwards
        # This handles cases where model hit max_tokens during thinking
        if '<think>' in cleaned:
            cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
        
        return cleaned.strip()

    async def health_check(self) -> bool:
        """Check if inference-service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        if self._client is None:
            return False

        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False


# =============================================================================
# Singleton Instance (Anti-Pattern #12)
# =============================================================================

_inference_client: InferenceClient | None = None


def get_inference_client() -> InferenceClient:
    """Get cached InferenceClient instance.

    Returns:
        Cached InferenceClient (must use with async context manager)
    """
    global _inference_client
    if _inference_client is None:
        _inference_client = InferenceClient()
    return _inference_client
