"""
Code-Orchestrator-Service - Orchestrator Class

WBS 3.1: Main orchestrator class that wraps the LangGraph pipeline.
WBS 3.1.4: Add retry logic - Max 3 retries on failure.

Patterns Applied:
- Protocol implementation for duck typing
- One-time configuration (Anti-Pattern #16)
- Retry with state tracking
"""

from __future__ import annotations

import time
from typing import Any

from src.core.logging import get_logger
from src.orchestrator.exceptions import RetryExhaustedError
from src.orchestrator.graph import create_orchestrator_graph
from src.orchestrator.state import (
    OrchestratorOptions,
    OrchestratorResult,
    SearchTerm,
    create_initial_state,
)

logger = get_logger(__name__)


class Orchestrator:
    """Main orchestrator class for the multi-model pipeline.

    WBS 3.1: Wraps LangGraph StateGraph with retry logic and configuration.

    Usage:
        orchestrator = Orchestrator()
        result = orchestrator.run({
            "query": "LLM document chunking",
            "domain": "ai-ml"
        })
    """

    def __init__(self, options: OrchestratorOptions | None = None) -> None:
        """Initialize orchestrator with configuration.

        Args:
            options: Pipeline configuration options. Defaults to OrchestratorOptions().
        """
        self._options = options or OrchestratorOptions()
        self._graph = create_orchestrator_graph()  # Returns CompiledStateGraph

        logger.info(
            "orchestrator_initialized",
            max_retries=self._options.max_retries,
            min_confidence=self._options.min_confidence,
            max_terms=self._options.max_terms,
        )

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts configured.

        WBS 3.1.4: Default is 3.
        """
        return self._options.max_retries

    def run(self, input_data: dict[str, Any]) -> OrchestratorResult:
        """Run the orchestration pipeline.

        WBS 3.1: Full pipeline: generate → validate → rank → consensus.

        Args:
            input_data: Dict with "query" and "domain" keys

        Returns:
            OrchestratorResult with stages_completed and search_terms

        Raises:
            RetryExhaustedError: If max retries exceeded
        """
        start_time = time.time()

        query = input_data.get("query", "")
        domain = input_data.get("domain", "general")

        logger.info("orchestrator_run_start", query=query, domain=domain)

        # Create initial state
        state = create_initial_state(query, domain)

        # Run with retry logic
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                # Invoke the graph
                final_state = self._graph.invoke(state)

                # Check for errors in state
                if final_state.get("errors") and self._should_retry(final_state):
                    last_error = final_state["errors"][-1]
                    state["retry_count"] = attempt + 1
                    logger.warning(
                        "orchestrator_retry",
                        attempt=attempt + 1,
                        error=last_error,
                    )
                    continue

                # Success - build result
                processing_time = (time.time() - start_time) * 1000
                result = self._build_result(final_state, processing_time)

                logger.info(
                    "orchestrator_run_complete",
                    stages=result.stages_completed,
                    term_count=len(result.search_terms),
                    processing_time_ms=result.processing_time_ms,
                )

                return result

            except Exception as e:
                last_error = str(e)
                state["retry_count"] = attempt + 1
                logger.error(
                    "orchestrator_error",
                    attempt=attempt + 1,
                    error=last_error,
                )

        # Max retries exhausted
        raise RetryExhaustedError(
            attempts=self.max_retries + 1,
            last_error=last_error,
        )

    def _should_retry(self, state: dict[str, Any]) -> bool:
        """Determine if retry should be attempted.

        WBS 3.1.4: Check for transient errors and retry count.

        Args:
            state: Current orchestrator state

        Returns:
            True if retry should be attempted
        """
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)

        if not errors:
            return False

        last_error = errors[-1] if errors else ""

        # Check if error is retryable (transient)
        transient_patterns = [
            "timeout",
            "network",
            "connection",
            "temporary",
            "transient",
        ]

        is_transient = any(
            pattern in last_error.lower() for pattern in transient_patterns
        )

        return is_transient and retry_count < self.max_retries

    def _build_result(
        self, state: dict[str, Any], processing_time_ms: float
    ) -> OrchestratorResult:
        """Build OrchestratorResult from final state.

        Args:
            state: Final orchestrator state after pipeline
            processing_time_ms: Total processing time in milliseconds

        Returns:
            OrchestratorResult with all fields populated
        """
        # Convert final_terms to SearchTerm objects
        search_terms: list[SearchTerm] = []
        for term_data in state.get("final_terms", []):
            search_terms.append(
                SearchTerm(
                    term=term_data["term"],
                    score=term_data["score"],
                    models_agreed=term_data.get("models_agreed", 1),
                )
            )

        # Apply max_terms limit
        search_terms = search_terms[: self._options.max_terms]

        # Filter by min_confidence
        search_terms = [
            t for t in search_terms if t.score >= self._options.min_confidence
        ]

        return OrchestratorResult(
            stages_completed=state.get("stages_completed", []),
            search_terms=search_terms,
            excluded_terms=state.get("excluded_terms", []),
            errors=state.get("errors", []),
            processing_time_ms=processing_time_ms,
        )
