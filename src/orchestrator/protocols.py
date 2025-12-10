"""
Code-Orchestrator-Service - Orchestrator Protocols

Protocols for duck typing in tests (FakeOrchestrator pattern).

Pattern: Protocol typing per CODING_PATTERNS_ANALYSIS.md line 130
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.orchestrator.state import OrchestratorResult


class OrchestratorProtocol(Protocol):
    """Protocol for Orchestrator duck typing.

    Enables FakeOrchestrator for testing without real models.
    Pattern: Repository Pattern + FakeClient per CODING_PATTERNS_ANALYSIS.md line 150
    """

    def run(self, input_data: dict[str, Any]) -> OrchestratorResult:
        """Run the orchestration pipeline.

        Args:
            input_data: Dict with "query" and "domain" keys

        Returns:
            OrchestratorResult with stages_completed and search_terms
        """
        ...

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts configured."""
        ...
