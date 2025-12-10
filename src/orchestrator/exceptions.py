"""
Code-Orchestrator-Service - Orchestrator Exceptions

WBS 3.1: Namespaced exceptions for orchestrator operations.

Patterns Applied:
- Anti-Pattern #7, #13 Prevention (CODING_PATTERNS_ANALYSIS.md)
- All exception classes namespaced to avoid shadowing builtins
- Per Comp_Static_Analysis_Report Issue #7 (ollama.py exception shadowing fix)
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors.

    Pattern: Namespaced base exception per CODING_PATTERNS_ANALYSIS.md
    Anti-Pattern #7, #13: Exception names should not shadow builtins.
    """


class StageError(OrchestratorError):
    """Exception raised when a pipeline stage fails.

    Captures which stage failed for debugging and retry logic.
    """

    def __init__(self, stage: str, message: str) -> None:
        """Initialize StageError with stage name and message.

        Args:
            stage: Name of the failed stage (generate, validate, rank, consensus)
            message: Error description
        """
        self.stage = stage
        self.message = message
        super().__init__(f"Stage '{stage}' failed: {message}")


class RetryExhaustedError(OrchestratorError):
    """Exception raised when max retry attempts are exhausted.

    WBS 3.1.4: Track retry attempts for debugging.
    """

    def __init__(self, attempts: int, last_error: str) -> None:
        """Initialize RetryExhaustedError with attempt count.

        Args:
            attempts: Number of attempts made
            last_error: Description of the last error
        """
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Retry exhausted after {attempts} attempts. Last error: {last_error}"
        )


class ValidationError(OrchestratorError):
    """Exception raised when input validation fails."""


class ConfigurationError(OrchestratorError):
    """Exception raised when orchestrator configuration is invalid."""
