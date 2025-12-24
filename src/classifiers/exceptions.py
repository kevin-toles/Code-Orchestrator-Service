"""
Custom exceptions for the classifiers module.

All exception classes follow AC-8.3: names end with "Error" and do not
shadow built-in exception names.
"""

from __future__ import annotations


class ConceptClassifierError(Exception):
    """
    Exception raised when the concept classifier encounters an error.

    This exception is raised in scenarios such as:
    - Model file not found
    - Invalid model format
    - Prediction failures

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize ConceptClassifierError with a message.

        Args:
            message: Human-readable description of the error.
        """
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """Return string representation of the error."""
        return self.message


class LLMFallbackError(Exception):
    """
    Exception raised when the LLM fallback tier encounters an error.

    This exception is raised in scenarios such as:
    - Timeout when calling ai-agents service
    - Invalid response format
    - Network errors

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize LLMFallbackError with a message.

        Args:
            message: Human-readable description of the error.
        """
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        """Return string representation of the error."""
        return self.message
