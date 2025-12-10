"""
Code-Orchestrator-Service - Custom Exceptions

WBS 1.2: FastAPI Application Shell

Anti-Patterns Avoided:
- #7, #13 (Exception Shadowing): Custom namespaced exceptions per CODING_PATTERNS_ANALYSIS.md
  Use CodeOrchestratorError instead of shadowing builtins like ConnectionError
"""


class CodeOrchestratorError(Exception):
    """Base exception for Code-Orchestrator-Service.

    All custom exceptions inherit from this base class.
    Pattern: Custom namespaced exceptions per CODING_PATTERNS_ANALYSIS.md Phase 2.
    """
    pass


class ModelLoadError(CodeOrchestratorError):
    """Raised when a model fails to load.

    Used in Phase 2 for HuggingFace model loading failures.
    """
    pass


class ModelNotReadyError(CodeOrchestratorError):
    """Raised when a model is accessed before it's loaded.

    Used by /ready endpoint to indicate service is not ready.
    """
    pass


class ConfigurationError(CodeOrchestratorError):
    """Raised when configuration is invalid or missing."""
    pass
