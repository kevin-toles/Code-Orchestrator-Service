"""
Code-Orchestrator-Service - Agent Protocols

WBS 2.1-2.4: Model Loading Infrastructure
Defines Protocol interfaces for duck typing support.

Patterns Applied:
- Protocol typing for duck typing (CODING_PATTERNS_ANALYSIS.md line 130)
- Structural subtyping (no inheritance required)

Anti-Patterns Avoided:
- Tight coupling to concrete implementations
"""

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class ModelRegistryProtocol(Protocol):
    """Protocol for ModelRegistry duck typing.

    Enables FakeModelRegistry for testing without real HuggingFace models.
    Pattern: Repository Pattern + FakeClient per CODING_PATTERNS_ANALYSIS.md line 130
    """

    def get_model(self, model_name: str) -> Any | None:
        """Get a loaded model by name."""
        ...

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        ...

    def register_model(self, model_name: str, model: Any) -> None:
        """Register a model in the cache."""
        ...

    def all_models_loaded(self) -> bool:
        """Check if all required models are loaded."""
        ...


class GeneratorAgentProtocol(Protocol):
    """Protocol for Generator agents (CodeT5+).

    Agents that generate/extract terms from text.
    WBS 2.2: CodeT5+ Generator Agent
    """

    def extract_terms(self, text: str, timeout_seconds: float = 30.0) -> Any:
        """Extract technical terms from text."""
        ...

    def extract_terms_batch(self, texts: list[str]) -> list[Any]:
        """Batch process multiple texts."""
        ...


class ValidatorAgentProtocol(Protocol):
    """Protocol for Validator agents (GraphCodeBERT).

    Agents that validate and filter terms.
    WBS 2.3: GraphCodeBERT Validator Agent
    """

    def validate_terms(
        self, terms: list[str], original_query: str, domain: str
    ) -> Any:
        """Validate terms against query and domain."""
        ...

    def classify_domain(self, text: str) -> str:
        """Classify the domain of given text."""
        ...

    def expand_terms(
        self, terms: list[str], domain: str, max_expansions: int = 3
    ) -> list[str]:
        """Expand terms with semantically related terms."""
        ...


class RankerAgentProtocol(Protocol):
    """Protocol for Ranker agents (CodeBERT).

    Agents that generate embeddings and rank terms.
    WBS 2.4: CodeBERT Ranker Agent
    """

    def get_embedding(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding vector for text."""
        ...

    def get_embeddings_batch(self, texts: list[str]) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple texts."""
        ...

    def calculate_similarity(self, term: str, query: str) -> float:
        """Calculate cosine similarity between term and query."""
        ...

    def rank_terms(self, terms: list[str], query: str) -> Any:
        """Rank terms by relevance to query."""
        ...
