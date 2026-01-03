"""
Code-Orchestrator-Service - Model Protocols

WBS 2.1-2.4: Model Loading Infrastructure
Defines Protocol interfaces for duck typing support.

Models loaded from local paths in models/ directory.

Naming Convention:
- ExtractorProtocol: For models that extract/generate terms (CodeT5+)
- ValidatorProtocol: For models that validate/filter terms (GraphCodeBERT)
- RankerProtocol: For models that rank terms by relevance (CodeBERT)

Patterns Applied:
- Protocol typing for duck typing (CODING_PATTERNS_ANALYSIS.md line 130)
- Structural subtyping (no inheritance required)
"""

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class ExtractorProtocol(Protocol):
    """Protocol for Extractor models (CodeT5+).

    Model wrappers that generate/extract terms from text.
    WBS 2.2: CodeT5+ Extractor
    """

    def extract_terms(self, text: str, timeout_seconds: float = 30.0) -> Any:
        """Extract technical terms from text."""
        ...

    def extract_terms_batch(self, texts: list[str]) -> list[Any]:
        """Batch process multiple texts."""
        ...


# Backward compatibility alias
GeneratorModelProtocol = ExtractorProtocol


class ValidatorProtocol(Protocol):
    """Protocol for Validator models (GraphCodeBERT).

    Model wrappers that validate and filter terms.
    WBS 2.3: GraphCodeBERT Validator
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


# Backward compatibility alias
ValidatorModelProtocol = ValidatorProtocol


class RankerProtocol(Protocol):
    """Protocol for Ranker models (CodeBERT).

    Model wrappers that generate embeddings and rank terms.
    WBS 2.4: CodeBERT Ranker
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


# Backward compatibility alias
RankerModelProtocol = RankerProtocol


class TopicClustererProtocol(Protocol):
    """Protocol for TopicClusterer models (BERTopic).

    Model wrappers that cluster documents into semantic topics.
    WBS: BERTOPIC_INTEGRATION_WBS.md

    Enables FakeTopicClusterer for testing without BERTopic.
    Pattern: Protocol typing for duck typing
    """

    @property
    def topics(self) -> list[Any]:
        """Get list of discovered topics."""
        ...

    @property
    def embedding_model(self) -> str:
        """Get the embedding model name."""
        ...

    @property
    def is_using_fallback(self) -> bool:
        """Check if using fallback clustering."""
        ...

    def cluster(
        self,
        corpus: list[str],
        embeddings: npt.NDArray[np.floating[Any]] | None = None,
    ) -> Any:
        """Cluster documents into topics."""
        ...

    def get_topic_info(self, topic_id: int) -> Any:
        """Get information about a specific topic."""
        ...
