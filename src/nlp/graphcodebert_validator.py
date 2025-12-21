"""GraphCodeBERT Concept Validator - Lightweight wrapper for pipeline integration.

Uses GraphCodeBERT embeddings to validate that extracted concepts are
technically meaningful code/programming terms.

This is a simplified validator for the concept extraction pipeline.
For full GraphCodeBERT functionality, see src/models/graphcodebert_validator.py.

Anti-Patterns Avoided:
- #12: Model loaded once and cached
- S1192: Constants at module level
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Module Constants
# =============================================================================

# Default model name
GRAPHCODEBERT_MODEL_ID: Final[str] = "microsoft/graphcodebert-base"

# Validation thresholds
DEFAULT_MIN_SIMILARITY: Final[float] = 0.15
DEFAULT_MIN_TERM_LENGTH: Final[int] = 2

# Technical reference text for similarity scoring
TECHNICAL_REFERENCE: Final[str] = """
software design architecture pattern interface module abstraction
complexity coupling cohesion encapsulation inheritance polymorphism
algorithm data structure implementation refactoring debugging testing
code documentation API endpoint function method class variable
microservices kubernetes docker container deployment CI/CD pipeline
database query SQL transaction schema migration ORM model
authentication authorization security encryption protocol networking
concurrency threading async parallel distributed system cache
machine learning neural network model training inference embedding
"""


# =============================================================================
# Configuration and Result Dataclasses
# =============================================================================


@dataclass
class GraphCodeBERTConfig:
    """Configuration for GraphCodeBERT validation.
    
    Attributes:
        min_similarity: Minimum cosine similarity to technical reference.
        min_term_length: Minimum characters for valid term.
        batch_size: Batch size for embedding computation.
    """
    
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    min_term_length: int = DEFAULT_MIN_TERM_LENGTH
    batch_size: int = 32


@dataclass
class GraphCodeBERTResult:
    """Result of GraphCodeBERT validation.
    
    Attributes:
        valid_concepts: Concepts that passed validation.
        rejected_concepts: Concepts that were rejected.
        rejection_reasons: Map of rejected concept -> reason.
        similarity_scores: Map of valid concept -> similarity score.
    """
    
    valid_concepts: list[str] = field(default_factory=list)
    rejected_concepts: list[str] = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)
    similarity_scores: dict[str, float] = field(default_factory=dict)


# =============================================================================
# GraphCodeBERT Validator Class
# =============================================================================


class GraphCodeBERTConceptValidator:
    """Validates concepts using GraphCodeBERT embeddings.
    
    Uses cosine similarity between concept embeddings and a technical
    reference text to filter out non-technical terms.
    
    The model is loaded lazily on first use and cached.
    """
    
    # Class-level cache for model (singleton pattern)
    _model: Any | None = None
    _tokenizer: Any | None = None
    _reference_embedding: NDArray[np.floating[Any]] | None = None
    
    def __init__(self, config: GraphCodeBERTConfig | None = None) -> None:
        """Initialize the validator.
        
        Args:
            config: Validation configuration.
        """
        self.config = config or GraphCodeBERTConfig()
        self._embedding_cache: dict[str, NDArray[np.floating[Any]]] = {}
    
    @classmethod
    def _load_model(cls) -> tuple[Any, Any]:
        """Load GraphCodeBERT model (lazy, cached).
        
        Returns:
            Tuple of (model, tokenizer).
            
        Raises:
            ImportError: If transformers not installed.
        """
        if cls._model is not None:
            return cls._model, cls._tokenizer
        
        logger.info("loading_graphcodebert_model")
        
        from transformers import AutoModel, AutoTokenizer
        
        cls._tokenizer = AutoTokenizer.from_pretrained(GRAPHCODEBERT_MODEL_ID)
        cls._model = AutoModel.from_pretrained(GRAPHCODEBERT_MODEL_ID)
        
        logger.info("graphcodebert_model_loaded")
        
        return cls._model, cls._tokenizer
    
    def _get_embedding(self, text: str) -> NDArray[np.floating[Any]]:
        """Get embedding for text using GraphCodeBERT.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector (768-dim).
        """
        # Check cache
        cache_key = text[:100]
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        model, tokenizer = self._load_model()
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        
        # Get embedding (mean pooling)
        import torch
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Cache
        self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _get_reference_embedding(self) -> NDArray[np.floating[Any]]:
        """Get cached reference text embedding.
        
        Returns:
            Technical reference embedding.
        """
        if GraphCodeBERTConceptValidator._reference_embedding is None:
            GraphCodeBERTConceptValidator._reference_embedding = self._get_embedding(
                TECHNICAL_REFERENCE
            )
        return GraphCodeBERTConceptValidator._reference_embedding
    
    def _cosine_similarity(
        self,
        vec1: NDArray[np.floating[Any]],
        vec2: NDArray[np.floating[Any]],
    ) -> float:
        """Calculate cosine similarity.
        
        Args:
            vec1: First vector.
            vec2: Second vector.
            
        Returns:
            Similarity score (0-1).
        """
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return max(0.0, float(dot / (norm1 * norm2)))
    
    def validate(self, concepts: list[str]) -> GraphCodeBERTResult:
        """Validate concepts using GraphCodeBERT embeddings.
        
        Args:
            concepts: List of concepts to validate.
            
        Returns:
            GraphCodeBERTResult with valid/rejected concepts.
        """
        result = GraphCodeBERTResult()
        
        if not concepts:
            return result
        
        # Get reference embedding
        ref_embedding = self._get_reference_embedding()
        
        for concept in concepts:
            # Length check
            if len(concept) < self.config.min_term_length:
                result.rejected_concepts.append(concept)
                result.rejection_reasons[concept] = "too_short"
                continue
            
            # Compute similarity
            concept_embedding = self._get_embedding(concept)
            similarity = self._cosine_similarity(concept_embedding, ref_embedding)
            
            if similarity >= self.config.min_similarity:
                result.valid_concepts.append(concept)
                result.similarity_scores[concept] = round(similarity, 4)
            else:
                result.rejected_concepts.append(concept)
                result.rejection_reasons[concept] = f"low_similarity:{similarity:.3f}"
        
        logger.debug(
            "graphcodebert_validation_complete",
            valid=len(result.valid_concepts),
            rejected=len(result.rejected_concepts),
        )
        
        return result
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if GraphCodeBERT can be loaded.
        
        Returns:
            True if transformers is installed.
        """
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
