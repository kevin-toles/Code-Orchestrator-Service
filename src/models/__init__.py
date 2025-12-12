"""Model wrappers for keyword extraction, validation, and ranking.

NOTE: These are HuggingFace model wrappers, NOT autonomous agents.
Autonomous agents (LangGraph workflows) live in the ai-agents service.

Model Wrappers:
- CodeT5Extractor: Extracts terms from text using CodeT5+ model
- GraphCodeBERTValidator: Filters generic terms using GraphCodeBERT model
- CodeBERTRanker: Scores and ranks by similarity using CodeBERT model

Patterns applied from CODING_PATTERNS_ANALYSIS.md:
- Repository Pattern with Protocol (Phase 2)
- FakeClient for testing (Anti-Pattern #12)
"""

from src.models.codebert_ranker import CodeBERTRanker, RankedTerm, RankingResult
from src.models.codet5_extractor import CodeT5Extractor, TermExtractionResult
from src.models.graphcodebert_validator import (
    GraphCodeBERTValidator,
    ValidatedTerm,
    ValidationResult,
)
from src.models.protocols import (
    ExtractorProtocol,
    ModelRegistryProtocol,
    RankerProtocol,
    ValidatorProtocol,
)
from src.models.registry import FakeModelRegistry, ModelRegistry

__all__ = [
    # Extractors
    "CodeT5Extractor",
    "TermExtractionResult",
    # Validators
    "GraphCodeBERTValidator",
    "ValidatedTerm",
    "ValidationResult",
    # Rankers
    "CodeBERTRanker",
    "RankedTerm",
    "RankingResult",
    # Protocols
    "ExtractorProtocol",
    "ValidatorProtocol",
    "RankerProtocol",
    "ModelRegistryProtocol",
    # Registry
    "ModelRegistry",
    "FakeModelRegistry",
]
