"""Model wrappers for code understanding, validation, and ranking.

Models and their purposes:
- CodeT5Extractor: Code generation/completion (NOT for keyword extraction)
- GraphCodeBERTValidator: Filters terms using GraphCodeBERT embeddings
- CodeBERTRanker: Scores and ranks by CodeBERT embedding similarity

Note: For metadata/keyword extraction, use YAKE+TextRank (src/nlp/).
CodeT5+ is a generative model trained on code completion, not extraction.

Patterns applied from CODING_PATTERNS_ANALYSIS.md:
- Singleton pattern for model caching
- Protocol typing for dependency injection
"""

from src.models.codebert_ranker import CodeBERTRanker, RankedTerm, RankingResult
from src.models.codet5_extractor import CodeT5Extractor, ExtractionResult
from src.models.graphcodebert_validator import (
    GraphCodeBERTValidator,
    ValidationResult,
)
from src.models.protocols import (
    ExtractorProtocol,
    RankerProtocol,
    ValidatorProtocol,
)

__all__ = [
    # Code Generation (NOT for keyword extraction)
    "CodeT5Extractor",
    "ExtractionResult",
    # Validators
    "GraphCodeBERTValidator",
    "ValidationResult",
    # Rankers
    "CodeBERTRanker",
    "RankedTerm",
    "RankingResult",
    # Protocols
    "ExtractorProtocol",
    "ValidatorProtocol",
    "RankerProtocol",
]
