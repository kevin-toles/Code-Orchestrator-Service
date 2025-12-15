"""SBERT (Sentence-BERT) model wrapper for semantic similarity.

This module provides the SemanticSimilarityEngine for computing semantic
similarity between text using Sentence Transformers (all-MiniLM-L6-v2).

Migrated from: llm-document-enhancer/workflows/metadata_enrichment/scripts/
Per: SBERT_EXTRACTION_MIGRATION_WBS.md Phase M1 (Kitchen Brigade Architecture)

Role in Kitchen Brigade:
- SBERT lives in Code-Orchestrator-Service (Sous Chef)
- Translates NL requirements from LLM Gateway
- Computes similar_chapters for cross-referencing
- Provides embeddings for semantic search

Patterns applied from CODING_PATTERNS_ANALYSIS.md:
- Service Layer Pattern (Architecture Patterns Ch. 4)
- Graceful degradation with TF-IDF fallback
- Singleton model loading with caching
"""

from src.models.sbert.semantic_similarity_engine import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    SemanticSimilarityEngine,
    SimilarityConfig,
    SimilarityResult,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "EMBEDDING_DIMENSIONS",
    "SemanticSimilarityEngine",
    "SimilarityConfig",
    "SimilarityResult",
]
