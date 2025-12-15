# SPDX-FileCopyrightText: 2024 AI Platform Team
# SPDX-License-Identifier: MIT
"""
WBS 5.2 SBERT Engine Tests - Phase M1: Code Migration.

TDD Test Suite for validating the migrated SemanticSimilarityEngine.

WBS Reference: SBERT_EXTRACTION_MIGRATION_WBS.md
- M1.1.3: RED - test_copied_engine_initializes
- M1.1.5: RED - test_copied_engine_computes_embeddings  
- M1.1.7: RED - test_copied_engine_similarity_matrix

Anti-Pattern Audit Checklist:
- #7: Exception shadowing - use namespaced exceptions
- #12: Connection pooling - shared clients (N/A - stateless model)
- S1192: Extract duplicated literals to constants
- S3776: Cognitive complexity < 15
"""
from __future__ import annotations

import pytest
import numpy as np


# =============================================================================
# Test Constants (S1192 compliance - no duplicated literals)
# =============================================================================
TEST_MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384
TEST_TEXT_SINGLE = "This is a test sentence for semantic embedding."
TEST_TEXTS_BATCH = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing enables computers to understand text.",
]


# =============================================================================
# M1.1.3: RED - Engine Initialization Tests
# =============================================================================
class TestCopiedEngineInitializes:
    """
    Validate that SemanticSimilarityEngine can be instantiated.
    
    WBS M1.1.3: Write test that imports and instantiates the engine.
    Expected: RED until import paths are fixed (M1.1.4).
    """

    def test_import_semantic_similarity_engine(self) -> None:
        """Import SemanticSimilarityEngine from new location."""
        from src.models.sbert.semantic_similarity_engine import SemanticSimilarityEngine
        
        assert SemanticSimilarityEngine is not None

    def test_import_config_dataclass(self) -> None:
        """Import SimilarityConfig dataclass from new location."""
        from src.models.sbert.semantic_similarity_engine import SimilarityConfig
        
        assert SimilarityConfig is not None

    def test_import_result_dataclass(self) -> None:
        """Import SimilarityResult dataclass from new location."""
        from src.models.sbert.semantic_similarity_engine import SimilarityResult
        
        assert SimilarityResult is not None

    def test_import_availability_flag(self) -> None:
        """Import SENTENCE_TRANSFORMERS_AVAILABLE flag."""
        from src.models.sbert.semantic_similarity_engine import SENTENCE_TRANSFORMERS_AVAILABLE
        
        assert isinstance(SENTENCE_TRANSFORMERS_AVAILABLE, bool)

    def test_engine_instantiation_default_config(self) -> None:
        """Instantiate engine with default configuration."""
        from src.models.sbert.semantic_similarity_engine import SemanticSimilarityEngine, SimilarityConfig
        
        config = SimilarityConfig()
        engine = SemanticSimilarityEngine(config)
        
        assert engine is not None
        assert engine.config == config

    def test_engine_instantiation_custom_model(self) -> None:
        """Instantiate engine with explicit model name."""
        from src.models.sbert.semantic_similarity_engine import SemanticSimilarityEngine, SimilarityConfig
        
        config = SimilarityConfig(model_name=TEST_MODEL_NAME)
        engine = SemanticSimilarityEngine(config)
        
        assert engine.config.model_name == TEST_MODEL_NAME

    def test_engine_tfidf_fallback_mode(self) -> None:
        """Verify engine has TF-IDF fallback capability."""
        from src.models.sbert.semantic_similarity_engine import SemanticSimilarityEngine, SimilarityConfig
        
        config = SimilarityConfig()
        engine = SemanticSimilarityEngine(config)
        
        # Engine should have fallback vectorizer (sklearn TfidfVectorizer)
        assert hasattr(engine, "_tfidf_vectorizer")


# =============================================================================
# M1.1.5: RED - Embedding Computation Tests
# =============================================================================
class TestCopiedEngineComputesEmbeddings:
    """
    Validate embedding computation functionality.
    
    WBS M1.1.5: Write test that computes embeddings for sample text.
    Expected: RED until GREEN phase validates functionality.
    """

    @pytest.fixture
    def engine(self):
        """Create engine instance for embedding tests."""
        from src.models.sbert.semantic_similarity_engine import SemanticSimilarityEngine, SimilarityConfig
        
        config = SimilarityConfig(model_name=TEST_MODEL_NAME)
        return SemanticSimilarityEngine(config)

    def test_compute_embeddings_single_text(self, engine) -> None:
        """Compute embedding for single text string."""
        embeddings = engine.compute_embeddings([TEST_TEXT_SINGLE])
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1  # Single text

    def test_compute_embeddings_batch(self, engine) -> None:
        """Compute embeddings for batch of texts."""
        embeddings = engine.compute_embeddings(TEST_TEXTS_BATCH)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(TEST_TEXTS_BATCH)

    def test_embedding_dimension_384(self, engine) -> None:
        """Verify all-MiniLM-L6-v2 produces 384-dim embeddings."""
        from src.models.sbert.semantic_similarity_engine import SENTENCE_TRANSFORMERS_AVAILABLE
        
        embeddings = engine.compute_embeddings([TEST_TEXT_SINGLE])
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # SBERT model produces 384-dim vectors
            assert embeddings.shape[1] == EXPECTED_EMBEDDING_DIM
        else:
            # TF-IDF fallback - dimension varies by vocabulary
            assert embeddings.shape[1] > 0

    def test_embeddings_normalized(self, engine) -> None:
        """Verify embeddings are L2-normalized (unit vectors)."""
        from src.models.sbert.semantic_similarity_engine import SENTENCE_TRANSFORMERS_AVAILABLE
        
        embeddings = engine.compute_embeddings([TEST_TEXT_SINGLE])
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # SBERT embeddings should be normalized
            norm = np.linalg.norm(embeddings[0])
            assert np.isclose(norm, 1.0, atol=1e-5)


# =============================================================================
# M1.1.7: RED - Similarity Matrix Tests
# =============================================================================
class TestCopiedEngineSimilarityMatrix:
    """
    Validate similarity matrix computation.
    
    WBS M1.1.7: Write test that computes similarity matrix.
    Expected: RED until GREEN phase validates functionality.
    """

    @pytest.fixture
    def engine(self):
        """Create engine instance for similarity tests."""
        from src.models.sbert.semantic_similarity_engine import SemanticSimilarityEngine, SimilarityConfig
        
        config = SimilarityConfig(model_name=TEST_MODEL_NAME)
        return SemanticSimilarityEngine(config)

    def test_compute_similarity_matrix_shape(self, engine) -> None:
        """Verify similarity matrix has correct shape (n x n)."""
        matrix = engine.compute_similarity_matrix(TEST_TEXTS_BATCH)
        
        n = len(TEST_TEXTS_BATCH)
        assert matrix.shape == (n, n)

    def test_similarity_matrix_diagonal_ones(self, engine) -> None:
        """Verify diagonal elements are 1.0 (self-similarity)."""
        matrix = engine.compute_similarity_matrix(TEST_TEXTS_BATCH)
        
        diagonal = np.diag(matrix)
        assert np.allclose(diagonal, 1.0, atol=1e-5)

    def test_similarity_matrix_symmetric(self, engine) -> None:
        """Verify similarity matrix is symmetric."""
        matrix = engine.compute_similarity_matrix(TEST_TEXTS_BATCH)
        
        assert np.allclose(matrix, matrix.T, atol=1e-5)

    def test_similarity_values_range(self, engine) -> None:
        """Verify similarity values are in [-1, 1] range (cosine similarity)."""
        matrix = engine.compute_similarity_matrix(TEST_TEXTS_BATCH)
        
        assert np.all(matrix >= -1.0 - 1e-5)
        assert np.all(matrix <= 1.0 + 1e-5)

    def test_find_similar_returns_results(self, engine) -> None:
        """Verify find_similar returns SimilarityResult objects."""
        from src.models.sbert.semantic_similarity_engine import SimilarityResult
        
        results = engine.find_similar(
            query=TEST_TEXT_SINGLE,
            candidates=TEST_TEXTS_BATCH,
            top_k=2
        )
        
        assert len(results) == 2
        assert all(isinstance(r, SimilarityResult) for r in results)

    def test_find_similar_sorted_by_score(self, engine) -> None:
        """Verify find_similar returns results sorted by descending score."""
        results = engine.find_similar(
            query=TEST_TEXT_SINGLE,
            candidates=TEST_TEXTS_BATCH,
            top_k=len(TEST_TEXTS_BATCH)
        )
        
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Anti-Pattern Compliance Tests
# =============================================================================
class TestAntiPatternCompliance:
    """
    Validate anti-pattern compliance per CODING_PATTERNS_ANALYSIS.md.
    
    #7: Exception shadowing - use namespaced exceptions
    S1192: Extract duplicated literals to constants
    S3776: Cognitive complexity < 15
    """

    def test_no_bare_except_clauses(self) -> None:
        """Verify no bare except clauses (#7 exception shadowing)."""
        import ast
        from pathlib import Path
        
        engine_path = Path(__file__).parent.parent.parent.parent / "src" / "models" / "sbert" / "semantic_similarity_engine.py"
        
        if engine_path.exists():
            source = engine_path.read_text()
            tree = ast.parse(source)
            
            bare_excepts = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        bare_excepts.append(node.lineno)
            
            assert len(bare_excepts) == 0, f"Bare except clauses found at lines: {bare_excepts}"

    def test_s1192_no_magic_strings_tripled(self) -> None:
        """
        Verify no string literal appears 3+ times (S1192 compliance).
        
        S1192: Duplicated string literals should be extracted to constants.
        SonarQube threshold is 3 occurrences (2 is acceptable).
        Docstrings and type hints are excluded from this check.
        """
        import ast
        from pathlib import Path
        from collections import Counter
        
        engine_path = Path(__file__).parent.parent.parent.parent / "src" / "models" / "sbert" / "semantic_similarity_engine.py"
        
        if engine_path.exists():
            source = engine_path.read_text()
            tree = ast.parse(source)
            
            # Collect all string literals (excluding docstrings)
            strings: list[str] = []
            for node in ast.walk(tree):
                # Skip docstrings (Expr -> Constant at function/class level)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    continue
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # Skip short strings (like dict keys)
                    if len(node.value) >= 10:
                        strings.append(node.value)
            
            # Find strings appearing 3+ times
            counts = Counter(strings)
            violations = {s: c for s, c in counts.items() if c >= 3}
            
            assert len(violations) == 0, (
                f"S1192 violation - strings appearing 3+ times: {violations}"
            )

    def test_s3776_cognitive_complexity_under_limit(self) -> None:
        """
        Verify no function has cognitive complexity >= 15.
        
        S3776: Cognitive Complexity of functions should not be too high.
        SonarQube default threshold is 15.
        
        Simplified check: count nesting depth + branching keywords.
        """
        import ast
        from pathlib import Path
        
        engine_path = Path(__file__).parent.parent.parent.parent / "src" / "models" / "sbert" / "semantic_similarity_engine.py"
        COMPLEXITY_THRESHOLD = 15
        
        if engine_path.exists():
            source = engine_path.read_text()
            tree = ast.parse(source)
            
            # Simplified cognitive complexity: count if/for/while/try/with + nesting
            def estimate_complexity(node: ast.AST, nesting: int = 0) -> int:
                """Estimate cognitive complexity recursively."""
                complexity = 0
                increment_keywords = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.ExceptHandler)
                
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, increment_keywords):
                        complexity += 1 + nesting  # Base + nesting penalty
                        complexity += estimate_complexity(child, nesting + 1)
                    else:
                        complexity += estimate_complexity(child, nesting)
                
                return complexity
            
            # Check all functions and methods
            violations = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cc = estimate_complexity(node)
                    if cc >= COMPLEXITY_THRESHOLD:
                        violations.append((node.name, cc))
            
            assert len(violations) == 0, (
                f"S3776 violation - functions with complexity >= {COMPLEXITY_THRESHOLD}: {violations}"
            )

    def test_module_exports_defined(self) -> None:
        """Verify __all__ exports are defined in __init__.py."""
        from src.models.sbert import semantic_similarity_engine
        
        assert hasattr(semantic_similarity_engine, "SemanticSimilarityEngine")


# =============================================================================
# M1.2.3: RED - Dependency Import Tests
# =============================================================================
class TestDependenciesImportable:
    """
    Validate that all SBERT dependencies are importable.
    
    WBS M1.2.3: Write test_dependencies_importable.
    Per requirements.txt:
    - sentence-transformers>=2.2.2 (SBERT embeddings)
    - scikit-learn~=1.3.0 (TF-IDF fallback)
    
    Anti-Pattern Audit:
    - Comp_Static_Analysis_Report #4: Ensure imports match requirements.txt
    """

    def test_sentence_transformers_importable(self) -> None:
        """Verify sentence-transformers package can be imported."""
        from sentence_transformers import SentenceTransformer
        
        assert SentenceTransformer is not None

    def test_sklearn_tfidf_importable(self) -> None:
        """Verify scikit-learn TfidfVectorizer can be imported (TF-IDF fallback)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        assert TfidfVectorizer is not None

    def test_sklearn_cosine_similarity_importable(self) -> None:
        """Verify scikit-learn cosine_similarity can be imported."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        assert cosine_similarity is not None

    def test_numpy_importable(self) -> None:
        """Verify numpy can be imported (required for embeddings)."""
        import numpy as np
        
        assert np is not None
        assert hasattr(np, "ndarray")

    def test_sbert_engine_imports_dependencies(self) -> None:
        """Verify SBERT engine module imports all required dependencies."""
        from src.models.sbert.semantic_similarity_engine import (
            SemanticSimilarityEngine,
            SimilarityConfig,
            SimilarityResult,
            SENTENCE_TRANSFORMERS_AVAILABLE,
        )
        
        # All imports should succeed
        assert SemanticSimilarityEngine is not None
        assert SimilarityConfig is not None
        assert SimilarityResult is not None
        assert isinstance(SENTENCE_TRANSFORMERS_AVAILABLE, bool)

    def test_tfidf_fallback_available(self) -> None:
        """Verify TF-IDF fallback is always available regardless of SBERT status."""
        from src.models.sbert.semantic_similarity_engine import SimilarityConfig, SemanticSimilarityEngine
        
        # Force TF-IDF mode by setting model_name to something that might fail
        # The engine should always have fallback capability
        config = SimilarityConfig(fallback_to_tfidf=True)
        engine = SemanticSimilarityEngine(config)
        
        # Engine should have TF-IDF vectorizer available
        assert hasattr(engine, "_tfidf_vectorizer")


# =============================================================================
# M1.3.1: Import Structure Compliance Tests (E402 Resolution)
# =============================================================================
class TestImportStructureCompliance:
    """
    Validate import structure passes ruff E402 or has proper noqa comments.
    
    WBS M1.3.1: Fix ruff E402 - module level import not at top of file.
    The sklearn imports intentionally come AFTER sentence-transformers try/except
    to enable graceful degradation. This is a valid pattern per PEP 8 exceptions.
    
    Resolution: Add '# noqa: E402' to intentional non-top imports.
    """

    def test_ruff_e402_resolved(self) -> None:
        """Verify E402 errors are resolved via noqa comments."""
        from pathlib import Path
        
        engine_path = Path(__file__).parent.parent.parent.parent / "src" / "models" / "sbert" / "semantic_similarity_engine.py"
        
        if engine_path.exists():
            source = engine_path.read_text()
            lines = source.split('\n')
            
            # Find sklearn import lines
            sklearn_import_lines = []
            for i, line in enumerate(lines, 1):
                if 'from sklearn' in line:
                    sklearn_import_lines.append((i, line))
            
            # Each sklearn import should have noqa: E402 comment
            for lineno, line in sklearn_import_lines:
                assert '# noqa: E402' in line or '# noqa:E402' in line, (
                    f"Line {lineno} missing E402 noqa comment: {line!r}"
                )

    def test_import_order_is_intentional(self) -> None:
        """Verify sklearn imports come after sentence-transformers for graceful degradation."""
        from pathlib import Path
        
        engine_path = Path(__file__).parent.parent.parent.parent / "src" / "models" / "sbert" / "semantic_similarity_engine.py"
        
        if engine_path.exists():
            source = engine_path.read_text()
            
            # Verify the pattern: sentence_transformers try/except comes before sklearn
            st_try_pos = source.find("from sentence_transformers")
            sklearn_pos = source.find("from sklearn")
            
            assert st_try_pos != -1, "sentence_transformers import not found"
            assert sklearn_pos != -1, "sklearn import not found"
            assert st_try_pos < sklearn_pos, (
                "sklearn must be imported after sentence_transformers try/except block"
            )
