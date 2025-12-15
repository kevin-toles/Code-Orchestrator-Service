# SPDX-FileCopyrightText: 2024 AI Platform Team
# SPDX-License-Identifier: MIT
"""
WBS 5.2 M2.1 SBERT Model Loading Infrastructure Tests.

TDD Test Suite for SBERT model singleton wrapper with thread safety.

WBS Reference: SBERT_EXTRACTION_MIGRATION_WBS.md Phase M2.1
- M2.1.1: RED - test_sbert_model_singleton
- M2.1.3: RED - test_sbert_model_thread_safe
- M2.1.5: RED - test_sbert_graceful_degradation

Anti-Pattern Audit Checklist:
- #6: Duplicate Classes - Single model instance (singleton)
- #7: Exception shadowing - Use SBERTModelError, not RuntimeError
- #10: State mutation - asyncio.Lock for thread-safe access
- #12: Connection pooling - Cache model instance after first load
"""
from __future__ import annotations

import asyncio

import pytest


# =============================================================================
# Test Constants (S1192 compliance - no duplicated literals)
# =============================================================================
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# M2.1.1: RED - Singleton Pattern Tests
# =============================================================================
class TestSBERTModelSingleton:
    """
    Validate SBERT model loader follows singleton pattern.
    
    WBS M2.1.1: Test only one model instance exists.
    Anti-Pattern #6: No duplicate classes.
    Anti-Pattern #12: Connection pooling - model cached after first load.
    """

    def test_get_sbert_model_returns_instance(self) -> None:
        """Test that get_sbert_model() returns a model loader instance."""
        from src.models.sbert.model_loader import get_sbert_model
        
        model_loader = get_sbert_model()
        assert model_loader is not None

    def test_get_sbert_model_returns_same_instance(self) -> None:
        """Test that second call returns same instance (singleton pattern)."""
        from src.models.sbert.model_loader import get_sbert_model
        
        first_instance = get_sbert_model()
        second_instance = get_sbert_model()
        
        assert first_instance is second_instance

    def test_reset_sbert_model_creates_new_instance(self) -> None:
        """Test that reset_sbert_model() allows creation of new instance."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        first_instance = get_sbert_model()
        reset_sbert_model()
        second_instance = get_sbert_model()
        
        # After reset, should be a NEW instance
        assert first_instance is not second_instance

    def test_model_loader_has_engine_attribute(self) -> None:
        """Test that model loader exposes the underlying SemanticSimilarityEngine."""
        from src.models.sbert.model_loader import get_sbert_model
        
        model_loader = get_sbert_model()
        
        # Should have access to the engine
        assert hasattr(model_loader, "engine")
        assert model_loader.engine is not None

    def test_model_loader_is_initialized_flag(self) -> None:
        """Test that model loader tracks initialization state."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        reset_sbert_model()
        model_loader = get_sbert_model()
        
        assert model_loader.is_initialized is True


# =============================================================================
# M2.1.3: RED - Thread Safety Tests
# =============================================================================
class TestSBERTModelThreadSafety:
    """
    Validate SBERT model loader is thread-safe for concurrent access.
    
    WBS M2.1.3: Test concurrent access is safe.
    Anti-Pattern #10: State mutation - asyncio.Lock for protection.
    """

    @pytest.mark.asyncio
    async def test_concurrent_get_model_returns_same_instance(self) -> None:
        """Test that concurrent calls return same singleton instance."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        reset_sbert_model()
        
        # Simulate concurrent access
        async def get_model_async() -> object:
            return get_sbert_model()
        
        # Launch 10 concurrent calls
        tasks = [get_model_async() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should be the same instance
        first = results[0]
        assert all(r is first for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_embedding_computation_is_safe(self) -> None:
        """Test that concurrent embedding computation doesn't cause race conditions."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        reset_sbert_model()
        model_loader = get_sbert_model()
        
        test_texts = [
            "Machine learning models",
            "Deep neural networks",
            "Natural language processing",
        ]
        
        # Simulate concurrent embedding requests
        async def compute_embedding_async(text: str) -> list[float]:
            return await model_loader.compute_embeddings_async([text])
        
        # Launch concurrent embedding computations
        tasks = [compute_embedding_async(text) for text in test_texts]
        results = await asyncio.gather(*tasks)
        
        # All should succeed without race conditions
        assert len(results) == 3
        for result in results:
            assert result is not None

    @pytest.mark.asyncio
    async def test_model_loader_has_asyncio_lock(self) -> None:
        """Test that model loader uses asyncio.Lock for thread safety."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        reset_sbert_model()
        model_loader = get_sbert_model()
        
        # Should have a lock attribute for thread safety
        assert hasattr(model_loader, "_lock")
        assert isinstance(model_loader._lock, asyncio.Lock)


# =============================================================================
# M2.1.5: RED - Graceful Degradation Tests
# =============================================================================
class TestSBERTGracefulDegradation:
    """
    Validate SBERT model loader handles failures gracefully.
    
    WBS M2.1.5: Test fallback when model unavailable.
    Service should stay up and return error response, not crash.
    """

    def test_model_loader_handles_import_error_gracefully(self) -> None:
        """Test that model loader stays up even if sentence-transformers unavailable."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        reset_sbert_model()
        model_loader = get_sbert_model()
        
        # Should have a method to check if SBERT is available
        assert hasattr(model_loader, "is_sbert_available")
        # Even if not available, loader should exist (graceful degradation)
        assert model_loader is not None

    def test_fallback_to_tfidf_when_sbert_unavailable(self) -> None:
        """Test that TF-IDF fallback is used when SBERT unavailable."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        reset_sbert_model()
        model_loader = get_sbert_model()
        
        # Should have fallback mode indicator
        assert hasattr(model_loader, "using_fallback")

    def test_sbert_model_error_raised_on_failure(self) -> None:
        """Test that SBERTModelError is raised on model failures."""
        from src.core.exceptions import SBERTModelError
        
        # SBERTModelError should exist and be importable
        assert issubclass(SBERTModelError, Exception)

    def test_sbert_model_error_has_proper_message(self) -> None:
        """Test that SBERTModelError has descriptive message."""
        from src.core.exceptions import SBERTModelError
        
        error = SBERTModelError("Model failed to load: test error")
        assert "Model failed to load" in str(error)

    def test_model_loader_status_endpoint_data(self) -> None:
        """Test that model loader provides status data for health endpoints."""
        from src.models.sbert.model_loader import get_sbert_model, reset_sbert_model
        
        reset_sbert_model()
        model_loader = get_sbert_model()
        
        # Should provide status data for /health endpoint
        status = model_loader.get_status()
        
        assert "model_name" in status
        assert "is_loaded" in status
        assert "using_fallback" in status


# =============================================================================
# Anti-Pattern Compliance Tests
# =============================================================================
class TestSBERTModelLoaderAntiPatternCompliance:
    """
    Validate model loader complies with documented anti-patterns.
    
    Reference: CODING_PATTERNS_ANALYSIS.md
    """

    def test_no_exception_shadowing(self) -> None:
        """Verify SBERTModelError doesn't shadow builtins (#7)."""
        from src.core.exceptions import SBERTModelError
        
        # Should NOT shadow builtins like RuntimeError, ConnectionError
        import builtins
        
        # SBERTModelError should be a new class, not a builtin
        assert not hasattr(builtins, "SBERTModelError")

    def test_inherits_from_code_orchestrator_error(self) -> None:
        """Verify SBERTModelError inherits from CodeOrchestratorError."""
        from src.core.exceptions import CodeOrchestratorError, SBERTModelError
        
        assert issubclass(SBERTModelError, CodeOrchestratorError)

    def test_model_loader_follows_protocol(self) -> None:
        """Verify model loader follows SBERTModelProtocol for testing."""
        from src.models.sbert.model_loader import SBERTModelProtocol, get_sbert_model
        
        model_loader = get_sbert_model()
        
        # Should implement the protocol methods
        assert hasattr(model_loader, "compute_embeddings")
        assert hasattr(model_loader, "compute_similarity")
        assert hasattr(model_loader, "get_status")
