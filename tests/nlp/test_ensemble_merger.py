"""Tests for EnsembleMerger - HCE-2.21 through HCE-2.25.

TDD RED Phase: Tests written before implementation.
These tests MUST FAIL initially (no implementation exists).

WBS Reference: HCE-2.21 through HCE-2.27
AC Reference: AC-2.7

Anti-Patterns Avoided:
- S1192: Constants extracted to module level
- S3776: Small, focused test methods
"""

from __future__ import annotations

from typing import Final

import pytest

# =============================================================================
# Module Constants (S1192 compliance)
# =============================================================================

DEFAULT_TEXTRANK_SCORE: Final[float] = 0.5


# =============================================================================
# HCE-2.21: Test EnsembleMerger can be imported
# =============================================================================


class TestEnsembleMergerImport:
    """HCE-2.21: Test EnsembleMerger can be imported."""

    def test_ensemble_merger_can_be_imported(self) -> None:
        """AC-2.7: EnsembleMerger class exists and can be imported."""
        from src.nlp.ensemble_merger import EnsembleMerger

        assert EnsembleMerger is not None

    def test_extracted_term_can_be_imported(self) -> None:
        """AC-2.7: ExtractedTerm dataclass can be imported."""
        from src.nlp.ensemble_merger import ExtractedTerm

        assert ExtractedTerm is not None


# =============================================================================
# HCE-2.22: Test merge() returns List[ExtractedTerm]
# =============================================================================


class TestEnsembleMergerMerge:
    """HCE-2.22: Test merge() returns List[ExtractedTerm]."""

    def test_merge_returns_list(self) -> None:
        """AC-2.7: merge() returns a list."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms = [("machine learning", 0.02)]
        textrank_terms = ["deep learning"]

        result = merger.merge(yake_terms, textrank_terms)

        assert isinstance(result, list)

    def test_merge_returns_extracted_terms(self) -> None:
        """AC-2.7: merge() returns list of ExtractedTerm objects."""
        from src.nlp.ensemble_merger import EnsembleMerger, ExtractedTerm

        merger = EnsembleMerger()
        yake_terms = [("machine learning", 0.02)]
        textrank_terms = ["deep learning"]

        result = merger.merge(yake_terms, textrank_terms)

        assert len(result) > 0
        assert all(isinstance(item, ExtractedTerm) for item in result)

    def test_extracted_term_has_required_fields(self) -> None:
        """AC-2.7: ExtractedTerm has term, score, source fields."""
        from src.nlp.ensemble_merger import ExtractedTerm

        term = ExtractedTerm(term="machine learning", score=0.02, source="yake")

        assert hasattr(term, "term")
        assert hasattr(term, "score")
        assert hasattr(term, "source")


# =============================================================================
# HCE-2.23: Test merge() preserves YAKE scores
# =============================================================================


class TestEnsembleMergerYAKEScores:
    """HCE-2.23: Test merge() preserves YAKE scores."""

    def test_merge_preserves_yake_scores(self) -> None:
        """AC-2.7: YAKE terms retain their original scores."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms = [("machine learning", 0.02), ("neural network", 0.05)]
        textrank_terms: list[str] = []

        result = merger.merge(yake_terms, textrank_terms)

        # Find the terms in result
        ml_term = next((t for t in result if t.term == "machine learning"), None)
        nn_term = next((t for t in result if t.term == "neural network"), None)

        assert ml_term is not None
        assert ml_term.score == 0.02

        assert nn_term is not None
        assert nn_term.score == 0.05


# =============================================================================
# HCE-2.24: Test merge() assigns default score to TextRank
# =============================================================================


class TestEnsembleMergerTextRankScores:
    """HCE-2.24: Test merge() assigns default score to TextRank."""

    def test_merge_assigns_default_score_to_textrank(self) -> None:
        """AC-2.7: TextRank terms get default score of 0.5."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms: list[tuple[str, float]] = []
        textrank_terms = ["deep learning", "artificial intelligence"]

        result = merger.merge(yake_terms, textrank_terms)

        for term_obj in result:
            assert term_obj.score == DEFAULT_TEXTRANK_SCORE


# =============================================================================
# HCE-2.25: Test merge() tracks source (yake/textrank/both)
# =============================================================================


class TestEnsembleMergerSourceTracking:
    """HCE-2.25: Test merge() tracks source (yake/textrank/both)."""

    def test_merge_tracks_yake_source(self) -> None:
        """AC-2.7: Terms only from YAKE have source='yake'."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms = [("machine learning", 0.02)]
        textrank_terms: list[str] = []

        result = merger.merge(yake_terms, textrank_terms)

        ml_term = next((t for t in result if t.term == "machine learning"), None)
        assert ml_term is not None
        assert ml_term.source == "yake"

    def test_merge_tracks_textrank_source(self) -> None:
        """AC-2.7: Terms only from TextRank have source='textrank'."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms: list[tuple[str, float]] = []
        textrank_terms = ["deep learning"]

        result = merger.merge(yake_terms, textrank_terms)

        dl_term = next((t for t in result if t.term == "deep learning"), None)
        assert dl_term is not None
        assert dl_term.source == "textrank"

    def test_merge_tracks_both_source(self) -> None:
        """AC-2.7: Terms from both extractors have source='both'."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms = [("machine learning", 0.02)]
        textrank_terms = ["machine learning"]  # Same term

        result = merger.merge(yake_terms, textrank_terms)

        ml_term = next((t for t in result if t.term == "machine learning"), None)
        assert ml_term is not None
        assert ml_term.source == "both"

    def test_merge_both_uses_yake_score(self) -> None:
        """AC-2.7: When source='both', YAKE score is preserved."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms = [("machine learning", 0.02)]
        textrank_terms = ["machine learning"]

        result = merger.merge(yake_terms, textrank_terms)

        ml_term = next((t for t in result if t.term == "machine learning"), None)
        assert ml_term is not None
        assert ml_term.score == 0.02  # YAKE score preserved

    def test_merge_empty_inputs_returns_empty_list(self) -> None:
        """AC-2.7: Empty inputs return empty list."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        result = merger.merge([], [])

        assert isinstance(result, list)
        assert len(result) == 0

    def test_merge_deduplicates_by_lowercase(self) -> None:
        """AC-2.7: Merge deduplicates case-insensitively."""
        from src.nlp.ensemble_merger import EnsembleMerger

        merger = EnsembleMerger()
        yake_terms = [("Machine Learning", 0.02)]
        textrank_terms = ["machine learning"]

        result = merger.merge(yake_terms, textrank_terms)

        # Should have only one entry for "machine learning"
        ml_terms = [t for t in result if t.term.lower() == "machine learning"]
        assert len(ml_terms) == 1
        assert ml_terms[0].source == "both"
