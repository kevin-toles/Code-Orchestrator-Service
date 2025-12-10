"""
WBS 4.2: Result Curation Tests (RED Phase)

Tests for ResultCurator (Chef de Partie):
- 4.2.1: Domain filter - Remove results from wrong domain
- 4.2.2: Relevance ranking - Re-rank by semantic similarity
- 4.2.3: Duplicate removal - Dedupe by book+chapter

Patterns Applied (from CODING_PATTERNS_ANALYSIS.md):
- Pure functions for curation logic
- Pydantic models for data validation
"""



# =============================================================================
# WBS 4.2.1: Domain Filter Tests
# =============================================================================


class TestCurationDomainFilter:
    """Tests for domain filtering in ResultCurator."""

    def test_curator_module_exists(self) -> None:
        """ResultCurator module exists."""
        from src.curation.curator import ResultCurator

        assert ResultCurator is not None

    def test_curator_has_filter_by_domain_method(self) -> None:
        """ResultCurator has filter_by_domain() method."""
        from src.curation.curator import ResultCurator

        curator = ResultCurator()
        assert hasattr(curator, "filter_by_domain")

    def test_domain_filter_removes_wrong_domain(self) -> None:
        """Domain filter removes results from wrong domain.

        Per WBS TDD test: C++ Concurrency should be filtered for ai-ml domain
        """
        from src.curation.curator import ResultCurator
        from src.curation.models import SearchResult

        curator = ResultCurator()

        raw_results = [
            SearchResult(book="AI Engineering", chapter=5, score=0.91, content="LLM chunking"),
            SearchResult(book="C++ Concurrency", chapter=3, score=0.45, content="memory chunk"),
            SearchResult(book="Building LLM Apps", chapter=8, score=0.88, content="RAG pipeline"),
        ]

        filtered = curator.filter_by_domain(
            results=raw_results,
            domain="ai-ml",
        )

        # C++ Concurrency should be filtered (wrong domain)
        assert len(filtered) == 2
        assert all(r.book != "C++ Concurrency" for r in filtered)

    def test_domain_filter_keeps_relevant_results(self) -> None:
        """Domain filter keeps results from correct domain."""
        from src.curation.curator import ResultCurator
        from src.curation.models import SearchResult

        curator = ResultCurator()

        raw_results = [
            SearchResult(book="AI Engineering", chapter=5, score=0.91, content="LLM chunking"),
            SearchResult(book="Building LLM Apps", chapter=8, score=0.88, content="RAG pipeline"),
        ]

        filtered = curator.filter_by_domain(
            results=raw_results,
            domain="ai-ml",
        )

        assert len(filtered) == 2

    def test_domain_filter_uses_domain_classifier(self) -> None:
        """Domain filter uses configurable domain classifier."""
        from src.curation.curator import ResultCurator

        curator = ResultCurator()

        # Should have a domain classifier
        assert hasattr(curator, "domain_classifier")


# =============================================================================
# WBS 4.2.2: Relevance Ranking Tests
# =============================================================================


class TestCurationRanking:
    """Tests for relevance ranking in ResultCurator."""

    def test_curator_has_rank_by_relevance_method(self) -> None:
        """ResultCurator has rank_by_relevance() method."""
        from src.curation.curator import ResultCurator

        curator = ResultCurator()
        assert hasattr(curator, "rank_by_relevance")

    def test_ranking_sorts_by_relevance_score(self) -> None:
        """Ranking sorts results by relevance score descending."""
        from src.curation.curator import ResultCurator
        from src.curation.models import SearchResult

        curator = ResultCurator()

        results = [
            SearchResult(book="Book A", chapter=1, score=0.5, content="test"),
            SearchResult(book="Book B", chapter=2, score=0.9, content="test"),
            SearchResult(book="Book C", chapter=3, score=0.7, content="test"),
        ]

        ranked = curator.rank_by_relevance(results, query="test query")

        # Should be sorted by score descending
        assert ranked[0].score == 0.9
        assert ranked[1].score == 0.7
        assert ranked[2].score == 0.5

    def test_ranking_can_recompute_scores(self) -> None:
        """Ranking can recompute scores based on query similarity."""
        from src.curation.curator import ResultCurator
        from src.curation.models import SearchResult

        curator = ResultCurator()

        results = [
            SearchResult(book="AI Engineering", chapter=5, score=0.5, content="LLM document chunking"),
            SearchResult(book="Building LLM Apps", chapter=8, score=0.5, content="network configuration"),
        ]

        ranked = curator.rank_by_relevance(
            results,
            query="LLM document chunking for RAG",
            recompute_scores=True,
        )

        # Result with "LLM document chunking" should rank higher
        assert ranked[0].book == "AI Engineering"


# =============================================================================
# WBS 4.2.3: Duplicate Removal Tests
# =============================================================================


class TestCurationDedup:
    """Tests for duplicate removal in ResultCurator."""

    def test_curator_has_remove_duplicates_method(self) -> None:
        """ResultCurator has remove_duplicates() method."""
        from src.curation.curator import ResultCurator

        curator = ResultCurator()
        assert hasattr(curator, "remove_duplicates")

    def test_dedup_removes_by_book_chapter(self) -> None:
        """Dedup removes duplicates by book+chapter key."""
        from src.curation.curator import ResultCurator
        from src.curation.models import SearchResult

        curator = ResultCurator()

        results = [
            SearchResult(book="AI Engineering", chapter=5, score=0.91, content="content A"),
            SearchResult(book="AI Engineering", chapter=5, score=0.85, content="content B"),  # Duplicate
            SearchResult(book="Building LLM Apps", chapter=8, score=0.88, content="content C"),
        ]

        deduped = curator.remove_duplicates(results)

        assert len(deduped) == 2
        # Should keep higher score
        ai_result = next(r for r in deduped if r.book == "AI Engineering")
        assert ai_result.score == 0.91

    def test_dedup_keeps_highest_score(self) -> None:
        """Dedup keeps the duplicate with highest score."""
        from src.curation.curator import ResultCurator
        from src.curation.models import SearchResult

        curator = ResultCurator()

        results = [
            SearchResult(book="Book A", chapter=1, score=0.5, content="lower"),
            SearchResult(book="Book A", chapter=1, score=0.9, content="higher"),
        ]

        deduped = curator.remove_duplicates(results)

        assert len(deduped) == 1
        assert deduped[0].score == 0.9


# =============================================================================
# Full Curation Pipeline Tests
# =============================================================================


class TestCurationPipeline:
    """Tests for full curation pipeline."""

    def test_curator_has_curate_method(self) -> None:
        """ResultCurator has curate() method for full pipeline."""
        from src.curation.curator import ResultCurator

        curator = ResultCurator()
        assert hasattr(curator, "curate")

    def test_curate_applies_all_steps(self) -> None:
        """curate() applies domain filter, ranking, and dedup."""
        from src.curation.curator import ResultCurator
        from src.curation.models import SearchResult

        curator = ResultCurator()

        raw_results = [
            SearchResult(book="AI Engineering", chapter=5, score=0.91, content="LLM chunking"),
            SearchResult(book="C++ Concurrency", chapter=3, score=0.95, content="memory chunk"),
            SearchResult(book="AI Engineering", chapter=5, score=0.85, content="LLM chunking v2"),
            SearchResult(book="Building LLM Apps", chapter=8, score=0.88, content="RAG pipeline"),
        ]

        curated = curator.curate(
            results=raw_results,
            query="LLM document chunking",
            domain="ai-ml",
        )

        # C++ filtered, duplicate removed
        assert len(curated) == 2
        assert all(r.book != "C++ Concurrency" for r in curated)


# =============================================================================
# Model Tests
# =============================================================================


class TestSearchResultModel:
    """Tests for SearchResult data model."""

    def test_search_result_model_exists(self) -> None:
        """SearchResult model exists."""
        from src.curation.models import SearchResult

        assert SearchResult is not None

    def test_search_result_has_required_fields(self) -> None:
        """SearchResult has book, chapter, score, content fields."""
        from src.curation.models import SearchResult

        result = SearchResult(
            book="Test Book",
            chapter=1,
            score=0.85,
            content="Test content",
        )

        assert result.book == "Test Book"
        assert result.chapter == 1
        assert result.score == 0.85
        assert result.content == "Test content"

    def test_search_result_has_relevance_score_alias(self) -> None:
        """SearchResult has relevance_score property (alias for score)."""
        from src.curation.models import SearchResult

        result = SearchResult(
            book="Test Book",
            chapter=1,
            score=0.85,
            content="Test content",
        )

        # Per WBS: results should have relevance_score >= 0.3
        assert result.relevance_score == 0.85


# =============================================================================
# Domain Classifier Tests
# =============================================================================


class TestDomainClassifier:
    """Tests for DomainClassifier."""

    def test_domain_classifier_exists(self) -> None:
        """DomainClassifier class exists."""
        from src.curation.classifier import DomainClassifier

        assert DomainClassifier is not None

    def test_classifier_identifies_ai_ml_books(self) -> None:
        """Classifier identifies AI/ML books."""
        from src.curation.classifier import DomainClassifier

        classifier = DomainClassifier()

        assert classifier.classify("AI Engineering") == "ai-ml"
        assert classifier.classify("Building LLM Apps") == "ai-ml"
        assert classifier.classify("Building LLM Powered Applications") == "ai-ml"

    def test_classifier_identifies_systems_books(self) -> None:
        """Classifier identifies systems/C++ books."""
        from src.curation.classifier import DomainClassifier

        classifier = DomainClassifier()

        assert classifier.classify("C++ Concurrency") in ["systems", "cpp"]
        assert classifier.classify("C++ Concurrency in Action") in ["systems", "cpp"]

    def test_classifier_matches_domain(self) -> None:
        """Classifier can check if book matches domain."""
        from src.curation.classifier import DomainClassifier

        classifier = DomainClassifier()

        assert classifier.matches_domain("AI Engineering", "ai-ml") is True
        assert classifier.matches_domain("C++ Concurrency", "ai-ml") is False
