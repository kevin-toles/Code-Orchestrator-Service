"""
WBS 4.2: Result Curator (Chef de Partie)

Curates search results with domain filtering, ranking, and deduplication.

Patterns Applied:
- Pure functions for curation logic
- Domain classifier for filtering
"""

from src.curation.classifier import DomainClassifier
from src.curation.models import SearchResult


class ResultCurator:
    """Curates search results with filtering, ranking, and deduplication.

    Chef de Partie in the Kitchen Brigade - responsible for result quality.

    Attributes:
        domain_classifier: Classifier for domain filtering
        min_relevance: Minimum relevance score threshold
    """

    def __init__(
        self,
        domain_classifier: DomainClassifier | None = None,
        min_relevance: float = 0.3,
    ) -> None:
        """Initialize the result curator.

        Args:
            domain_classifier: Custom classifier (default: DomainClassifier)
            min_relevance: Minimum relevance score threshold
        """
        self.domain_classifier = domain_classifier or DomainClassifier()
        self.min_relevance = min_relevance

    def curate(
        self,
        results: list[SearchResult],
        query: str,
        domain: str,
    ) -> list[SearchResult]:
        """Curate results: filter by domain, dedupe, rank.

        Full curation pipeline:
        1. Filter by domain
        2. Remove duplicates
        3. Rank by relevance
        4. Filter by minimum score

        Args:
            results: Raw search results
            query: Original query for relevance ranking
            domain: Target domain for filtering

        Returns:
            Curated list of SearchResult
        """
        # Step 1: Filter by domain
        filtered = self.filter_by_domain(results, domain)

        # Step 2: Remove duplicates
        deduped = self.remove_duplicates(filtered)

        # Step 3: Rank by relevance
        ranked = self.rank_by_relevance(deduped, query)

        # Step 4: Filter by minimum score
        final = [r for r in ranked if r.score >= self.min_relevance]

        return final

    def filter_by_domain(
        self,
        results: list[SearchResult],
        domain: str,
    ) -> list[SearchResult]:
        """Filter results to only include matching domain.

        Args:
            results: Search results to filter
            domain: Target domain (e.g., 'ai-ml')

        Returns:
            Filtered results matching domain
        """
        return [
            r
            for r in results
            if self.domain_classifier.matches_domain(r.book, domain)
        ]

    def remove_duplicates(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Remove duplicate results by book+chapter, keeping highest score.

        Args:
            results: Search results with potential duplicates

        Returns:
            Deduplicated results
        """
        # Track best result for each book+chapter
        best_results: dict[tuple[str, int], SearchResult] = {}

        for result in results:
            key = (result.book, result.chapter)
            if key not in best_results or result.score > best_results[key].score:
                best_results[key] = result

        return list(best_results.values())

    def rank_by_relevance(
        self,
        results: list[SearchResult],
        query: str,
        recompute_scores: bool = False,
    ) -> list[SearchResult]:
        """Rank results by relevance score.

        Args:
            results: Search results to rank
            query: Original query for potential re-ranking
            recompute_scores: If True, recompute scores based on query similarity

        Returns:
            Results sorted by score descending
        """
        if recompute_scores:
            # Simple keyword-based re-scoring
            results = self._recompute_scores(results, query)

        return sorted(results, key=lambda r: r.score, reverse=True)

    def _recompute_scores(
        self,
        results: list[SearchResult],
        query: str,
    ) -> list[SearchResult]:
        """Recompute scores based on query keyword overlap.

        Simple TF-like scoring based on query term overlap with content.

        Args:
            results: Results to re-score
            query: Query to match against

        Returns:
            Results with updated scores
        """
        query_terms = set(query.lower().split())
        reranked = []

        for result in results:
            content_terms = set(result.content.lower().split())
            overlap = len(query_terms & content_terms)
            total_terms = len(query_terms) if query_terms else 1

            # Blend original score with keyword overlap
            new_score = (result.score * 0.5) + (overlap / total_terms * 0.5)

            reranked.append(
                SearchResult(
                    book=result.book,
                    chapter=result.chapter,
                    score=new_score,
                    content=result.content,
                )
            )

        return reranked
