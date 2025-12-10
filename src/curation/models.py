"""
WBS 4.2: Curation Models

Data models for result curation.
"""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """A search result with book, chapter, score, and content.

    Attributes:
        book: The book title
        chapter: Chapter number
        score: Raw relevance score from search
        content: Matched content snippet
    """

    book: str
    chapter: int
    score: float
    content: str = ""

    @property
    def relevance_score(self) -> float:
        """Alias for score (per WBS response schema)."""
        return self.score
