"""
WBS 4.2: Result Curation Module

Curates search results with domain filtering, ranking, and deduplication.
"""

from src.curation.classifier import DomainClassifier
from src.curation.curator import ResultCurator
from src.curation.models import SearchResult

__all__ = [
    "DomainClassifier",
    "ResultCurator",
    "SearchResult",
]
