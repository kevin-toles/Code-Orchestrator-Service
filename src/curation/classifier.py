"""
WBS 4.2: Domain Classifier

Classifies books into domains (ai-ml, systems, etc.) for filtering.
"""


class DomainClassifier:
    """Classifies books into domains.

    Uses simple keyword matching for domain classification.
    Can be extended to use ML-based classification.
    """

    # Domain keyword mappings
    AI_ML_KEYWORDS = [
        "ai",
        "llm",
        "machine learning",
        "deep learning",
        "neural",
        "transformer",
        "embedding",
        "rag",
        "nlp",
        "python",
        "architecture patterns",
    ]

    SYSTEMS_KEYWORDS = [
        "c++",
        "concurrency",
        "systems programming",
        "memory",
        "operating system",
        "kernel",
        "low-level",
    ]

    def classify(self, book_title: str) -> str:
        """Classify a book into a domain.

        Args:
            book_title: The book title to classify

        Returns:
            Domain string: 'ai-ml', 'systems', or 'general'
        """
        title_lower = book_title.lower()

        # Check AI/ML keywords
        for keyword in self.AI_ML_KEYWORDS:
            if keyword in title_lower:
                return "ai-ml"

        # Check systems keywords
        for keyword in self.SYSTEMS_KEYWORDS:
            if keyword in title_lower:
                return "systems"

        return "general"

    def matches_domain(self, book_title: str, target_domain: str) -> bool:
        """Check if a book matches a target domain.

        Args:
            book_title: The book title to check
            target_domain: The target domain to match

        Returns:
            True if book matches domain, False otherwise
        """
        book_domain = self.classify(book_title)

        # Exact match
        if book_domain == target_domain:
            return True

        # General domain matches everything
        return book_domain == "general"
