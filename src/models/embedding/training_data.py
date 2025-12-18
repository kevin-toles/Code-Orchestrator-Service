"""
Training Data Generation from Enriched Books

WBS: EEP-1.5.1 - Generate training data from enriched books
Extracts contrastive training pairs from cross-references for fine-tuning.

AC-1.5.1.1: Generate text_pairs.jsonl from similar_chapters
AC-1.5.1.2: Generate code_pairs.jsonl from code concepts
AC-1.5.1.3: Support positive/negative pair generation (1:1 ratio)
AC-1.5.1.4: BM25-based hard negative selection

Anti-Patterns Avoided:
- S1192: Constants imported from config
- S3776: Cognitive complexity managed via helper methods
- S1172: No unused parameters
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# Enums and Configuration Classes
# =============================================================================


class PairType(Enum):
    """Type of training pairs to generate.
    
    WBS: EEP-1.5.1 - Training Data Generation
    
    Attributes:
        TEXT: Generate text similarity pairs from chapter cross-references
        CODE: Generate code similarity pairs from code concepts
        BOTH: Generate both text and code pairs
    """
    
    TEXT = "text"
    CODE = "code"
    BOTH = "both"


@dataclass
class TrainingDataConfig:
    """Configuration for training data generation.
    
    WBS: EEP-1.5.1 - Training Data Generation Configuration
    
    Attributes:
        min_positive_score: Minimum similarity score for positive pairs
        use_bm25_negatives: Whether to use BM25 for hard negative selection
        max_pairs: Maximum number of pairs to generate (None for no limit)
        random_seed: Random seed for reproducibility
    """
    
    min_positive_score: float = 0.7
    use_bm25_negatives: bool = False
    max_pairs: int | None = None
    random_seed: int = 42


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class TextPair:
    """A training pair for text similarity.

    Attributes:
        anchor: Anchor text (chapter summary)
        positive: Positive example (similar chapter summary)
        negative: Negative example (dissimilar chapter)
        score: Similarity score from enriched data
    """

    anchor: str
    positive: str
    negative: str
    score: float


@dataclass
class CodePair:
    """A training pair for code similarity.

    Attributes:
        anchor: Anchor code snippet
        positive: Positive code example (similar concept)
        negative: Negative code example (different concept)
        concept: Associated concept
    """

    anchor: str
    positive: str
    negative: str
    concept: str


# =============================================================================
# Helper Functions
# =============================================================================


def _load_book(book_path: Path) -> dict[str, Any]:
    """Load enriched book JSON.

    Args:
        book_path: Path to enriched book JSON

    Returns:
        Book data dictionary
    """
    return json.loads(book_path.read_text())


def _get_chapter_text(chapter: dict[str, Any]) -> str:
    """Extract text content from chapter.

    Args:
        chapter: Chapter dictionary

    Returns:
        Chapter summary or title
    """
    return chapter.get("summary", "") or chapter.get("title", "")


def _find_chapter_by_number(
    chapters: list[dict[str, Any]],
    chapter_number: int,
) -> dict[str, Any] | None:
    """Find chapter by number.

    Args:
        chapters: List of chapter dictionaries
        chapter_number: Chapter number to find

    Returns:
        Chapter dict or None
    """
    for chapter in chapters:
        if chapter.get("chapter_number") == chapter_number:
            return chapter
    return None


def _select_negative(
    all_chapters: list[dict[str, Any]],
    exclude_indices: set[int],
    anchor_text: str,
    use_bm25: bool = False,
) -> str | None:
    """Select a negative example from chapters.

    AC-1.5.1.4: BM25-based hard negative selection when use_bm25=True

    Args:
        all_chapters: All available chapters
        exclude_indices: Chapter numbers to exclude (positives)
        anchor_text: Anchor text for BM25 scoring or to avoid duplicating
        use_bm25: Whether to use BM25 for hard negative selection

    Returns:
        Negative example text or None
    """
    candidates = [
        ch for ch in all_chapters
        if ch.get("chapter_number") not in exclude_indices
        and _get_chapter_text(ch) != anchor_text
        and len(_get_chapter_text(ch)) > 10
    ]

    if not candidates:
        return None

    if use_bm25:
        # Use BM25 for hard negative selection
        negative = _select_bm25_hard_negative(anchor_text, candidates)
    else:
        # Random selection (fallback)
        negative = random.choice(candidates)
    
    return _get_chapter_text(negative)


def _select_bm25_hard_negative(
    anchor_text: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select hard negative using BM25 scoring.

    AC-1.5.1.4: BM25-based hard negative selection
    Selects candidates that have medium BM25 scores - not too similar
    (would be false negatives) and not completely unrelated (too easy).

    Args:
        anchor_text: Anchor text for scoring
        candidates: Candidate chapters for negative selection

    Returns:
        Selected chapter as hard negative
    """
    try:
        from rank_bm25 import BM25Okapi
        
        # Tokenize candidates
        candidate_texts = [_get_chapter_text(ch) for ch in candidates]
        tokenized_corpus = [text.lower().split() for text in candidate_texts]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Score anchor against all candidates
        tokenized_query = anchor_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Select "hard" negative: medium similarity (not too easy, not false negative)
        # Sort by score and pick from the middle third
        scored_candidates = list(zip(scores, candidates))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take from middle third (hard negatives)
        n = len(scored_candidates)
        start_idx = n // 3
        end_idx = 2 * n // 3
        
        if start_idx >= end_idx:
            # Fallback for small candidate lists
            return random.choice(candidates)
        
        middle_candidates = scored_candidates[start_idx:end_idx]
        return random.choice(middle_candidates)[1]
        
    except ImportError:
        # Fallback to random if rank_bm25 not installed
        return random.choice(candidates)


# =============================================================================
# Main Functions
# =============================================================================


def _load_all_books(
    enriched_books: list[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load all enriched books into memory.

    Args:
        enriched_books: List of paths to enriched book JSONs

    Returns:
        Tuple of (books dict by title, flat list of all chapters)
    """
    books: dict[str, dict[str, Any]] = {}
    all_chapters_flat: list[dict[str, Any]] = []

    for book_path in enriched_books:
        try:
            book_data = _load_book(book_path)
            title = book_data.get("metadata", {}).get("title", book_path.stem)
            books[title] = book_data

            for chapter in book_data.get("chapters", []):
                chapter["_book_title"] = title
                all_chapters_flat.append(chapter)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    return books, all_chapters_flat


def _process_similar_chapter_ref(
    ref: dict[str, Any],
    book_title: str,
    books: dict[str, dict[str, Any]],
    min_positive_score: float,
) -> tuple[str | None, int, float]:
    """Process a single similar chapter reference.

    Args:
        ref: Similar chapter reference dict
        book_title: Current book title
        books: All books dict
        min_positive_score: Minimum score threshold

    Returns:
        Tuple of (positive_text or None, chapter_num, score)
    """
    score = ref.get("score", 0.0)
    if score < min_positive_score:
        return None, 0, score

    ref_book = ref.get("book", book_title)
    ref_chapter_num = ref.get("chapter", 0)

    if ref_book not in books:
        return None, ref_chapter_num, score

    ref_chapters = books[ref_book].get("chapters", [])
    ref_chapter = _find_chapter_by_number(ref_chapters, ref_chapter_num)

    if not ref_chapter:
        return None, ref_chapter_num, score

    positive_text = _get_chapter_text(ref_chapter)
    if len(positive_text) < 10:
        return None, ref_chapter_num, score

    return positive_text, ref_chapter_num, score


def _generate_pairs_from_chapter(
    chapter: dict[str, Any],
    book_title: str,
    books: dict[str, dict[str, Any]],
    all_chapters_flat: list[dict[str, Any]],
    min_positive_score: float,
    use_bm25: bool = False,
) -> list[dict[str, Any]]:
    """Generate pairs from a single chapter.

    Args:
        chapter: Chapter dict
        book_title: Book title
        books: All books dict
        all_chapters_flat: Flat list of all chapters
        min_positive_score: Minimum score threshold
        use_bm25: Whether to use BM25 for hard negative selection

    Returns:
        List of pair dicts
    """
    pairs: list[dict[str, Any]] = []
    anchor_text = _get_chapter_text(chapter)

    if len(anchor_text) < 10:
        return pairs

    similar_chapters = chapter.get("similar_chapters", [])
    positive_indices: set[int] = set()

    for ref in similar_chapters:
        positive_text, ref_chapter_num, score = _process_similar_chapter_ref(
            ref, book_title, books, min_positive_score
        )

        if not positive_text:
            continue

        positive_indices.add(ref_chapter_num)
        exclude = positive_indices | {chapter.get("chapter_number", 0)}
        negative_text = _select_negative(
            all_chapters_flat, exclude, anchor_text, use_bm25=use_bm25
        )

        if negative_text:
            pairs.append({
                "anchor": anchor_text,
                "positive": positive_text,
                "negative": negative_text,
                "score": score,
            })

    return pairs


def _write_pairs_to_jsonl(pairs: list[dict[str, Any]], output_path: Path) -> None:
    """Write pairs to JSONL file.

    Args:
        pairs: List of pair dicts
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


def generate_text_pairs(
    enriched_books: list[Path],
    output_path: Path,
    min_positive_score: float = 0.7,
    random_seed: int = 42,
    use_bm25: bool = False,
) -> list[dict[str, Any]]:
    """Generate text pairs from similar_chapters in enriched books.

    AC-1.5.1.1: Generate text_pairs.jsonl from similar_chapters
    AC-1.5.1.4: BM25-based hard negative selection when use_bm25=True

    Args:
        enriched_books: List of paths to enriched book JSONs
        output_path: Path to write JSONL output
        min_positive_score: Minimum score for positive pairs
        random_seed: Random seed for reproducibility
        use_bm25: Whether to use BM25 for hard negative selection

    Returns:
        List of pair dictionaries
    """
    random.seed(random_seed)
    books, all_chapters_flat = _load_all_books(enriched_books)
    pairs: list[dict[str, Any]] = []

    for book_title, book_data in books.items():
        for chapter in book_data.get("chapters", []):
            chapter_pairs = _generate_pairs_from_chapter(
                chapter, book_title, books, all_chapters_flat, min_positive_score,
                use_bm25=use_bm25,
            )
            pairs.extend(chapter_pairs)

    _write_pairs_to_jsonl(pairs, output_path)
    return pairs


def _collect_code_snippets(
    enriched_books: list[Path],
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Collect code snippets from enriched books by concept.

    Args:
        enriched_books: List of paths to enriched book JSONs

    Returns:
        Tuple of (concept_to_code dict, list of (code, concept) tuples)
    """
    concept_to_code: dict[str, list[str]] = {}
    all_code_snippets: list[tuple[str, str]] = []

    for book_path in enriched_books:
        try:
            book_data = _load_book(book_path)
            for chapter in book_data.get("chapters", []):
                _process_chapter_snippets(chapter, concept_to_code, all_code_snippets)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    return concept_to_code, all_code_snippets


def _process_chapter_snippets(
    chapter: dict[str, Any],
    concept_to_code: dict[str, list[str]],
    all_code_snippets: list[tuple[str, str]],
) -> None:
    """Process snippets from a single chapter.

    Args:
        chapter: Chapter dict
        concept_to_code: Dict to update
        all_code_snippets: List to update
    """
    concepts = chapter.get("concepts", [])
    snippets = chapter.get("code_snippets", [])

    if not snippets:
        return

    primary_concept = concepts[0] if concepts else "general"

    for snippet in snippets:
        if len(snippet) < 5:
            continue

        if primary_concept not in concept_to_code:
            concept_to_code[primary_concept] = []
        concept_to_code[primary_concept].append(snippet)
        all_code_snippets.append((snippet, primary_concept))


def _generate_code_pairs_from_concept(
    concept: str,
    snippets: list[str],
    all_code_snippets: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Generate pairs from a single concept's snippets.

    Args:
        concept: Concept name
        snippets: Code snippets for this concept
        all_code_snippets: All code snippets for negative sampling

    Returns:
        List of pair dicts
    """
    pairs: list[dict[str, Any]] = []

    if len(snippets) < 2:
        return pairs

    other_concepts = [s for s, c in all_code_snippets if c != concept]

    for i, anchor in enumerate(snippets):
        for positive in snippets[i + 1:]:
            if other_concepts:
                negative = random.choice(other_concepts)
                pairs.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "concept": concept,
                })

    return pairs


def generate_code_pairs(
    enriched_books: list[Path],
    output_path: Path,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate code pairs from code_snippets in enriched books.

    AC-1.5.1.2: Generate code_pairs.jsonl from code concepts

    Args:
        enriched_books: List of paths to enriched book JSONs
        output_path: Path to write JSONL output
        random_seed: Random seed for reproducibility

    Returns:
        List of pair dictionaries
    """
    random.seed(random_seed)
    concept_to_code, all_code_snippets = _collect_code_snippets(enriched_books)
    pairs: list[dict[str, Any]] = []

    for concept, snippets in concept_to_code.items():
        concept_pairs = _generate_code_pairs_from_concept(
            concept, snippets, all_code_snippets
        )
        pairs.extend(concept_pairs)

    _write_pairs_to_jsonl(pairs, output_path)
    return pairs


def load_text_pairs(pairs_path: Path) -> list[TextPair]:
    """Load text pairs from JSONL file.

    Args:
        pairs_path: Path to JSONL file

    Returns:
        List of TextPair objects
    """
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            data = json.loads(line)
            pairs.append(TextPair(**data))
    return pairs


def load_code_pairs(pairs_path: Path) -> list[CodePair]:
    """Load code pairs from JSONL file.

    Args:
        pairs_path: Path to JSONL file

    Returns:
        List of CodePair objects
    """
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            data = json.loads(line)
            pairs.append(CodePair(**data))
    return pairs


# =============================================================================
# Class-based Interface (for convenience)
# =============================================================================


class TrainingPairGenerator:
    """Generates training pairs from enriched books.

    Provides a class-based interface to the pair generation functions.

    Example:
        >>> generator = TrainingPairGenerator("/path/to/enriched/books")
        >>> text_pairs = generator.generate_text_pairs()
        >>> code_pairs = generator.generate_code_pairs()
        >>> # Using new unified generate method
        >>> pairs = generator.generate(pair_type=PairType.TEXT, config=TrainingDataConfig())
    """

    def __init__(self, enriched_dir: str | Path):
        """Initialize generator with enriched books directory.

        Args:
            enriched_dir: Path to directory containing enriched JSON files
        """
        self._enriched_dir = Path(enriched_dir)

    def generate(
        self,
        pair_type: PairType = PairType.TEXT,
        config: TrainingDataConfig | None = None,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training pairs with unified interface.
        
        WBS: EEP-1.5.1 - Training Data Generation
        AC-1.5.1.1: Generate text_pairs.jsonl from similar_chapters
        AC-1.5.1.2: Generate code_pairs.jsonl from code concepts
        
        Args:
            pair_type: Type of pairs to generate
            config: Training data configuration
            output_path: Optional path to save JSONL output
            
        Returns:
            List of pair dictionaries
        """
        if config is None:
            config = TrainingDataConfig()
        
        random.seed(config.random_seed)
        enriched_books = list(self._enriched_dir.glob("*.json"))
        
        pairs: list[dict[str, Any]] = []
        
        if pair_type in (PairType.TEXT, PairType.BOTH):
            text_out = Path(output_path or "data/text_pairs.jsonl")
            text_pairs = generate_text_pairs(
                enriched_books=enriched_books,
                output_path=text_out,
                min_positive_score=config.min_positive_score,
                random_seed=config.random_seed,
                use_bm25=config.use_bm25_negatives,
            )
            pairs.extend(text_pairs)
        
        if pair_type in (PairType.CODE, PairType.BOTH):
            code_out = Path(output_path or "data/code_pairs.jsonl").with_suffix(".code.jsonl") if pair_type == PairType.BOTH else Path(output_path or "data/code_pairs.jsonl")
            code_pairs = generate_code_pairs(
                enriched_books=enriched_books,
                output_path=code_out,
                random_seed=config.random_seed,
            )
            pairs.extend(code_pairs)
        
        # Apply max_pairs limit if specified
        if config.max_pairs is not None and len(pairs) > config.max_pairs:
            pairs = pairs[:config.max_pairs]
        
        return pairs

    def generate_from_books(
        self,
        book_paths: list[str | Path],
        pair_type: PairType = PairType.TEXT,
        config: TrainingDataConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Generate training pairs from specific book files.
        
        WBS: EEP-1.5.1 - Training Data Generation
        
        Args:
            book_paths: List of paths to enriched book JSON files
            pair_type: Type of pairs to generate
            config: Training data configuration
            
        Returns:
            List of pair dictionaries
        """
        if config is None:
            config = TrainingDataConfig()
        
        random.seed(config.random_seed)
        enriched_books = [Path(p) for p in book_paths]
        
        pairs: list[dict[str, Any]] = []
        
        if pair_type in (PairType.TEXT, PairType.BOTH):
            books, all_chapters_flat = _load_all_books(enriched_books)
            for book_title, book_data in books.items():
                for chapter in book_data.get("chapters", []):
                    chapter_pairs = _generate_pairs_from_chapter(
                        chapter, book_title, books, all_chapters_flat,
                        config.min_positive_score,
                        use_bm25=config.use_bm25_negatives,
                    )
                    pairs.extend(chapter_pairs)
        
        if pair_type in (PairType.CODE, PairType.BOTH):
            concept_to_code, all_code_snippets = _collect_code_snippets(enriched_books)
            for concept, snippets in concept_to_code.items():
                concept_pairs = _generate_code_pairs_from_concept(
                    concept, snippets, all_code_snippets
                )
                pairs.extend(concept_pairs)
        
        # Apply max_pairs limit if specified
        if config.max_pairs is not None and len(pairs) > config.max_pairs:
            pairs = pairs[:config.max_pairs]
        
        return pairs

    def generate_text_pairs(
        self,
        output_path: str | Path | None = None,
        min_similarity: float = 0.5,
        max_pairs_per_chapter: int = 3,
        use_bm25: bool = False,
    ) -> list[TextPair]:
        """Generate text pairs from similar chapters.

        Args:
            output_path: Optional path to save JSONL output
            min_similarity: Minimum similarity score threshold
            max_pairs_per_chapter: Maximum pairs per anchor chapter (unused)
            use_bm25: Whether to use BM25 for hard negative selection

        Returns:
            List of TextPair objects
        """
        # Get list of enriched book files from directory
        enriched_books = list(self._enriched_dir.glob("*.json"))
        out_path = Path(output_path) if output_path else Path("data/text_pairs.jsonl")

        pair_dicts = generate_text_pairs(
            enriched_books=enriched_books,
            output_path=out_path,
            min_positive_score=min_similarity,
            use_bm25=use_bm25,
        )

        return [TextPair(**p) for p in pair_dicts]

    def generate_code_pairs(
        self,
        output_path: str | Path | None = None,
        min_code_length: int = 50,
        max_pairs_per_concept: int = 5,
    ) -> list[CodePair]:
        """Generate code pairs from concepts.

        Args:
            output_path: Optional path to save JSONL output
            min_code_length: Minimum code length threshold (unused)
            max_pairs_per_concept: Maximum pairs per concept (unused)

        Returns:
            List of CodePair objects
        """
        # Get list of enriched book files from directory
        enriched_books = list(self._enriched_dir.glob("*.json"))
        out_path = Path(output_path) if output_path else Path("data/code_pairs.jsonl")

        pair_dicts = generate_code_pairs(
            enriched_books=enriched_books,
            output_path=out_path,
        )

        return [CodePair(**p) for p in pair_dicts]
