#!/usr/bin/env python3
"""
Extract index/glossary terms from book appendices and insert into appropriate chapters.
Maps terms to chapters using page numbers.

Process:
1. Parse index pages from raw JSON books
2. Extract term -> page number mappings
3. Map pages to chapters using chapter start_page/end_page
4. Insert terms as keywords/concepts into chapters
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Paths
JSON_TEXTS_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json")
METADATA_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output")
OUTPUT_LOG = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/index_extraction.jsonl")

# Validated terms for filtering
VALIDATED_TERMS_FILE = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/validated_term_filter.json")


def find_index_pages(pages: list) -> str:
    """Find and concatenate index/glossary pages."""
    index_content = []
    in_index = False
    
    for page in pages:
        content = page.get("content", "") if isinstance(page, dict) else str(page)
        
        # Start capturing at Index, Glossary, or Appendix (if it's an index appendix)
        if not in_index:
            # Check if this page starts an index section
            first_100 = content[:100].lower()
            if any(marker in first_100 for marker in ['index\n', 'index\r', 'glossary\n', 'glossary\r']):
                in_index = True
        
        if in_index:
            index_content.append(content)
            
            # Stop if we hit a different section (but not immediately)
            if len(index_content) > 2:
                first_50 = content[:50].lower()
                if any(end in first_50 for end in ['other books', 'about the author', 'colophon']):
                    break
    
    return "\n".join(index_content)


def parse_index_entries(index_content: str) -> list[dict]:
    """Parse index entries from content. Returns list of {term, pages}."""
    entries = []
    
    for line in index_content.split('\n'):
        line = line.strip()
        
        # Skip empty, single letters (section headers), page numbers alone
        if not line or len(line) <= 2:
            continue
        if line.isdigit():
            continue
        if line in ['Index', 'Symbols', 'Glossary'] or (len(line) == 1 and line.isalpha()):
            continue
        
        # Multiple patterns for different index formats
        patterns = [
            # Pattern 1: "term  page_numbers" (2+ spaces)
            r'^(.+?)\s{2,}(\d[\d,\s\-–n]+)$',
            # Pattern 2: "term, page_numbers" (comma before pages)
            r'^([^,]+),\s*(\d[\d,\s\-–n]+)$',
            # Pattern 3: "term page_numbers" (single space, term ends with letter)
            r'^([a-zA-Z][\w\s\(\)\-]+?)\s+(\d{1,4}(?:[\s,\-–n]+\d{1,4})*)$',
            # Pattern 4: lines ending in page numbers (common in PDFs)
            r'^(.+?)\s+(\d{1,4}(?:[\s,\-–n]+\d{1,4})*)$',
        ]
        
        term = None
        pages_str = None
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                candidate_term = match.group(1).strip()
                candidate_pages = match.group(2).strip()
                
                # Validate: term should have letters, pages should have numbers
                if re.search(r'[a-zA-Z]{2,}', candidate_term) and re.search(r'\d', candidate_pages):
                    term = candidate_term
                    pages_str = candidate_pages
                    break
        
        if not term or not pages_str:
            continue
        
        # Clean up term
        term = re.sub(r'\s+', ' ', term)  # Normalize whitespace
        term = re.sub(r'^[•\-\*]\s*', '', term)  # Remove bullet points
        
        # Skip if term is too short/long or looks like noise
        if len(term) < 3 or len(term) > 80:
            continue
        if term.lower() in ['see', 'see also', 'also', 'versus']:
            continue
        
        # Parse page numbers (handle ranges like 123-456, 123n, lists like 123, 456)
        pages = []
        # Remove 'n' suffix (footnote indicator)
        pages_str = re.sub(r'n(?=\s|,|$)', '', pages_str)
        
        for part in re.split(r'[,\s]+', pages_str):
            part = part.strip()
            if not part:
                continue
            if '-' in part or '–' in part:
                # Range
                range_parts = re.split(r'[-–]', part)
                if len(range_parts) == 2:
                    try:
                        start = int(range_parts[0])
                        end = int(range_parts[1])
                        if end > start and (end - start) < 100:  # Reasonable range
                            pages.extend(range(start, end + 1))
                    except ValueError:
                        pass
            else:
                try:
                    pages.append(int(part))
                except ValueError:
                    pass
        
        if pages and term:
            entries.append({"term": term, "pages": pages})
    
    return entries


def map_pages_to_chapters(entries: list[dict], chapters: list[dict]) -> dict:
    """Map index entries to chapters based on page numbers.
    Returns: {chapter_index: [terms]}
    """
    chapter_terms = defaultdict(set)
    
    # Build page -> chapter mapping
    page_to_chapter = {}
    for i, ch in enumerate(chapters):
        start = ch.get("start_page", 0)
        end = ch.get("end_page", 0)
        if start and end:
            for p in range(start, end + 1):
                page_to_chapter[p] = i
    
    # Map entries to chapters
    for entry in entries:
        term = entry["term"]
        for page in entry["pages"]:
            if page in page_to_chapter:
                chapter_idx = page_to_chapter[page]
                chapter_terms[chapter_idx].add(term)
    
    return chapter_terms


def process_book(json_path: Path, metadata_path: Path, validated_terms: set, dry_run: bool = True) -> dict:
    """Process a single book: extract index, map to chapters, update metadata."""
    results = {
        "book": json_path.stem,
        "index_entries": 0,
        "chapters_updated": 0,
        "terms_added": 0,
        "terms_by_chapter": {}
    }
    
    # Load raw book JSON
    try:
        with open(json_path) as f:
            book_data = json.load(f)
    except Exception as e:
        results["error"] = f"Failed to load JSON: {e}"
        return results
    
    pages = book_data.get("pages", [])
    if not pages:
        results["error"] = "No pages found"
        return results
    
    # Find and parse index
    index_content = find_index_pages(pages)
    if not index_content:
        results["error"] = "No index found"
        return results
    
    entries = parse_index_entries(index_content)
    results["index_entries"] = len(entries)
    
    if not entries:
        results["error"] = "No entries parsed from index"
        return results
    
    # Load metadata
    if not metadata_path.exists():
        results["error"] = f"Metadata file not found: {metadata_path.name}"
        return results
    
    with open(metadata_path) as f:
        chapters = json.load(f)
    
    # Map entries to chapters
    chapter_terms = map_pages_to_chapters(entries, chapters)
    
    # Update chapters
    modified = False
    for ch_idx, terms in chapter_terms.items():
        if ch_idx >= len(chapters):
            continue
        
        chapter = chapters[ch_idx]
        existing_keywords = set(k.lower() for k in chapter.get("keywords", []))
        
        # Filter to validated terms and non-duplicates
        new_terms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower in validated_terms and term_lower not in existing_keywords:
                new_terms.append(term)
                existing_keywords.add(term_lower)
        
        if new_terms:
            if "keywords" not in chapter:
                chapter["keywords"] = []
            chapter["keywords"].extend(new_terms)
            results["terms_added"] += len(new_terms)
            results["chapters_updated"] += 1
            results["terms_by_chapter"][ch_idx] = new_terms
            modified = True
    
    # Save if not dry run
    if modified and not dry_run:
        with open(metadata_path, 'w') as f:
            json.dump(chapters, f, indent=2)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract index terms and insert into chapters")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report")
    parser.add_argument("--limit", type=int, help="Limit number of books to process")
    parser.add_argument("--book", type=str, help="Process specific book by name")
    parser.add_argument("--exclude-file", type=str, help="File with book names to exclude (one per line)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("INDEX TERM EXTRACTION")
    print("=" * 70)
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Load validated terms
    with open(VALIDATED_TERMS_FILE) as f:
        validated_data = json.load(f)
    validated_terms = set(k.lower() for k in validated_data.get("keywords", []))
    validated_terms |= set(c.lower() for c in validated_data.get("concepts", []))
    print(f"Validated terms loaded: {len(validated_terms)}")
    print()
    
    # Load exclusion list if provided
    exclude_set = set()
    if args.exclude_file:
        with open(args.exclude_file) as f:
            exclude_set = set(line.strip() for line in f if line.strip())
        print(f"Excluding {len(exclude_set)} previously processed books")
        print()
    
    # Find books
    json_files = sorted(JSON_TEXTS_DIR.glob("*.json"))
    if args.book:
        json_files = [f for f in json_files if args.book.lower() in f.stem.lower()]
    if exclude_set:
        json_files = [f for f in json_files if f.stem not in exclude_set]
    if args.limit:
        json_files = json_files[:args.limit]
    
    print(f"Processing {len(json_files)} books...")
    print()
    
    # Process books
    stats = {
        "books_processed": 0,
        "books_with_index": 0,
        "total_entries": 0,
        "total_terms_added": 0,
        "total_chapters_updated": 0
    }
    
    with open(OUTPUT_LOG, 'w') as log_file:
        for json_path in json_files:
            book_name = json_path.stem
            
            # Find corresponding metadata file
            # Try exact match first, then fuzzy
            metadata_path = METADATA_DIR / f"{book_name}_metadata.json"
            if not metadata_path.exists():
                # Try to find a close match
                candidates = list(METADATA_DIR.glob(f"*{book_name[:20]}*_metadata.json"))
                if candidates:
                    metadata_path = candidates[0]
            
            print(f"[{stats['books_processed']+1}/{len(json_files)}] {book_name[:50]}...", end=" ")
            
            result = process_book(json_path, metadata_path, validated_terms, dry_run=args.dry_run)
            
            # Log result
            log_file.write(json.dumps(result) + "\n")
            
            if "error" in result:
                print(f"⚠️ {result['error']}")
            elif result["terms_added"] > 0:
                print(f"✅ +{result['terms_added']} terms in {result['chapters_updated']} chapters (from {result['index_entries']} index entries)")
                stats["books_with_index"] += 1
                stats["total_entries"] += result["index_entries"]
                stats["total_terms_added"] += result["terms_added"]
                stats["total_chapters_updated"] += result["chapters_updated"]
            else:
                print(f"📖 {result['index_entries']} entries, 0 new terms")
                if result["index_entries"] > 0:
                    stats["books_with_index"] += 1
                    stats["total_entries"] += result["index_entries"]
            
            stats["books_processed"] += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Books processed: {stats['books_processed']}")
    print(f"Books with index: {stats['books_with_index']}")
    print(f"Total index entries found: {stats['total_entries']}")
    print(f"Total terms added: {stats['total_terms_added']}")
    print(f"Total chapters updated: {stats['total_chapters_updated']}")
    print()
    
    if args.dry_run:
        print("⚠️ DRY RUN - No files were modified")
        print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
