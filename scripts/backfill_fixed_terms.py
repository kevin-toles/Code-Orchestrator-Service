#!/usr/bin/env python3
"""Backfill fixed uncertain terms to chapters based on index page numbers."""

import json
import re
from pathlib import Path

# Paths
JSON_TEXTS_DIR = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json')
METADATA_DIR = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output')

# Terms to backfill: fixed term -> variants to search for in index
TERMS_TO_FIND = {
    'quality design decisions': ['quality design decisions'],
    'general scenario': ['general scenario'],
    'surrogate': ['surrogate'],
    'pipe processor': ['pipe processor'],
    'pipes': ['pipes'],
    'function overloading': ['function overloads', 'function overloading'],
    'automatic storage duration': ['automatic storage', 'automatic'],
    'heuristics': ['heuristics'],
    'preprocessor directives': ['#define', 'preprocessor']
}


def find_index_pages(pages):
    """Extract index/glossary content from book pages."""
    index_content = []
    in_index = False
    for page in pages:
        content = page.get('content', '') if isinstance(page, dict) else str(page)
        if not in_index:
            first_100 = content[:100].lower()
            if any(marker in first_100 for marker in ['index\n', 'index\r', 'glossary\n']):
                in_index = True
        if in_index:
            index_content.append(content)
    return '\n'.join(index_content)


def parse_index_for_terms(index_content, search_terms):
    """Find page numbers for specific terms in index."""
    results = {}
    for line in index_content.split('\n'):
        line_lower = line.lower().strip()
        for term, variants in search_terms.items():
            for variant in variants:
                if variant.lower() in line_lower:
                    # Extract page numbers from end of line
                    match = re.search(r'(\d[\d,\s\-–]+)\s*$', line)
                    if match:
                        pages_str = match.group(1)
                        pages = []
                        for part in re.split(r'[,\s]+', pages_str):
                            part = part.strip()
                            if '-' in part or '–' in part:
                                range_parts = re.split(r'[-–]', part)
                                if len(range_parts) == 2:
                                    try:
                                        start, end = int(range_parts[0]), int(range_parts[1])
                                        if end > start and (end - start) < 100:
                                            pages.extend(range(start, end + 1))
                                    except ValueError:
                                        pass
                            else:
                                try:
                                    pages.append(int(part))
                                except ValueError:
                                    pass
                        if pages:
                            if term not in results:
                                results[term] = set()
                            results[term].update(pages)
    return results


def main():
    stats = {'books': 0, 'chapters': 0, 'terms': 0}
    term_counts = {term: 0 for term in TERMS_TO_FIND}

    for json_path in sorted(JSON_TEXTS_DIR.glob('*.json')):
        book_name = json_path.stem
        
        # Find metadata
        metadata_path = METADATA_DIR / f'{book_name}_metadata.json'
        if not metadata_path.exists():
            candidates = list(METADATA_DIR.glob(f'*{book_name[:20]}*_metadata.json'))
            if candidates:
                metadata_path = candidates[0]
            else:
                continue
        
        try:
            with open(json_path) as f:
                book_data = json.load(f)
            pages = book_data.get('pages', [])
            if not pages:
                continue
            
            index_content = find_index_pages(pages)
            if not index_content:
                continue
            
            # Find terms and their pages
            term_pages = parse_index_for_terms(index_content, TERMS_TO_FIND)
            if not term_pages:
                continue
            
            # Load chapters
            with open(metadata_path) as f:
                chapters = json.load(f)
            
            # Build page->chapter map
            page_to_ch = {}
            for i, ch in enumerate(chapters):
                start = ch.get('start_page', 0)
                end = ch.get('end_page', 0)
                if start and end:
                    for p in range(start, end + 1):
                        page_to_ch[p] = i
            
            # Add terms to chapters
            book_modified = False
            for term, pages in term_pages.items():
                for page in pages:
                    if page in page_to_ch:
                        ch_idx = page_to_ch[page]
                        ch = chapters[ch_idx]
                        existing = set(k.lower() for k in ch.get('keywords', []))
                        if term.lower() not in existing:
                            if 'keywords' not in ch:
                                ch['keywords'] = []
                            ch['keywords'].append(term)
                            stats['terms'] += 1
                            term_counts[term] += 1
                            book_modified = True
            
            if book_modified:
                with open(metadata_path, 'w') as f:
                    json.dump(chapters, f, indent=2)
                stats['books'] += 1
                
        except Exception as e:
            print(f"Error processing {book_name}: {e}")
            continue

    print(f"\nResults:")
    print(f"  Books updated: {stats['books']}")
    print(f"  Terms added: {stats['terms']}")
    print(f"\nPer-term counts:")
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {term}: {count}")


if __name__ == '__main__':
    main()
