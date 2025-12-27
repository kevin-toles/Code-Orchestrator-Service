#!/usr/bin/env python3
"""Find page numbers for missing terms from book indexes and backfill to chapters."""

import json
import re
from pathlib import Path

JSON_TEXTS_DIR = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json')
METADATA_DIR = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output')

# Missing terms to find - include variants with ligatures
MISSING_TERMS = {
    "amdahl's law": ["amdahl's law", "amdahl's law"],
    "buffer overflow": ["buffer overflow", "buffer overﬂow"],
    "bypassing filters": ["bypassing filters", "bypassing ﬁ lters", "bypassing ﬁlters"],
    "change, implementing": ["change, implementing"],
    "cloud data platforms": ["cloud data platforms"],
    "compared with inheritance": ["compared with inheritance"],
    "content-length:": ["content-length:"],
    "data, inspecting": ["data, inspecting"],
    "description and interests": ["description and interests"],
    "design of restaurant service": ["design of restaurant service"],
    "devops cafe podcast": ["devops cafe podcast"],
    "domain-specific examples": ["domain-specific examples"],
    "employee profiles": ["employee profiles", "employee proﬁ les", "employee proﬁles"],
    "establishing": ["establishing"],
    "executable object files": ["executable object files", "executable object ﬁles"],
    "hacker's methodology": ["hacker's methodology", "hacker's methodology"],
    "less or equal": ["less or equal"],
    "maintenance notes": ["maintenance notes"],
    "moore's law": ["moore's law", "moore's law"],
    "operand specifiers": ["operand specifiers", "operand speciﬁers"],
    "port:": ["port:"],
    "register files": ["register files", "register ﬁles"],
    "relfrozenxid of table": ["relfrozenxid of table"],
    "running parametrized": ["running parametrized"],
    "snowflaking": ["snowflaking", "snowﬂ aking", "snowﬂaking"],
    "starting with thens": ["starting with thens"],
    "testing in": ["testing in"],
    "tests for order service": ["tests for order service"],
    "thomas' write rule": ["thomas' write rule", "thomas ' write rule"],
    "using - -stateful=links": ["using - -stateful=links", "using --stateful=links"],
    "using one when per scenario": ["using one when per scenario"],
    "virtual offices": ["virtual offices", "virtual ofﬁces"],
    "warning notes": ["warning notes"],
    "first web era": ["first web era", "ﬁrst web era"],
    "five-stage pipelines": ["five-stage pipelines", "ﬁve-stage pipelines"],
}


def find_index_section(pages):
    """Find and return index content from book pages."""
    index_content = []
    in_index = False
    index_start_page = None
    
    for i, page in enumerate(pages):
        content = page.get('content', '') if isinstance(page, dict) else str(page)
        if not in_index:
            first_150 = content[:150].lower()
            if any(marker in first_150 for marker in ['index\n', 'index\r', '\nindex\n', 'glossary\n']):
                in_index = True
                index_start_page = i + 1
        if in_index:
            index_content.append((i + 1, content))  # (page_num, content)
    
    return index_content, index_start_page


def search_index_for_term(index_content, variants):
    """Search index content for term variants and extract page numbers."""
    results = []
    
    for idx_page, content in index_content:
        for line in content.split('\n'):
            line_lower = line.lower().strip()
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
                            results.append({
                                'variant_found': variant,
                                'line': line.strip()[:100],
                                'pages': pages
                            })
    return results


def main():
    term_findings = {}
    
    print("Scanning all JSON books for missing terms in indexes...\n")
    
    for json_path in sorted(JSON_TEXTS_DIR.glob('*.json')):
        book_name = json_path.stem
        
        try:
            with open(json_path) as f:
                book_data = json.load(f)
            pages = book_data.get('pages', [])
            if not pages:
                continue
            
            index_content, index_start = find_index_section(pages)
            if not index_content:
                continue
            
            # Search for each missing term
            for term, variants in MISSING_TERMS.items():
                results = search_index_for_term(index_content, variants)
                if results:
                    if term not in term_findings:
                        term_findings[term] = []
                    for r in results:
                        term_findings[term].append({
                            'book': book_name,
                            'variant': r['variant_found'],
                            'line': r['line'],
                            'pages': r['pages']
                        })
        except Exception as e:
            continue
    
    # Print findings
    print("=" * 80)
    print("MISSING TERMS FOUND IN INDEXES")
    print("=" * 80)
    
    for term in MISSING_TERMS.keys():
        findings = term_findings.get(term, [])
        if findings:
            print(f"\n📖 {term.upper()}")
            print("-" * 60)
            for f in findings:
                print(f"  Book: {f['book']}")
                print(f"  Line: {f['line']}")
                print(f"  Pages: {f['pages']}")
                print()
        else:
            print(f"\n❌ {term.upper()} - NOT FOUND IN ANY INDEX")
    
    # Now do the backfill
    print("\n" + "=" * 80)
    print("BACKFILLING TO CHAPTERS")
    print("=" * 80)
    
    stats = {'books': 0, 'chapters': 0, 'terms': 0}
    
    for term, findings in term_findings.items():
        for finding in findings:
            book_name = finding['book']
            pages = finding['pages']
            
            # Find metadata file
            metadata_path = METADATA_DIR / f'{book_name}_metadata.json'
            if not metadata_path.exists():
                candidates = list(METADATA_DIR.glob(f'*{book_name[:30]}*_metadata.json'))
                if candidates:
                    metadata_path = candidates[0]
                else:
                    print(f"  No metadata for {book_name}")
                    continue
            
            try:
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
                
                # Add term to appropriate chapters
                modified = False
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
                            stats['chapters'] += 1
                            modified = True
                            print(f"  + Added '{term}' to {book_name} chapter {ch_idx + 1} (page {page})")
                
                if modified:
                    with open(metadata_path, 'w') as f:
                        json.dump(chapters, f, indent=2)
                    stats['books'] += 1
                    
            except Exception as e:
                print(f"  Error processing {book_name}: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {stats['terms']} terms added to {stats['chapters']} chapters in {stats['books']} books")


if __name__ == '__main__':
    main()
