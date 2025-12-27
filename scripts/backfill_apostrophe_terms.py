#!/usr/bin/env python3
"""Backfill apostrophe terms found in full text to chapters."""

import json
from pathlib import Path

JSON_TEXTS_DIR = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json')
METADATA_DIR = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output')

# Terms found in full text search
terms_to_backfill = {
    "moore's law": [
        {"book": "Microservices from Day One - Build robust and scalable software from the start", "page": 192}
    ],
    "amdahl's law": [
        {"book": "C++ Concurrency in Action", "page": 11},
        {"book": "C++ Concurrency in Action", "page": 271},
        {"book": "C++ Concurrency in Action", "page": 277},
        {"book": "Computer Systems A Programmer's Perspective", "page": 58},
        {"book": "Database System Concepts 7th Edition", "page": 981},
        {"book": "Effective-Python", "page": 344},
        {"book": "Game Programming Gems 8", "page": 533},
    ],
    "hacker's methodology": [
        {"book": "The Web Application Hackers Handbook 2nd Edition", "page": 606}
    ],
    "conway's law": [
        {"book": "Building Evolutionary Architectures", "page": 239},
        {"book": "Infrastructure as Code 2nd Edition", "page": 265},
        {"book": "Infrastructure as Code 2nd Edition", "page": 303},
        {"book": "Microservices Best Practices for Java", "page": 20},
        {"book": "Microservices for the Enterprise - Designing, Developing, and Deploying", "page": 25},
        {"book": "Microservices in Action", "page": 327},
        {"book": "Python Architecture Patterns", "page": 6},
        {"book": "Spring Microservices - Build scalable microservices with Spring, Docker, and Mesos", "page": 71},
        {"book": "Writing Great Specifications", "page": 163},
    ]
}

print("BACKFILLING TERMS TO CHAPTERS\n")
stats = {'books': set(), 'terms': 0}

for term, locations in terms_to_backfill.items():
    print(f"\n--- {term} ---")
    
    for loc in locations:
        book_name = loc['book']
        page = loc['page']
        
        # Find metadata file
        metadata_path = METADATA_DIR / f'{book_name}_metadata.json'
        if not metadata_path.exists():
            # Try partial match
            candidates = list(METADATA_DIR.glob(f'*{book_name[:25]}*_metadata.json'))
            if candidates:
                metadata_path = candidates[0]
            else:
                print(f"  ❌ No metadata for: {book_name}")
                continue
        
        try:
            with open(metadata_path) as f:
                chapters = json.load(f)
            
            # Find chapter for this page
            found_chapter = False
            for i, ch in enumerate(chapters):
                start = ch.get('start_page', 0)
                end = ch.get('end_page', 0)
                if start and end and start <= page <= end:
                    existing = set(k.lower() for k in ch.get('keywords', []))
                    if term.lower() not in existing:
                        if 'keywords' not in ch:
                            ch['keywords'] = []
                        ch['keywords'].append(term)
                        stats['terms'] += 1
                        stats['books'].add(metadata_path.stem)
                        print(f"  + Added to {book_name[:40]}... ch{i+1} (page {page})")
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(chapters, f, indent=2)
                    else:
                        print(f"  ✓ Already in {book_name[:40]}... ch{i+1}")
                    found_chapter = True
                    break
            
            if not found_chapter:
                print(f"  ⚠️  Page {page} not in any chapter range for {book_name[:40]}...")
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")

print(f"\n{'='*60}")
print(f"SUMMARY: {stats['terms']} terms added across {len(stats['books'])} books")
