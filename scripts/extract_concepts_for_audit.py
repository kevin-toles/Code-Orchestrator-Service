#!/usr/bin/env python3
"""
Extract concepts from chapters with LOW or ZERO concept coverage.
Logs to JSONL for manual audit - does NOT modify metadata files.

Finds chapters where:
  - concepts is empty OR
  - concepts count < threshold (default: 2)

Output: data/concept_extraction.jsonl
Format per line:
{
    "book": "Book Title",
    "chapter": "Chapter Title", 
    "chapter_index": 0,
    "keywords": ["kw1", "kw2", ...],
    "existing_concepts": ["existing1", ...],
    "concepts_extracted": [
        {"name": "concept1", "domain": "domain1", "confidence": 0.7},
        ...
    ]
}
"""

import json
import argparse
import requests
from pathlib import Path
from datetime import datetime

# Paths
METADATA_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output")
OUTPUT_FILE = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/concept_extraction.jsonl")
CONCEPTS_ENDPOINT = "http://localhost:8083/api/v1/concepts"


def check_service_health():
    """Check if orchestrator service is running."""
    try:
        response = requests.get("http://localhost:8083/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def extract_concepts(keywords: list[str]) -> list[dict]:
    """Call concepts API to extract concepts from keywords."""
    if not keywords:
        return []
    
    try:
        response = requests.post(
            CONCEPTS_ENDPOINT,
            json={"keywords": keywords},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("concepts", [])
    except Exception as e:
        print(f"      API error: {e}")
    return []


def process_metadata_file(metadata_path: Path, output_file, stats: dict, min_concepts: int):
    """Process chapters with low/no concepts and log extractions."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            chapters = json.load(f)
    except Exception as e:
        print(f"  ❌ Error reading {metadata_path.name}: {e}")
        return
    
    if not isinstance(chapters, list):
        return
    
    # Extract book title from filename (remove _metadata.json suffix)
    book_title = metadata_path.stem.replace("_metadata", "")
    
    chapters_needing_concepts = 0
    concepts_added = 0
    
    for idx, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            continue
        
        chapter_title = chapter.get("title", f"Chapter {idx + 1}")
        keywords = chapter.get("keywords", [])
        existing_concepts = chapter.get("concepts", [])
        
        stats["chapters_total"] += 1
        
        # Only process chapters with LOW or ZERO concepts
        if len(existing_concepts) >= min_concepts:
            stats["chapters_sufficient"] += 1
            continue
        
        stats["chapters_low_concepts"] += 1
        
        # Skip if no keywords to extract from
        if not keywords:
            stats["chapters_no_keywords"] += 1
            continue
        
        # Extract concepts from keywords using expanded taxonomy
        concepts_extracted = extract_concepts(keywords)
        
        # Log the extraction (even if no concepts found)
        log_entry = {
            "book": book_title,
            "chapter": chapter_title,
            "chapter_index": idx,
            "keywords": keywords,
            "existing_concepts": existing_concepts,
            "concepts_extracted": concepts_extracted
        }
        
        output_file.write(json.dumps(log_entry) + "\n")
        
        if concepts_extracted:
            chapters_needing_concepts += 1
            concepts_added += len(concepts_extracted)
            stats["concepts_extracted"] += len(concepts_extracted)
            
            # Track unique concepts
            for c in concepts_extracted:
                stats["unique_concepts"].add(c["name"].lower())
                stats["domains"][c["domain"]] = stats["domains"].get(c["domain"], 0) + 1
    
    return chapters_needing_concepts, concepts_added


def main():
    parser = argparse.ArgumentParser(description="Extract concepts for chapters with low coverage (audit only)")
    parser.add_argument("--limit", type=int, help="Limit number of books to process")
    parser.add_argument("--min-concepts", type=int, default=1, 
                        help="Threshold: chapters with fewer concepts than this will be processed (default: 1 = only empty)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONCEPT EXTRACTION FOR AUDIT (LOW/ZERO COVERAGE CHAPTERS)")
    print("=" * 70)
    print(f"Metadata dir: {METADATA_DIR}")
    print(f"Output file:  {OUTPUT_FILE}")
    print(f"Min concepts threshold: {args.min_concepts} (chapters with < {args.min_concepts} concepts)")
    print()
    
    # Check service
    if not check_service_health():
        print("❌ Orchestrator service not running. Start with: docker-compose up -d")
        return
    print("✅ Orchestrator service is healthy")
    print()
    
    # Find all metadata files
    metadata_files = sorted(METADATA_DIR.glob("*_metadata.json"))
    if args.limit:
        metadata_files = metadata_files[:args.limit]
    
    print(f"Processing {len(metadata_files)} books...")
    print()
    
    stats = {
        "chapters_total": 0,
        "chapters_sufficient": 0,
        "chapters_low_concepts": 0,
        "chapters_no_keywords": 0,
        "concepts_extracted": 0,
        "unique_concepts": set(),
        "domains": {}
    }
    
    start_time = datetime.now()
    books_with_extractions = 0
    
    # Open output file and process all books
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        for i, metadata_path in enumerate(metadata_files, 1):
            book_name = metadata_path.stem.replace("_metadata", "")[:50]
            print(f"[{i}/{len(metadata_files)}] {book_name}...", end=" ", flush=True)
            
            result = process_metadata_file(metadata_path, output_file, stats, args.min_concepts)
            
            if result:
                chapters_updated, concepts_added = result
                if concepts_added > 0:
                    books_with_extractions += 1
                    print(f"+{concepts_added} concepts in {chapters_updated} chapters")
                else:
                    print("(no new concepts)")
            else:
                print("(skipped)")
    
    elapsed = datetime.now() - start_time
    
    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE - READY FOR AUDIT")
    print("=" * 70)
    print(f"Time elapsed: {elapsed}")
    print(f"Books processed: {len(metadata_files)}")
    print(f"Books with extractions: {books_with_extractions}")
    print()
    print(f"Chapters total: {stats['chapters_total']}")
    print(f"Chapters with sufficient concepts (>= {args.min_concepts}): {stats['chapters_sufficient']}")
    print(f"Chapters with low/no concepts: {stats['chapters_low_concepts']}")
    print(f"  - With keywords (processed): {stats['chapters_low_concepts'] - stats['chapters_no_keywords']}")
    print(f"  - No keywords (skipped): {stats['chapters_no_keywords']}")
    print()
    print(f"Total concepts extracted: {stats['concepts_extracted']}")
    print(f"Unique concepts: {len(stats['unique_concepts'])}")
    print()
    print("Domain breakdown:")
    for domain, count in sorted(stats["domains"].items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")
    print()
    print(f"Output saved to: {OUTPUT_FILE}")
    print()
    print("Next steps:")
    print("  1. Run: python3 scripts/prepare_concepts_for_llm_audit.py")
    print("  2. Send to DeepSeek + ChatGPT for consensus filtering")
    print("  3. Run backfill with validated concepts")


if __name__ == "__main__":
    main()
