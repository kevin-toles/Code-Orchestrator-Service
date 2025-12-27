#!/usr/bin/env python3
"""
Backfill Concepts Script

This script backfills concepts for chapters that are missing them.
It uses the existing keywords in each chapter to extract concepts
via the /api/v1/concepts endpoint.

IMPORTANT: This script does NOT duplicate:
- Existing concepts are preserved
- New concepts are only added if they don't already exist
- Keywords are not modified

Usage:
    python scripts/backfill_concepts.py [--dry-run] [--limit N]
"""

import argparse
import json
import requests
from pathlib import Path
from typing import Set, List, Dict, Any
import time

# Configuration
ORCHESTRATOR_URL = "http://localhost:8083"
METADATA_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output")
CONCEPTS_ENDPOINT = f"{ORCHESTRATOR_URL}/api/v1/concepts"


def extract_concepts_from_keywords(keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Call the concepts API with keywords to extract concepts.
    
    Args:
        keywords: List of keywords to extract concepts from
        
    Returns:
        List of concept dictionaries from API response
    """
    if not keywords:
        return []
    
    try:
        response = requests.post(
            CONCEPTS_ENDPOINT,
            json={"keywords": keywords},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("concepts", [])
        else:
            print(f"      API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"      Request error: {e}")
        return []


def backfill_metadata_file(metadata_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Backfill concepts for a single metadata file.
    
    Args:
        metadata_path: Path to the metadata JSON file
        dry_run: If True, don't actually modify the file
        
    Returns:
        Dict with stats: chapters_processed, chapters_updated, concepts_added
    """
    stats = {
        "chapters_total": 0,
        "chapters_missing_concepts": 0,
        "chapters_updated": 0,
        "concepts_added": 0,
        "concepts_skipped_duplicate": 0
    }
    
    # Load metadata
    with open(metadata_path) as f:
        chapters = json.load(f)
    
    modified = False
    
    for chapter in chapters:
        stats["chapters_total"] += 1
        
        # Get existing data
        existing_keywords = set(chapter.get("keywords", []))
        existing_concepts = set(chapter.get("concepts", []))
        
        # Skip if chapter already has concepts
        if existing_concepts:
            continue
        
        stats["chapters_missing_concepts"] += 1
        
        # Skip if no keywords to work with
        if not existing_keywords:
            continue
        
        # Extract concepts from keywords
        keywords_list = list(existing_keywords)[:50]  # Limit to 50 keywords per call
        new_concepts_raw = extract_concepts_from_keywords(keywords_list)
        
        # Extract concept names and deduplicate
        new_concept_names = set()
        for c in new_concepts_raw:
            name = c.get("name", "").lower().strip()
            if name:
                new_concept_names.add(name)
        
        # Filter out any that already exist (case-insensitive)
        existing_concepts_lower = {c.lower() for c in existing_concepts}
        concepts_to_add = new_concept_names - existing_concepts_lower
        
        stats["concepts_skipped_duplicate"] += len(new_concept_names - concepts_to_add)
        
        if concepts_to_add:
            stats["chapters_updated"] += 1
            stats["concepts_added"] += len(concepts_to_add)
            
            if not dry_run:
                # Add new concepts (preserve original case from API)
                chapter["concepts"] = sorted(existing_concepts | concepts_to_add)
                modified = True
    
    # Save if modified
    if modified and not dry_run:
        with open(metadata_path, "w") as f:
            json.dump(chapters, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill concepts for chapters missing them")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of books to process")
    parser.add_argument("--book", type=str, default=None, help="Process specific book only")
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONCEPT BACKFILL")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Metadata dir: {METADATA_DIR}")
    print(f"API endpoint: {CONCEPTS_ENDPOINT}")
    print()
    
    # Check API health
    try:
        health = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=5)
        if health.status_code != 200:
            print("❌ Orchestrator service not healthy")
            return
        print("✅ Orchestrator service is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to orchestrator: {e}")
        return
    
    # Get metadata files
    if args.book:
        metadata_files = [METADATA_DIR / f"{args.book}_metadata.json"]
        if not metadata_files[0].exists():
            print(f"❌ Book not found: {args.book}")
            return
    else:
        metadata_files = sorted(METADATA_DIR.glob("*_metadata.json"))
    
    if args.limit:
        metadata_files = metadata_files[:args.limit]
    
    print(f"\nProcessing {len(metadata_files)} metadata files...\n")
    
    # Aggregate stats
    total_stats = {
        "books_processed": 0,
        "books_updated": 0,
        "chapters_total": 0,
        "chapters_missing_concepts": 0,
        "chapters_updated": 0,
        "concepts_added": 0,
        "concepts_skipped_duplicate": 0
    }
    
    for i, meta_file in enumerate(metadata_files, 1):
        book_name = meta_file.stem.replace("_metadata", "")
        print(f"[{i}/{len(metadata_files)}] {book_name[:50]}...")
        
        stats = backfill_metadata_file(meta_file, dry_run=args.dry_run)
        
        total_stats["books_processed"] += 1
        total_stats["chapters_total"] += stats["chapters_total"]
        total_stats["chapters_missing_concepts"] += stats["chapters_missing_concepts"]
        total_stats["chapters_updated"] += stats["chapters_updated"]
        total_stats["concepts_added"] += stats["concepts_added"]
        total_stats["concepts_skipped_duplicate"] += stats["concepts_skipped_duplicate"]
        
        if stats["chapters_updated"] > 0:
            total_stats["books_updated"] += 1
            print(f"   +{stats['concepts_added']} concepts in {stats['chapters_updated']} chapters")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Books processed:           {total_stats['books_processed']}")
    print(f"Books updated:             {total_stats['books_updated']}")
    print(f"Chapters total:            {total_stats['chapters_total']:,}")
    print(f"Chapters missing concepts: {total_stats['chapters_missing_concepts']:,}")
    print(f"Chapters updated:          {total_stats['chapters_updated']:,}")
    print(f"Concepts added:            {total_stats['concepts_added']:,}")
    print(f"Duplicates skipped:        {total_stats['concepts_skipped_duplicate']:,}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No files were modified")
        print("   Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
