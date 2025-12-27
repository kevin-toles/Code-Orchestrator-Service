#!/usr/bin/env python3
"""
Backfill validated concepts to chapters with low/no concept coverage.
Uses the extraction log and only adds concepts that passed LLM validation.
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Paths
EXTRACTION_LOG = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/concept_extraction.jsonl")
LLM_VALIDATION = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/taxonomy_llm_validation.json")
METADATA_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output")


def main():
    print("=" * 70)
    print("CONCEPT BACKFILL FROM LLM-VALIDATED TERMS")
    print("=" * 70)
    print()
    
    # 1. Load LLM-validated concepts (consensus KEEP)
    with open(LLM_VALIDATION) as f:
        llm_data = json.load(f)
    
    validated_concepts = set(c.lower() for c in llm_data.get("consensus_keep", []))
    print(f"LLM-validated concepts: {len(validated_concepts)}")
    
    # 2. Build map of book -> chapter_index -> concepts to add
    print(f"Reading extraction log: {EXTRACTION_LOG}")
    
    books_to_update = defaultdict(lambda: defaultdict(set))
    
    with open(EXTRACTION_LOG) as f:
        for line in f:
            entry = json.loads(line)
            book = entry["book"]
            chapter_idx = entry["chapter_index"]
            existing = set(c.lower() for c in entry.get("existing_concepts", []))
            
            # Only add validated concepts that aren't already present
            for concept in entry.get("concepts_extracted", []):
                concept_name = concept["name"]
                if concept_name.lower() in validated_concepts and concept_name.lower() not in existing:
                    books_to_update[book][chapter_idx].add(concept_name)
    
    print(f"Books to update: {len(books_to_update)}")
    
    # 3. Update metadata files
    print()
    print("Updating metadata files...")
    
    stats = {
        "books_updated": 0,
        "chapters_updated": 0,
        "concepts_added": 0,
        "duplicates_skipped": 0
    }
    
    for book_name, chapters_data in sorted(books_to_update.items()):
        metadata_file = METADATA_DIR / f"{book_name}_metadata.json"
        
        if not metadata_file.exists():
            print(f"  ❌ Not found: {book_name}_metadata.json")
            continue
        
        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        book_modified = False
        book_concepts_added = 0
        
        for chapter_idx, concepts_to_add in chapters_data.items():
            if chapter_idx >= len(metadata):
                continue
            
            chapter = metadata[chapter_idx]
            existing_concepts = set(c.lower() for c in chapter.get("concepts", []))
            
            # Add only non-duplicate concepts
            new_concepts = []
            for concept in concepts_to_add:
                if concept.lower() not in existing_concepts:
                    new_concepts.append(concept)
                    existing_concepts.add(concept.lower())
                else:
                    stats["duplicates_skipped"] += 1
            
            if new_concepts:
                if "concepts" not in chapter:
                    chapter["concepts"] = []
                chapter["concepts"].extend(new_concepts)
                book_modified = True
                book_concepts_added += len(new_concepts)
                stats["chapters_updated"] += 1
        
        if book_modified:
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            stats["books_updated"] += 1
            stats["concepts_added"] += book_concepts_added
            print(f"  ✅ {book_name[:50]}: +{book_concepts_added} concepts")
    
    # 4. Summary
    print()
    print("=" * 70)
    print("BACKFILL COMPLETE")
    print("=" * 70)
    print(f"Books updated: {stats['books_updated']}")
    print(f"Chapters updated: {stats['chapters_updated']}")
    print(f"Concepts added: {stats['concepts_added']}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']}")


if __name__ == "__main__":
    main()
