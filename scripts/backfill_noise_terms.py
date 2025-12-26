#!/usr/bin/env python3
"""
Backfill script to restore incorrectly filtered technical terms to metadata files.
These terms were removed by the noise filter but should have been kept.
"""

import json
from pathlib import Path
from collections import defaultdict

# Terms that were incorrectly filtered (from user's audited CSV)
KEEP_TERMS = {
    "__asan_memcpy", "__interceptor_malloc", "__malloc_hook", "__malloc_initialize_hook",
    "__free_hook", "__realloc_hook", "__stack_chk_fail", "__wrap_malloc", "__wrap_free",
    "__real_malloc", "__real_free", "__gc", "__cdecl", "__declspec", "__cplusplus",
    "__cxx11", "__gnu", "__gnu_cxx", "__builtin_popcount", "__int128", "__int64",
    "__type_traits", "__make_integer_seq", "__kernel", "__lock_t", "__queue_t",
    "__node_t", "__counter_t", "__hash_t", "__restrict_key_type", "__syncthreads",
    "__sendrecv", "__meta_ec2_instance_id", "__meta_ec2_instance_type", "__meta_ec2_private_ip",
    "__meta_ec2_public_ip", "__meta_ec2_subnet_id", "__meta_ec2_vpc_id",
    "__meta_kubernetes_namespace", "__meta_kubernetes_node_name",
    "__meta_kubernetes_pod_container_name", "__meta_kubernetes_service_name",
    "__meta_kubernetes_service_annotation_prometheus_io_scrape", "__meta_consul_address",
    "__meta_consul_node", "__init__", "__call__", "__getattr__", "__getattribute__",
    "__setattr__", "__getitem__", "__setitem__", "__iter__", "__next__", "__enter__",
    "__exit__", "__await__", "__anext__", "__hash__", "__eq__", "__repr__", "__str__",
    "__slots__", "__mro__", "__class__", "__dict__", "__data", "__file", "__line__var",
    "__list", "__object", "__model", "__property", "__status", "__storage", "__value",
    "__text", "__name"
}

def main():
    log_path = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/logs/pre_filter_extraction.jsonl")
    metadata_dir = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output")

    # Step 1: Build map of book -> chapter -> terms to add
    print("Step 1: Scanning extraction log for terms to restore...")
    books_to_update = defaultdict(lambda: defaultdict(set))

    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                book = entry.get("book", "Unknown")
                chapter = entry.get("chapter", "Unknown")
                
                kw = entry.get("extraction", {}).get("keywords", {})
                raw_terms = set(kw.get("raw_terms", []))
                post_noise_terms = set(kw.get("post_noise_terms", []))
                
                noise_filtered = raw_terms - post_noise_terms
                affected_terms = noise_filtered & KEEP_TERMS
                
                if affected_terms:
                    for term in affected_terms:
                        books_to_update[book][chapter].add(term)
            except:
                continue

    print(f"   Found {len(books_to_update)} books to update")

    # Step 2: Update each metadata file
    print("\nStep 2: Updating metadata files...")
    updated_count = 0
    failed_count = 0
    terms_added_total = 0

    for book_name, chapters_data in sorted(books_to_update.items()):
        # Find the metadata file
        metadata_file = metadata_dir / f"{book_name}_metadata.json"
        
        if not metadata_file.exists():
            print(f"   ❌ Not found: {book_name}_metadata.json")
            failed_count += 1
            continue
        
        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Update chapters - metadata is a list of chapters directly
        terms_added_book = 0
        chapters_updated = 0
        
        # Handle both list format and dict format
        chapters = metadata if isinstance(metadata, list) else metadata.get("chapters", [])
        
        for chapter in chapters:
            # Try different field names for chapter title
            chapter_title = chapter.get("chapter_title") or chapter.get("title", "")
            
            # Find matching chapter in our data
            if chapter_title in chapters_data:
                terms_to_add = chapters_data[chapter_title]
                existing_keywords = set(chapter.get("keywords", []))
                new_keywords = terms_to_add - existing_keywords
                
                if new_keywords:
                    chapter["keywords"] = sorted(existing_keywords | new_keywords)
                    terms_added_book += len(new_keywords)
                    chapters_updated += 1
        
        if terms_added_book > 0:
            # Save updated metadata
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"   ✅ {book_name}: +{terms_added_book} terms in {chapters_updated} chapters")
            updated_count += 1
            terms_added_total += terms_added_book
        else:
            print(f"   ⏭️  {book_name}: no new terms to add (already present)")

    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"Books updated: {updated_count}")
    print(f"Books not found: {failed_count}")
    print(f"Total terms added: {terms_added_total}")

if __name__ == "__main__":
    main()
