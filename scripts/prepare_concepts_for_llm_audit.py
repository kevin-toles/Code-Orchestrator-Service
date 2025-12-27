#!/usr/bin/env python3
"""
Prepare extracted concepts for LLM audit.
Creates CSV with unique concepts for DeepSeek + ChatGPT validation.

Input: data/concept_extraction.jsonl
Output: data/concepts_for_llm_audit.csv
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/concept_extraction.jsonl")
OUTPUT_CSV = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/concepts_for_llm_audit.csv")


def main():
    print("=" * 70)
    print("PREPARING CONCEPTS FOR LLM AUDIT")
    print("=" * 70)
    
    # Track concepts with their metadata
    concept_stats = defaultdict(lambda: {
        "domain": "",
        "total_occurrences": 0,
        "books": set(),
        "chapters": 0,
        "sample_keywords": set()
    })
    
    # Read extraction log
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            book = entry["book"]
            keywords = entry.get("keywords", [])
            
            for concept in entry.get("concepts_extracted", []):
                name = concept["name"]
                domain = concept["domain"]
                
                concept_stats[name]["domain"] = domain
                concept_stats[name]["total_occurrences"] += 1
                concept_stats[name]["books"].add(book)
                concept_stats[name]["chapters"] += 1
                
                # Add sample keywords (limit to 10)
                for kw in keywords[:5]:
                    if len(concept_stats[name]["sample_keywords"]) < 10:
                        concept_stats[name]["sample_keywords"].add(kw)
    
    print(f"Unique concepts found: {len(concept_stats)}")
    
    # Write CSV for audit
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Concept", 
            "Domain", 
            "Occurrences", 
            "Books", 
            "Chapters",
            "Sample_Keywords",
            "Keep (Y/N)"
        ])
        
        # Sort by occurrences descending
        for concept, stats in sorted(concept_stats.items(), key=lambda x: -x[1]["total_occurrences"]):
            writer.writerow([
                concept,
                stats["domain"],
                stats["total_occurrences"],
                len(stats["books"]),
                stats["chapters"],
                "; ".join(list(stats["sample_keywords"])[:5]),
                ""  # For manual audit
            ])
    
    print(f"Output saved to: {OUTPUT_CSV}")
    print()
    
    # Summary by domain
    domain_counts = defaultdict(int)
    for stats in concept_stats.values():
        domain_counts[stats["domain"]] += 1
    
    print("Concepts by domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain}: {count}")
    
    print()
    print("Top 20 concepts by occurrence:")
    for concept, stats in sorted(concept_stats.items(), key=lambda x: -x[1]["total_occurrences"])[:20]:
        print(f"  {concept}: {stats['total_occurrences']} ({stats['domain']})")


if __name__ == "__main__":
    main()
