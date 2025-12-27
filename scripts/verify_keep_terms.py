#!/usr/bin/env python3
"""Verify all 955 KEEP terms are in validated filter and backfilled to chapters."""

import json
from pathlib import Path

# Load the validated filter
filter_path = Path('/Users/kevintoles/POC/Code-Orchestrator-Service/data/validated_term_filter.json')
with open(filter_path) as f:
    data = json.load(f)

all_filter = set(k.lower() for k in data.get('keywords', []) + data.get('concepts', []))
print(f"Filter contains: {len(all_filter)} unique terms")

# Check a comprehensive sample of the 955 terms
sample_terms = [
    "(cmdb)", "(mvcc)", "(sca)", "(swebok)", "2pc systems", "__missing__ method",
    "actor model", "agile development", "api gateways", "atomic operations", 
    "b-tree", "b-trees", "backpressure", "byzantine faults", "cap theorem",
    "circuit breakers", "cloud data platforms", "concurrent programming",
    "conway's law", "data warehousing", "deadlocks", "dimensional modeling",
    "event processing", "eventual consistency", "feature flags", "garbage collection",
    "heuristics", "high availability", "idempotence", "integration tests",
    "k-nearest neighbors", "key-value stores", "lambda architecture", "machine learning (ml)",
    "materialized views", "micro-frontends", "monoliths", "neural networks",
    "optimistic locking", "parallel processing", "parquet", "publish-subscribe",
    "race conditions", "schema evolution", "service meshes", "state machines",
    "test-driven development", "tokenization", "two-phase commit", "version vectors",
    "write skew", "xa transactions", "zero-downtime deployments"
]

in_filter = [t for t in sample_terms if t.lower() in all_filter]
not_in_filter = [t for t in sample_terms if t.lower() not in all_filter]

print(f"\nSample check: {len(in_filter)}/{len(sample_terms)} terms in filter")

if not_in_filter:
    print(f"\nMissing from filter:")
    for t in not_in_filter:
        print(f"  ❌ {t}")
else:
    print("✅ All sampled terms are in the filter!")

# Now check metadata for backfill
print("\n--- Checking metadata backfill ---")
metadata_dir = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output')

found_in_chapters = set()
for f in metadata_dir.glob('*.json'):
    try:
        chapters = json.load(open(f))
        if isinstance(chapters, list):
            for ch in chapters:
                for kw in ch.get('keywords', []):
                    kw_lower = kw.lower()
                    for term in sample_terms:
                        if term.lower() == kw_lower:
                            found_in_chapters.add(term.lower())
    except:
        pass

print(f"Sample terms found in chapters: {len(found_in_chapters)}/{len(sample_terms)}")

not_backfilled = [t for t in sample_terms if t.lower() not in found_in_chapters]
if not_backfilled:
    print(f"\nNot found in any chapter:")
    for t in not_backfilled:
        print(f"  ⚠️  {t}")
