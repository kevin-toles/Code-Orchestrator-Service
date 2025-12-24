#!/usr/bin/env python3
"""Generate clean concept labels from BERTopic clusters using LLM.

This is a ONE-TIME batch operation to:
1. Load BERTopic-discovered term clusters
2. Send clusters to LLM to generate proper concept names
3. Save cleaned concepts to config file for runtime use

Usage:
    python scripts/generate_concept_labels.py

Output:
    config/concept_vocabulary.json - Canonical concept names with term mappings
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI
from src.nlp.concept_discovery import (
    ConceptDiscoveryConfig,
    discover_concepts_from_metadata,
)

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Configuration - loaded from environment
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not set in environment. "
        "Please add it to .env file or set as environment variable."
    )

METADATA_DIR = "/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output"
OUTPUT_PATH = "config/concept_vocabulary.json"

# How many clusters to process per LLM call (to stay within context limits)
BATCH_SIZE = 50


def create_client() -> OpenAI:
    """Create OpenRouter client."""
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def build_prompt(clusters: list[dict]) -> str:
    """Build prompt for LLM to generate concept labels.
    
    Args:
        clusters: List of {"id": int, "terms": list[str]}
    
    Returns:
        Prompt string.
    """
    cluster_text = "\n".join(
        f"Cluster {c['id']}: {', '.join(c['terms'][:10])}"
        for c in clusters
    )
    
    return f"""You are a software engineering domain expert. I have clusters of related terms extracted from technical books about software architecture, DevOps, and programming.

For each cluster, provide a SINGLE canonical concept name that represents the abstract idea these terms relate to.

Rules:
1. Concept names should be 2-4 words (e.g., "Container Orchestration", "Test-Driven Development", "Domain-Driven Design")
2. Use proper technical terminology
3. If a cluster contains noise/unrelated terms, respond with "SKIP" for that cluster
4. Concepts should be abstract ideas, not specific tools (e.g., "Container Orchestration" not "Kubernetes")

Clusters:
{cluster_text}

Respond in JSON format:
{{
  "concepts": [
    {{"cluster_id": 1, "concept_name": "Software Testing", "confidence": "high"}},
    {{"cluster_id": 2, "concept_name": "SKIP", "confidence": "low"}},
    ...
  ]
}}

Only output valid JSON, no other text."""


def call_llm(client: OpenAI, prompt: str) -> dict:
    """Call LLM to generate concept labels.
    
    Args:
        client: OpenAI client.
        prompt: Prompt string.
    
    Returns:
        Parsed JSON response.
    """
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a software engineering expert. Output only valid JSON."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4000,
    )
    
    content = response.choices[0].message.content
    
    # Handle potential markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    return json.loads(content.strip())


def main():
    print("=" * 60)
    print("CONCEPT LABEL GENERATION")
    print("=" * 60)
    
    # Step 1: Discover clusters with BERTopic
    print("\n[1/3] Discovering term clusters with BERTopic...")
    
    config = ConceptDiscoveryConfig(
        min_topic_size=5,
        min_cluster_size=5,
        top_n_words=10,
        filter_noise_patterns=True,
    )
    
    result = discover_concepts_from_metadata(METADATA_DIR, config=config)
    
    print(f"      Found {result.topic_count} clusters from {result.total_terms} terms")
    
    # Step 2: Prepare clusters for LLM
    print("\n[2/3] Sending clusters to LLM for labeling...")
    
    clusters = [
        {"id": c.topic_id, "terms": c.representative_terms}
        for c in result.concepts
        if c.quality_score >= 0.5  # Only process higher quality clusters
    ]
    
    print(f"      Processing {len(clusters)} high-quality clusters")
    
    # Step 3: Call LLM in batches
    client = create_client()
    all_concepts = []
    
    for i in range(0, len(clusters), BATCH_SIZE):
        batch = clusters[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(clusters) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"      Batch {batch_num}/{total_batches} ({len(batch)} clusters)...")
        
        prompt = build_prompt(batch)
        
        try:
            response = call_llm(client, prompt)
            concepts = response.get("concepts", [])
            
            # Map back to original cluster data
            cluster_map = {c["id"]: c for c in batch}
            
            for concept in concepts:
                cluster_id = concept.get("cluster_id")
                name = concept.get("concept_name", "").strip()
                confidence = concept.get("confidence", "medium")
                
                if name and name != "SKIP" and cluster_id in cluster_map:
                    all_concepts.append({
                        "name": name,
                        "confidence": confidence,
                        "representative_terms": cluster_map[cluster_id]["terms"][:10],
                        "cluster_id": cluster_id,
                    })
            
            print(f"        → {len(concepts)} concepts labeled")
            
        except Exception as e:
            print(f"        ✗ Error: {e}")
            continue
    
    # Step 4: Save results
    print("\n[3/3] Saving concept vocabulary...")
    
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    vocabulary = {
        "generated_at": str(Path(__file__).stat().st_mtime),
        "model": MODEL,
        "total_clusters": result.topic_count,
        "concepts_generated": len(all_concepts),
        "concepts": sorted(all_concepts, key=lambda c: c["name"]),
    }
    
    with open(output_path, "w") as f:
        json.dump(vocabulary, f, indent=2)
    
    print(f"      Saved {len(all_concepts)} concepts to {OUTPUT_PATH}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total clusters discovered: {result.topic_count}")
    print(f"High-quality clusters: {len(clusters)}")
    print(f"Concepts generated: {len(all_concepts)}")
    print(f"\nSample concepts:")
    for c in all_concepts[:15]:
        print(f"  - {c['name']}")
        print(f"    Terms: {c['representative_terms'][:5]}")


if __name__ == "__main__":
    main()
