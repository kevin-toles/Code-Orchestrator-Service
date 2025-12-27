#!/usr/bin/env python3
"""
Validate taxonomy terms using LLM consensus (DeepSeek + ChatGPT).
Uses discussion-style prompting where LLMs reason about each term.

Terms are validated if BOTH LLMs agree they are legitimate technical concepts.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
import requests

# Paths
INPUT_FILE = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/taxonomy_terms_to_validate.json")
OUTPUT_FILE = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/taxonomy_llm_validation.json")
VALIDATED_FILTER = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/validated_term_filter.json")

# LLM Gateway
LLM_GATEWAY = "http://localhost:8080/v1/chat/completions"

# Model names - GPT 5.2 and DeepSeek-V3.2 reasoning mode
MODEL_DEEPSEEK = "deepseek-reasoner"
MODEL_GPT = "gpt-5.2"

# Batch size for efficiency
BATCH_SIZE = 25


def get_validation_prompt(terms_batch: list[dict]) -> str:
    """Create a discussion-style prompt for term validation."""
    terms_list = "\n".join([f"- {t['term']} (domain: {t['domain']})" for t in terms_batch])
    
    return f"""You are a senior software engineer reviewing terms for a technical knowledge base.

For each term below, determine if it is a **specific, meaningful technical concept** that would be useful for categorizing software engineering content.

KEEP terms that are:
- Specific technical concepts (e.g., "dependency injection", "kubernetes", "backpropagation")
- Named technologies, frameworks, or tools (e.g., "pytest", "terraform", "langchain")
- Well-defined patterns or practices (e.g., "circuit breaker", "blue-green deployment")

REJECT terms that are:
- Too generic/common words (e.g., "list", "new", "build", "include", "set")
- Ambiguous without context (e.g., "index", "table", "class", "def")
- Not specifically technical (e.g., "production", "environment", "response")

For each term, provide:
1. Your verdict: KEEP or REJECT
2. Brief reasoning (1 sentence)

Terms to evaluate:
{terms_list}

Respond in JSON format:
{{
  "evaluations": [
    {{"term": "term_name", "verdict": "KEEP/REJECT", "reasoning": "brief explanation"}}
  ]
}}"""


def call_llm(model: str, prompt: str, max_retries: int = 3) -> dict:
    """Call LLM gateway with retries."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                LLM_GATEWAY,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                timeout=120
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # Parse JSON from response
                # Handle markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                return json.loads(content)
            else:
                print(f"    LLM error ({model}): {response.status_code}")
                
        except json.JSONDecodeError as e:
            print(f"    JSON parse error ({model}): {e}")
        except Exception as e:
            print(f"    Request error ({model}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    return None


def validate_batch(terms_batch: list[dict]) -> dict:
    """Validate a batch of terms with both LLMs."""
    prompt = get_validation_prompt(terms_batch)
    
    # Call both LLMs
    print(f"  Calling DeepSeek-Reasoner...", end=" ", flush=True)
    deepseek_result = call_llm(MODEL_DEEPSEEK, prompt)
    print("✓" if deepseek_result else "✗")
    
    print(f"  Calling GPT-5.2...", end=" ", flush=True)
    chatgpt_result = call_llm(MODEL_GPT, prompt)
    print("✓" if chatgpt_result else "✗")
    
    # Combine results
    results = {}
    
    if deepseek_result and "evaluations" in deepseek_result:
        for eval in deepseek_result["evaluations"]:
            term = eval["term"].lower()
            results[term] = {
                "deepseek": eval["verdict"],
                "deepseek_reasoning": eval.get("reasoning", "")
            }
    
    if chatgpt_result and "evaluations" in chatgpt_result:
        for eval in chatgpt_result["evaluations"]:
            term = eval["term"].lower()
            if term not in results:
                results[term] = {}
            results[term]["chatgpt"] = eval["verdict"]
            results[term]["chatgpt_reasoning"] = eval.get("reasoning", "")
    
    return results


def main():
    print("=" * 70)
    print("TAXONOMY TERM VALIDATION (LLM CONSENSUS)")
    print("=" * 70)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Load terms to validate
    with open(INPUT_FILE) as f:
        terms = json.load(f)
    
    print(f"Terms to validate: {len(terms)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches: {(len(terms) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print()
    
    # Process in batches
    all_results = {}
    start_time = datetime.now()
    
    for i in range(0, len(terms), BATCH_SIZE):
        batch = terms[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(terms) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"[Batch {batch_num}/{total_batches}] Processing {len(batch)} terms...")
        
        results = validate_batch(batch)
        all_results.update(results)
        
        # Rate limiting
        if i + BATCH_SIZE < len(terms):
            time.sleep(1)
    
    # Analyze consensus
    keep_both = []
    keep_deepseek_only = []
    keep_chatgpt_only = []
    reject_both = []
    
    for term, result in all_results.items():
        ds = result.get("deepseek", "").upper()
        gpt = result.get("chatgpt", "").upper()
        
        if "KEEP" in ds and "KEEP" in gpt:
            keep_both.append(term)
        elif "KEEP" in ds:
            keep_deepseek_only.append(term)
        elif "KEEP" in gpt:
            keep_chatgpt_only.append(term)
        else:
            reject_both.append(term)
    
    elapsed = datetime.now() - start_time
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Time elapsed: {elapsed}")
    print()
    print(f"✅ CONSENSUS KEEP (both agree): {len(keep_both)}")
    print(f"❌ CONSENSUS REJECT (both agree): {len(reject_both)}")
    print(f"🔶 DeepSeek only KEEP: {len(keep_deepseek_only)}")
    print(f"🔷 ChatGPT only KEEP: {len(keep_chatgpt_only)}")
    print()
    
    # Save detailed results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_terms": len(terms),
        "consensus_keep": keep_both,
        "consensus_reject": reject_both,
        "deepseek_only_keep": keep_deepseek_only,
        "chatgpt_only_keep": keep_chatgpt_only,
        "detailed_results": all_results
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Detailed results saved to: {OUTPUT_FILE}")
    print()
    
    # Show consensus KEEP terms
    if keep_both:
        print("CONSENSUS KEEP terms:")
        for term in sorted(keep_both):
            print(f"  ✅ {term}")
    
    print()
    print("Next steps:")
    print("  1. Review results in the output file")
    print("  2. Run: python3 scripts/add_validated_taxonomy_terms.py")


if __name__ == "__main__":
    main()
