# Code Understanding Orchestrator Service

## Executive Summary

A standalone microservice that coordinates multiple specialized code understanding models (CodeT5+, GraphCodeBERT, CodeBERT) to dynamically extract, validate, and rank search terms from natural language queries. This service replaces hardcoded keyword mappings with intelligent, context-aware term generation.

This service acts as the **"Sous Chef"** in the Kitchen Brigade architecture—interpreting orders (queries), preparing ingredients (keywords), curating results, and auditing output before serving to the customer.

---

## Kitchen Brigade Architecture Model

### The Analogy

The platform follows a **Kitchen Brigade** organizational model where each service has a specific role:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          🍽️  KITCHEN BRIGADE MODEL                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  👤 CUSTOMER (Claude/GPT/User)                                              │
│     └─→ Places order: "I need code for document chunking with overlap"      │
│                                                                              │
│  👨‍🍳 SOUS CHEF (Code Understanding Orchestrator) ← THIS SERVICE             │
│     └─→ SMART: Interprets the order                                         │
│     └─→ Extracts keywords/concepts using code understanding models          │
│     └─→ Sends keyword list to Cookbook                                      │
│                                                                              │
│  📖 COOKBOOK (Semantic Search Service) ← DUMB RETRIEVAL                     │
│     └─→ Takes keywords as INPUT (does NOT generate them)                    │
│     └─→ Queries vector DBs (Qdrant, Neo4j) where content lives              │
│     └─→ Returns ALL matches without filtering or judgment                   │
│     └─→ Just a retrieval engine - like looking up recipes in a book         │
│                                                                              │
│  👨‍🍳 CHEF DE PARTIE (Orchestrator - Curation Phase)                         │
│     └─→ Receives raw results from Cookbook                                  │
│     └─→ SMART: Filters out irrelevant results (C++ "chunk of memory")       │
│     └─→ Ranks by domain relevance                                           │
│     └─→ Prepares curated instructions for Line Cook                         │
│                                                                              │
│  👨‍🍳 LINE COOK (Code Generation Model via LLM Gateway)                      │
│     └─→ Receives curated context + instructions                             │
│     └─→ Generates actual code from the instructions                         │
│                                                                              │
│  👨‍🍳 CHEF DE PARTIE (Orchestrator - Audit Phase)                            │
│     └─→ Validates generated code quality                                    │
│     └─→ Ensures code matches original intent                                │
│                                                                              │
│  👤 CUSTOMER receives the final plated dish (working code)                  │
│     └─→ Implements the code in their project                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Service Responsibility Matrix

| Service | Role | Intelligence | What It Does | What It Does NOT Do |
|---------|------|--------------|--------------|---------------------|
| **LLM Gateway** | Router | Routing only | Routes requests to appropriate models | Make decisions about content |
| **Code Understanding Orchestrator** | Sous Chef + Chef de Partie | **SMART** | Extracts keywords, curates results, audits output | Store content, execute searches |
| **Semantic Search Service** | Cookbook | **DUMB** | Takes keywords as input, queries vector DBs, returns all matches | Generate keywords, filter results, make judgments |
| **Code Generation Model** | Line Cook | Executor | Generates code from curated instructions | Decide what to generate |
| **Vector DBs (Qdrant/Neo4j)** | Pantry | Storage | Stores embeddings and relationships | Nothing else |

### Key Insight: Semantic Search is DUMB

The **Semantic Search Service** is intentionally dumb:
- It does NOT contain knowledge itself—it queries databases that contain knowledge
- It does NOT generate keywords—it receives them as input
- It does NOT filter results—it returns ALL matches
- It's just a query executor, like looking up recipes in a cookbook

The **intelligence lives in the Orchestrator**, which:
1. **Interprets** the customer's order (query understanding)
2. **Generates** the right keywords to search for
3. **Curates** the raw results (filters irrelevant matches)
4. **Instructs** the line cook (prepares context for code generation)
5. **Audits** the final output (validates generated code)

---

## Problem Statement

### Current State
The existing cross-reference system uses **hardcoded `FOCUS_SEARCH_TERMS`** mappings:

```python
FOCUS_SEARCH_TERMS = {
    "multi-stage chunking": [
        "chunk", "chunking", "split", "segment", ...  # Static, brittle
    ],
}
```

### Issues
1. **False Positives**: "chunk" matches C++ memory allocation ("chunk of memory") instead of LLM document chunking
2. **Not Portable**: Hardcoded terms don't transfer across taxonomies/domains
3. **Maintenance Burden**: Manual updates required for new concepts
4. **Limited Coverage**: Misses semantically related terms not in the list

### Proposed Solution
A multi-model orchestration service that dynamically generates contextually-relevant search terms.

---

## Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     👤 CUSTOMER (Claude/GPT/User)                            │
│                "I need code for document chunking with overlap"              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Request
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              👨‍🍳 CODE UNDERSTANDING ORCHESTRATOR (Sous Chef)                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         API Gateway                                    │  │
│  │                    /extract, /validate, /search                        │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │                                           │
│  ┌───────────────────────────────▼───────────────────────────────────┐  │
│  │                  Model Wrapper Orchestrator                            │  │
│  │                   (LangGraph State Machine)                            │  │
│  └───┬───────────────────────┬───────────────────────┬───────────────────┘  │
│      │                       │                       │                       │
│      ▼                       ▼                       ▼                       │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐                │
│  │  CodeT5+    │       │GraphCodeBERT│       │  CodeBERT   │                │
│  │  Extractor  │       │  Validator  │       │   Ranker    │                │
│  │ (Generator) │       │ (Validator) │       │  (Ranker)   │                │
│  └─────────────┘       └─────────────┘       └─────────────┘                │
│                                                                              │
│  Output: ["chunking", "text_splitter", "overlap", "RAG", "embedding"]       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Keywords (INPUT to Cookbook)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                📖 SEMANTIC SEARCH SERVICE (Cookbook) - DUMB                  │
│                                                                              │
│  Input:  Keywords from Orchestrator                                          │
│  Action: Query vector databases                                              │
│  Output: ALL matches (no filtering, no judgment)                            │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Qdrant        │  │   Neo4j Graph   │  │   Hybrid        │             │
│  │   Retriever     │  │   Retriever     │  │   Search        │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                        │
│           └────────────────────┼────────────────────┘                        │
│                                │                                             │
│           Returns: [C++ memory chunk, LLM chunking, game chunks, ...]       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Raw Results
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           👨‍🍳 ORCHESTRATOR (Chef de Partie) - Curation Phase                 │
│                                                                              │
│  ✓ Filter: Remove C++ "chunk of memory" (wrong domain)                      │
│  ✓ Rank: Score by relevance to LLM/AI context                               │
│  ✓ Prepare: Curated context for Line Cook                                   │
│                                                                              │
│  Output: Curated references + instructions for code generation              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Curated Context
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    👨‍🍳 LINE COOK (Code Generation Model)                     │
│                                                                              │
│  Input:  Curated context + generation instructions                          │
│  Action: Generate code based on best practices from references              │
│  Output: Working code implementation                                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Generated Code
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            👨‍🍳 ORCHESTRATOR (Chef de Partie) - Audit Phase                   │
│                                                                              │
│  ✓ Validate: Code quality checks                                            │
│  ✓ Verify: Matches original intent                                          │
│  ✓ Format: Prepare final output                                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ Final Result
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         👤 CUSTOMER receives final dish                      │
│                      (Working code ready to implement)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Where Content Actually Lives

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🗄️  DATA LAYER (Pantry)                           │
│                                                                              │
│  These are the ACTUAL STORAGE systems - where content lives:                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  QDRANT (Vector Database)                                            │   │
│  │  └─→ Stores: Document embeddings, chunk vectors                      │   │
│  │  └─→ Contains: Textbook content, code patterns, technical docs       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  NEO4J (Graph Database)                                              │   │
│  │  └─→ Stores: Relationships between concepts, cross-references        │   │
│  │  └─→ Contains: Book→Chapter→Section→Concept relationships           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  JSON FILES (Local Textbooks)                                        │   │
│  │  └─→ Stores: Raw textbook JSON files                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

The Semantic Search Service QUERIES these systems - it doesn't contain them.
```

---

## Multi-Model Coordination Flow

### Model Wrapper Orchestration Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Query                                       │
│          "LLM code understanding with multi-stage chunking for RAG"          │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR STATE MACHINE                            │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 1: EXTRACTION                                                     │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ CodeT5+ Extractor                                                    │ │ │
│  │ │                                                                       │ │ │
│  │ │ Input:  "Extract technical search terms for: LLM code understanding  │ │ │
│  │ │          with multi-stage chunking for RAG"                          │ │ │
│  │ │                                                                       │ │ │
│  │ │ Output: {                                                             │ │ │
│  │ │   "primary_terms": ["chunking", "RAG", "embedding", "LLM"],          │ │ │
│  │ │   "related_terms": ["tokenization", "vector", "retrieval"],          │ │ │
│  │ │   "code_patterns": ["text_splitter", "chunk_size", "overlap"]        │ │ │
│  │ │ }                                                                     │ │ │
│  │ └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 2: VALIDATION                                                      │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ GraphCodeBERT Validator                                              │ │ │
│  │ │                                                                       │ │ │
│  │ │ Input:  Generated terms + Original query + Domain context            │ │ │
│  │ │                                                                       │ │ │
│  │ │ Validation Rules:                                                     │ │ │
│  │ │   ✓ "chunking" - Valid (LLM context, not memory allocation)          │ │ │
│  │ │   ✓ "RAG" - Valid (retrieval augmented generation)                   │ │ │
│  │ │   ✓ "embedding" - Valid (vector representations)                     │ │ │
│  │ │   ✗ "split" - Rejected (too generic, high false positive rate)       │ │ │
│  │ │                                                                       │ │ │
│  │ │ Expansions Added:                                                     │ │ │
│  │ │   + "semantic_search" (related to RAG)                               │ │ │
│  │ │   + "context_window" (related to chunking)                           │ │ │
│  │ │   + "HNSW" (related to vector indexing)                              │ │ │
│  │ └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 3: RANKING                                                         │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ CodeBERT Ranker                                                      │ │ │
│  │ │                                                                       │ │ │
│  │ │ Input:  Validated terms + Original query embedding                   │ │ │
│  │ │                                                                       │ │ │
│  │ │ Similarity Scoring:                                                   │ │ │
│  │ │   1. chunking         → 0.95 (highest relevance)                     │ │ │
│  │ │   2. RAG              → 0.92                                         │ │ │
│  │ │   3. embedding        → 0.89                                         │ │ │
│  │ │   4. context_window   → 0.85                                         │ │ │
│  │ │   5. semantic_search  → 0.82                                         │ │ │
│  │ │   6. tokenization     → 0.78                                         │ │ │
│  │ │   7. vector           → 0.75                                         │ │ │
│  │ └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────┬───────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ STATE 4: CONSENSUS                                                       │ │
│  │                                                                           │ │
│  │ Agreement Filter: Terms must be approved by ≥2 models                    │ │
│  │                                                                           │ │
│  │ Final Output:                                                             │ │
│  │ {                                                                         │ │
│  │   "search_terms": [                                                       │ │
│  │     {"term": "chunking", "score": 0.95, "models_agreed": 3},             │ │
│  │     {"term": "RAG", "score": 0.92, "models_agreed": 3},                  │ │
│  │     {"term": "embedding", "score": 0.89, "models_agreed": 3},            │ │
│  │     {"term": "context_window", "score": 0.85, "models_agreed": 2},       │ │
│  │     {"term": "semantic_search", "score": 0.82, "models_agreed": 2}       │ │
│  │   ],                                                                      │ │
│  │   "excluded_terms": [                                                     │ │
│  │     {"term": "split", "reason": "Too generic", "models_agreed": 1}       │ │
│  │   ]                                                                       │ │
│  │ }                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Selection

### Keyword Extraction Pipeline (Sous Chef)

| Model | Role | Strength | HuggingFace ID |
|-------|------|----------|----------------|
| **CodeT5+** | Generator | Encoder-decoder architecture enables text generation; trained on NL↔Code pairs | `Salesforce/codet5p-220m` |
| **GraphCodeBERT** | Validator | Understands code structure via data flow graphs; catches semantic mismatches | `microsoft/graphcodebert-base` |
| **CodeBERT** | Ranker | Fast embeddings for similarity scoring; well-established baseline | `microsoft/codebert-base` |

### Code Generation (Line Cook)

| Model | Parameters | VRAM (BF16) | HumanEval | Notes |
|-------|------------|-------------|-----------|-------|
| **Qwen2.5-Coder-32B-Instruct** | 32B | ~64GB | 92.7% | Primary - Best open-source coding |
| **Qwen2.5-Coder-7B-Instruct** | 7.6B | ~16GB | ~73% | Fallback - Single GPU friendly |
| **DeepSeek Coder 33B-Instruct** | 33B | ~66GB | 79.3% | Alternative - Excellent multi-file reasoning |

### Model Comparison Matrix

```
┌────────────────────┬────────────────┬────────────────┬────────────────┐
│     Capability     │    CodeT5+     │ GraphCodeBERT  │    CodeBERT    │
├────────────────────┼────────────────┼────────────────┼────────────────┤
│ Text Generation    │       ✅       │       ❌       │       ❌       │
│ Code Structure     │       ⚠️       │       ✅       │       ⚠️       │
│ Embeddings         │       ✅       │       ✅       │       ✅       │
│ Zero-shot Ready    │       ✅       │       ⚠️       │       ⚠️       │
│ Parameters         │    220M-6B     │     125M       │     125M       │
│ Inference Speed    │    Medium      │     Fast       │     Fast       │
└────────────────────┴────────────────┴────────────────┴────────────────┘

Legend: ✅ Excellent  ⚠️ Partial  ❌ Not supported
```

---

## Service API Design

### REST Endpoints

```yaml
openapi: 3.0.0
info:
  title: Code Understanding Orchestrator API
  version: 1.0.0

paths:
  /api/v1/extract:
    post:
      summary: Extract search terms from query
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  example: "LLM code understanding with multi-stage chunking"
                domain:
                  type: string
                  example: "ai-ml"
                options:
                  type: object
                  properties:
                    min_confidence:
                      type: number
                      default: 0.7
                    max_terms:
                      type: integer
                      default: 10
                    require_consensus:
                      type: boolean
                      default: true
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ExtractionResult'

  /api/v1/validate:
    post:
      summary: Validate terms against domain context

  /api/v1/search:
    post:
      summary: Full pipeline - extract, validate, and search

  /api/v1/generate:
    post:
      summary: Generate code from curated context

components:
  schemas:
    ExtractionResult:
      type: object
      properties:
        search_terms:
          type: array
          items:
            type: object
            properties:
              term:
                type: string
              score:
                type: number
              models_agreed:
                type: integer
        excluded_terms:
          type: array
        metadata:
          type: object
          properties:
            processing_time_ms:
              type: integer
            models_used:
              type: array
```

---

## Use Cases

| Use Case | Description |
|----------|-------------|
| **Cross-Reference Enhancement** | Fix false positives like C++ "chunk of memory" vs LLM chunking |
| **Code Search** | Extract search terms from natural language queries about code |
| **Documentation Retrieval** | Find relevant docs based on technical questions |
| **API Discovery** | Match user intent to available API endpoints |
| **Codebase Q&A** | Power RAG systems for code understanding |
| **Code Review** | Identify related code patterns and best practices |

---

## Related Services

| Service | Repository | Role |
|---------|------------|------|
| **LLM Gateway** | `llm-gateway` | Routes requests to appropriate models |
| **Semantic Search Service** | `semantic-search-service` | Queries vector DBs (Cookbook) |
| **AI Agents** | `ai-agents` | Main orchestration layer |
| **LLM Document Enhancer** | `llm-document-enhancer` | Document processing pipeline |

---

## Next Steps

1. **Phase 1**: Basic FastAPI structure with health endpoints
2. **Phase 2**: Implement CodeT5+ Extractor (model wrapper)
3. **Phase 3**: Add GraphCodeBERT Validator (model wrapper)
4. **Phase 4**: Add CodeBERT Ranker (model wrapper)
5. **Phase 5**: Implement LangGraph orchestration
6. **Phase 6**: Add Line Cook (code generation) integration
7. **Phase 7**: Integration tests with semantic-search-service
8. **Phase 8**: Docker/Kubernetes deployment

---

## References

- [CodeT5+ Paper](https://arxiv.org/abs/2305.07922)
- [GraphCodeBERT Paper](https://arxiv.org/abs/2009.08366)
- [CodeBERT Paper](https://arxiv.org/abs/2002.08155)
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
