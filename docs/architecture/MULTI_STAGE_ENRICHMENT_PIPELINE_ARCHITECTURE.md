# Multi-Stage Enrichment Pipeline Architecture

**Version**: 2.0.0  
**Status**: Proposed  
**Created**: December 16, 2025  
**Author**: AI Coding Platform Team  
**Corrected**: December 16, 2025 (Kitchen Brigade alignment)

---

## Executive Summary

The **Multi-Stage Enrichment Pipeline (MSEP)** transforms metadata enrichment from a sequential fallback cascade into a **parallel, layered system** that leverages all available enrichment tools simultaneously. This approach maximizes semantic richness, accuracy, and robustness while maintaining graceful degradation.

**Key Innovation**: Instead of treating SBERT, BERTopic, Hybrid Search, and TF-IDF as mutually exclusive alternatives, MSEP orchestrates them as **complementary components** that each contribute unique value to the enrichment output.

**⚠️ ARCHITECTURE CONSTRAINT**: Per the Kitchen Brigade model, MSEP orchestration lives in **ai-agents** (Expeditor). The `llm-document-enhancer` is a **CUSTOMER** — it only calls APIs and receives results.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Kitchen Brigade Integration](#kitchen-brigade-integration)
4. [Service Responsibilities](#service-responsibilities)
5. [Data Flow](#data-flow)
6. [Parallel Execution Model](#parallel-execution-model)
7. [Result Merging Strategy](#result-merging-strategy)
8. [Configuration](#configuration)
9. [Schema Definitions](#schema-definitions)
10. [Anti-Pattern Compliance](#anti-pattern-compliance)
11. [Expected Outcomes](#expected-outcomes)

---

## Problem Statement

### Current State Limitations

The existing enrichment uses a **sequential fallback cascade**:

```
API → Local SBERT → TF-IDF (emergency only)
```

**Current TF-IDF Duplication (Architecture Violation)**:

| Location | File | Purpose | Status |
|----------|------|---------|--------|
| **llm-document-enhancer** | `workflows/metadata_enrichment/scripts/semantic_similarity_engine.py` | Local TF-IDF fallback | ❌ VIOLATES Kitchen Brigade (Customer has enricher logic) |
| **Code-Orchestrator-Service** | `src/models/sbert/semantic_similarity_engine.py` | Internal SBERT fallback | ✅ Correct location (Sous Chef) |
| **Code-Orchestrator-Service** | `src/models/bertopic_clusterer.py` | BERTopic KMeans fallback | ✅ Correct location (Sous Chef) |

**Issues with Current Approach:**

| Issue | Impact |
|-------|--------|
| **Sequential Execution** | Latency grows linearly; resources underutilized |
| **Mutually Exclusive** | Only one method contributes to final output |
| **TF-IDF Underutilized** | Cheap keyword extraction wasted as fallback-only |
| **TF-IDF DUPLICATED** | Same logic in llm-document-enhancer AND Code-Orchestrator |
| **No `/api/v1/keywords` endpoint** | TF-IDF not exposed as standalone API |
| **No Topic Context** | BERTopic available but not integrated |
| **No Graph Context** | Hybrid search (Neo4j+Qdrant) not leveraged |
| **Fixed Thresholds** | One-size-fits-all doesn't adapt to corpus diversity |

### Target State

A **parallel, multi-stage pipeline** where:

1. **SBERT API** provides baseline similarity scores (Code-Orchestrator-Service)
2. **BERTopic** discovers topics and boosts same-topic matches (Code-Orchestrator-Service)
3. **TF-IDF** extracts keywords (Code-Orchestrator-Service)
4. **Hybrid Search** adds graph relationships (semantic-search-service)
5. **Results are merged** with provenance tracking (ai-agents)

---

## Solution Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE ENRICHMENT PIPELINE (MSEP)                        │
│                    Parallel Execution with Intelligent Merging                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CUSTOMER (llm-document-enhancer)                                               │
│    └─→ POST /v1/agents/enrich-metadata                                          │
│                                                                                  │
│  INPUT: Corpus (list[str]) + Chapter Index (list[ChapterMeta])                  │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                 EXPEDITOR (ai-agents :8082)                              │    │
│  │                 Orchestrates MSEP Workflow                               │    │
│  │                                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │                STAGE 1: PARALLEL ENRICHMENT                      │   │    │
│  │  │                                                                   │   │    │
│  │  │  ┌────────────────────────────────────────────────────────────┐ │   │    │
│  │  │  │     SOUS CHEF (Code-Orchestrator-Service :8083)             │ │   │    │
│  │  │  │     ┌──────────┐  ┌──────────┐  ┌──────────┐               │ │   │    │
│  │  │  │     │  SBERT   │  │ BERTopic │  │  TF-IDF  │               │ │   │    │
│  │  │  │     │/embeddings│ │/topics   │  │/keywords │               │ │   │    │
│  │  │  │     │/similarity│ │/cluster  │  │          │               │ │   │    │
│  │  │  │     │ REQUIRED │  │ REQUIRED │  │ REQUIRED │               │ │   │    │
│  │  │  │     └──────────┘  └──────────┘  └──────────┘               │ │   │    │
│  │  │  └────────────────────────────────────────────────────────────┘ │   │    │
│  │  │                                                                   │   │    │
│  │  │  ┌────────────────────────────────────────────────────────────┐ │   │    │
│  │  │  │     COOKBOOK (semantic-search-service :8081)                │ │   │    │
│  │  │  │     ┌───────────────────┐                                   │ │   │    │
│  │  │  │     │   Hybrid Search   │                                   │ │   │    │
│  │  │  │     │/search (Qdrant)   │                                   │ │   │    │
│  │  │  │     │/relationships     │                                   │ │   │    │
│  │  │  │     │ (Neo4j) OPTIONAL  │                                   │ │   │    │
│  │  │  │     └───────────────────┘                                   │ │   │    │
│  │  │  └────────────────────────────────────────────────────────────┘ │   │    │
│  │  │                                                                   │   │    │
│  │  │         asyncio.gather() - Parallel Execution                    │   │    │
│  │  └───────────────────────────────────────────────────────────────────┘   │    │
│  │                                    │                                      │    │
│  │  ┌─────────────────────────────────┼─────────────────────────────────┐   │    │
│  │  │            STAGE 2: RESULT MERGER (ai-agents)                     │   │    │
│  │  │                                 ▼                                  │   │    │
│  │  │  1. Compute base similarity matrix (SBERT)                        │   │    │
│  │  │  2. Apply topic boost (+0.15 for same-topic chapters)             │   │    │
│  │  │  3. Merge TF-IDF keywords with SBERT-generated concepts           │   │    │
│  │  │  4. Add graph relationships (if available)                        │   │    │
│  │  │  5. Apply dynamic threshold (corpus-size aware)                   │   │    │
│  │  │  6. Deduplicate and rank by final score                           │   │    │
│  │  │  7. Track provenance (which method contributed)                   │   │    │
│  │  └───────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                          │    │
│  └──────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                          │
│  OUTPUT: EnrichedMetadata with cross_references, keywords, topics, provenance   │
│                                                                                  │
│  CUSTOMER receives result and writes to {book}_enriched.json                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Kitchen Brigade Integration

**⚠️ CRITICAL**: This section defines correct service boundaries per `AI_CODING_PLATFORM_ARCHITECTURE.md`.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KITCHEN BRIGADE: MSEP INTEGRATION                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  👤 CUSTOMER (llm-document-enhancer)                                            │
│     └─→ POST /v1/agents/enrich-metadata (corpus, chapter_index, config)        │
│     └─→ Receives EnrichedMetadata response                                      │
│     └─→ Writes {book}_enriched.json                                             │
│     └─→ Does NOT orchestrate enrichment                                         │
│     └─→ Does NOT implement enrichers                                            │
│                                                                                  │
│  📋 EXPEDITOR (ai-agents :8082) ← MSEP ORCHESTRATION LIVES HERE                │
│     └─→ Receives enrichment request from CUSTOMER                              │
│     └─→ Calls Code-Orchestrator for SBERT + BERTopic + TF-IDF                  │
│     └─→ Calls semantic-search-service for hybrid search                         │
│     └─→ Merges results from all sources                                         │
│     └─→ Returns EnrichedMetadata to CUSTOMER                                    │
│                                                                                  │
│  👨‍🍳 SOUS CHEF (Code-Orchestrator-Service :8083)                                │
│     ├─→ POST /api/v1/embeddings     ← SBERT embeddings                          │
│     ├─→ POST /api/v1/similarity     ← Cosine similarity matrix                  │
│     ├─→ POST /api/v1/topics         ← BERTopic topic discovery                  │
│     ├─→ POST /api/v1/cluster        ← Document-topic assignments                │
│     └─→ POST /api/v1/keywords       ← TF-IDF keyword extraction (NEW)          │
│                                                                                  │
│  📖 COOKBOOK (semantic-search-service :8081)                                    │
│     ├─→ POST /api/v1/search         ← Hybrid vector+graph search                │
│     └─→ GET /api/v1/relationships   ← Graph traversal (Neo4j)                   │
│                                                                                  │
│  🚪 ROUTER (llm-gateway :8080)                                                  │
│     └─→ NOT USED for internal platform calls                                    │
│     └─→ Only for LLM inference from outside AI platform                         │
│                                                                                  │
│  🗄️ PANTRY (ai-platform-data)                                                   │
│     ├─→ Neo4j: Graph relationships (PARALLEL, PERPENDICULAR, SKIP_TIER)        │
│     └─→ Qdrant: Vector embeddings for semantic search                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Service Responsibility Matrix for MSEP

| Service | Role | MSEP Responsibility | What It Does NOT Do |
|---------|------|---------------------|---------------------|
| **llm-document-enhancer** | Customer | Call API, receive result, write JSON | Orchestrate, implement enrichers, merge results |
| **ai-agents** | Expeditor | **Orchestrate MSEP**, call services, merge results | Host models, execute searches directly |
| **Code-Orchestrator** | Sous Chef | SBERT, BERTopic, TF-IDF APIs | Store content, merge results |
| **semantic-search** | Cookbook | Hybrid search (Qdrant + Neo4j) | Generate keywords, rank results |
| **llm-gateway** | Router | N/A for MSEP (internal platform) | Participate in MSEP |

---

## Service Responsibilities

### Service 1: Code-Orchestrator-Service (Port 8083)

**Provides**: SBERT embeddings, similarity matrix, BERTopic topics/clustering, TF-IDF keywords

| Endpoint | Method | Purpose | Input | Output | Status |
|----------|--------|---------|-------|--------|--------|
| `/api/v1/embeddings` | POST | Generate SBERT embeddings | `{"texts": [...]}` | `{"embeddings": [[...], ...]}` | ✅ EXISTS |
| `/api/v1/similarity` | POST | Compute similarity matrix | `{"texts": [...]}` or `{"embeddings": [...]}` | `{"similarity_matrix": [[...]]}` | ✅ EXISTS |
| `/api/v1/topics` | POST | Discover BERTopic topics | `{"corpus": [...], "min_topic_size": 5}` | `{"topics": [...], "topic_count": N}` | ✅ EXISTS |
| `/api/v1/cluster` | POST | Assign documents to topics | `{"corpus": [...], "chapter_index": [...]}` | `{"assignments": [...]}` | ✅ EXISTS |
| `/api/v1/keywords` | POST | Extract TF-IDF keywords | `{"corpus": [...], "top_k": 10}` | `{"keywords": [[...], ...]}` | ⚠️ **TO BE CREATED** |

**⚠️ IMPLEMENTATION NOTE for `/api/v1/keywords`**:

The TF-IDF implementation **already exists** in Code-Orchestrator-Service at:
- `src/models/sbert/semantic_similarity_engine.py` → `TfidfVectorizer` (currently used as SBERT fallback)
- `src/models/bertopic_clusterer.py` → `TfidfVectorizer` (used for KMeans fallback clustering)

**To create the `/api/v1/keywords` endpoint (WBS Phase MSE-1)**:
1. Extract keyword extraction logic into `src/models/tfidf_extractor.py`
2. Create `src/api/keywords.py` with the POST endpoint
3. Register router in `src/main.py`
4. Reuse existing `TfidfVectorizer` configuration from `semantic_similarity_engine.py`

**Current TF-IDF DUPLICATION to resolve**:
- `llm-document-enhancer/workflows/metadata_enrichment/scripts/semantic_similarity_engine.py` — **SHOULD BE REMOVED** after MSE-6
- `Code-Orchestrator-Service/src/models/sbert/semantic_similarity_engine.py` — **CANONICAL LOCATION**

**Anti-Patterns Mitigated**:
- S1192: Model name in constant `DEFAULT_MODEL_NAME`
- #12: Connection pooling via single model instance

---

### Service 2: semantic-search-service (Port 8081)

**Provides**: Hybrid vector+graph search, graph relationships

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/api/v1/search` | POST | Hybrid search | `{"query": "...", "top_k": 10}` | `{"results": [...]}` |
| `/api/v1/relationships` | GET | Get graph relationships | `?chapter_id=...` | `{"relationships": [...]}` |

**Fallback Strategy**:
- If Neo4j unavailable: Use Qdrant-only (vector search)
- If both unavailable: Return empty relationships, MSEP continues without graph context

---

### Service 3: ai-agents (Port 8082) — MSEP Orchestrator

**Provides**: MSEP orchestration endpoint

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/v1/agents/enrich-metadata` | POST | MSEP orchestration | `MSEPRequest` | `EnrichedMetadata` |

**MSEP Request Schema**:
```json
{
  "corpus": ["Chapter 1 text...", "Chapter 2 text..."],
  "chapter_index": [
    {"book": "arch_patterns.json", "chapter": 1, "title": "Domain Modeling"}
  ],
  "config": {
    "enable_topic_boost": true,
    "enable_hybrid_search": true,
    "base_threshold": 0.5,
    "top_k": 5
  }
}
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MSEP DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CUSTOMER (llm-document-enhancer)                                               │
│  ─────────────────────────────────                                              │
│  1. Loads book content + chapter metadata                                       │
│  2. POST /v1/agents/enrich-metadata (corpus, chapter_index, config)            │
│                                                                                  │
│                          │                                                       │
│                          ▼                                                       │
│                                                                                  │
│  EXPEDITOR (ai-agents :8082)                                                    │
│  ───────────────────────────                                                    │
│  3. Receives request, validates input                                           │
│  4. Dispatches PARALLEL enrichment tasks:                                       │
│                                                                                  │
│         ┌─────────────────────────────────────────────────────────┐             │
│         │              asyncio.gather()                           │             │
│         │                                                         │             │
│         │  Task A: POST Code-Orchestrator/api/v1/embeddings       │             │
│         │          POST Code-Orchestrator/api/v1/similarity       │             │
│         │          → Returns: similarity_matrix                   │             │
│         │                                                         │             │
│         │  Task B: POST Code-Orchestrator/api/v1/topics           │             │
│         │          POST Code-Orchestrator/api/v1/cluster          │             │
│         │          → Returns: topic_assignments                   │             │
│         │                                                         │             │
│         │  Task C: POST Code-Orchestrator/api/v1/keywords         │             │
│         │          → Returns: keywords per chapter                │             │
│         │                                                         │             │
│         │  Task D: POST semantic-search/api/v1/search (OPTIONAL)  │             │
│         │          GET semantic-search/api/v1/relationships       │             │
│         │          → Returns: graph_relationships                 │             │
│         │                                                         │             │
│         └─────────────────────────────────────────────────────────┘             │
│                                                                                  │
│  5. MERGE RESULTS (in ai-agents):                                               │
│     • base_score = similarity_matrix[i][j]                                      │
│     • topic_boost = 0.15 if same_topic else 0.0                                 │
│     • final_score = base_score + topic_boost                                    │
│     • merged_keywords = deduplicate(tfidf ∪ sbert)                              │
│     • graph_context = relationships (if available)                              │
│     • apply threshold, sort by final_score                                      │
│     • attach provenance                                                         │
│                                                                                  │
│  6. Return EnrichedMetadata to CUSTOMER                                         │
│                                                                                  │
│                          │                                                       │
│                          ▼                                                       │
│                                                                                  │
│  CUSTOMER (llm-document-enhancer)                                               │
│  ─────────────────────────────────                                              │
│  7. Receives EnrichedMetadata                                                   │
│  8. Writes {book}_enriched.json to ai-platform-data/books/enriched/            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Parallel Execution Model

### Async Architecture (in ai-agents)

MSEP uses Python's `asyncio` for parallel execution **within the ai-agents service**:

```python
# Located in: ai-agents/src/agents/msep/orchestrator.py

async def enrich_metadata(
    request: MSEPRequest,
    code_orchestrator: CodeOrchestratorClient,
    semantic_search: SemanticSearchClient,
) -> EnrichedMetadata:
    """Execute MSEP pipeline - orchestrated by ai-agents (Expeditor)."""
    
    corpus = request.corpus
    chapter_index = request.chapter_index
    config = request.config
    
    # Stage 1: Parallel enrichment via service calls
    sbert_task = asyncio.create_task(
        code_orchestrator.get_similarity_matrix(corpus)
    )
    tfidf_task = asyncio.create_task(
        code_orchestrator.extract_keywords(corpus, top_k=10)
    )
    bertopic_task = asyncio.create_task(
        code_orchestrator.cluster_topics(corpus, chapter_index)
    )
    
    # Hybrid search is conditional
    hybrid_task = None
    if config.enable_hybrid_search:
        hybrid_task = asyncio.create_task(
            semantic_search.get_relationships_batch(
                [ch.id for ch in chapter_index]
            )
        )
    
    # Wait for all tasks
    sbert_result, tfidf_result, bertopic_result = await asyncio.gather(
        sbert_task, tfidf_task, bertopic_task
    )
    
    hybrid_result = None
    if hybrid_task:
        try:
            hybrid_result = await asyncio.wait_for(hybrid_task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Hybrid search timed out, proceeding without graph context")
    
    # Stage 2: Merge results
    return merge_results(
        sbert_result, tfidf_result, bertopic_result, hybrid_result,
        chapter_index, config
    )
```

### Error Handling

Each parallel task has independent error handling:

| Task | On Error | Fallback |
|------|----------|----------|
| SBERT (similarity) | Retry 3x, then fail | Critical - no fallback |
| TF-IDF (keywords) | Log & continue | Empty keywords |
| BERTopic (topics) | Retry 3x, then skip | No topic boost |
| Hybrid Search | Timeout after 30s | Skip graph context |

---

## Result Merging Strategy

### Algorithm (in ai-agents)

```python
# Located in: ai-agents/src/agents/msep/merger.py

def merge_results(
    sbert: SBERTResult,
    tfidf: TFIDFResult,
    bertopic: BERTopicResult | None,
    hybrid: HybridResult | None,
    chapter_index: list[ChapterMeta],
    config: MSEPConfig
) -> EnrichedMetadata:
    """Merge results from all enrichers with priority-based scoring.
    
    This function runs in ai-agents (Expeditor) - NOT in llm-document-enhancer.
    """
    
    enriched_chapters = []
    
    for i, chapter in enumerate(chapter_index):
        # 1. Start with SBERT similarity scores
        cross_refs = []
        for j, score in enumerate(sbert.similarity_matrix[i]):
            if i == j:
                continue  # Skip self
            
            # 2. Apply topic boost
            topic_boost = 0.0
            if bertopic and bertopic.assignments[i] == bertopic.assignments[j]:
                topic_boost = config.same_topic_boost  # Default: 0.15
            
            final_score = score + topic_boost
            
            # 3. Apply dynamic threshold
            threshold = compute_dynamic_threshold(
                sbert.similarity_matrix,
                config.base_threshold,
                config.use_dynamic_threshold
            )
            
            if final_score >= threshold:
                cross_refs.append(CrossReference(
                    target=chapter_index[j],
                    score=final_score,
                    base_score=score,
                    topic_boost=topic_boost,
                    method="sbert+bertopic" if topic_boost > 0 else "sbert"
                ))
        
        # 4. Add graph relationships (if available)
        graph_rels = []
        if hybrid and chapter.id in hybrid.relationships:
            graph_rels = hybrid.relationships[chapter.id]
        
        # 5. Merge keywords
        merged_keywords = MergedKeywords(
            tfidf=tfidf.keywords[i] if tfidf else [],
            semantic=[],  # SBERT concepts extracted separately if needed
            merged=tfidf.keywords[i] if tfidf else []
        )
        
        # 6. Build enriched chapter
        enriched_chapters.append(EnrichedChapter(
            book=chapter.book,
            chapter=chapter.chapter,
            title=chapter.title,
            cross_references=sorted(cross_refs, key=lambda x: -x.score)[:config.top_k],
            keywords=merged_keywords,
            topic_id=bertopic.assignments[i] if bertopic else None,
            topic_name=bertopic.topics[bertopic.assignments[i]].name if bertopic else None,
            graph_relationships=graph_rels,
            provenance=Provenance(
                methods_used=["sbert", "tfidf"] + 
                             (["bertopic"] if bertopic else []) + 
                             (["hybrid"] if hybrid else []),
                sbert_score=score,
                topic_boost=topic_boost,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        ))
    
    return EnrichedMetadata(chapters=enriched_chapters)
```

---

## Configuration

### Environment Variables (ai-agents)

```bash
# MSEP Feature Flags
MSEP_ENABLE_PARALLEL=true
MSEP_ENABLE_TOPIC_BOOST=true
MSEP_ENABLE_HYBRID_SEARCH=true
MSEP_ENABLE_DYNAMIC_THRESHOLD=true

# Service URLs (ai-agents → other services)
CODE_ORCHESTRATOR_URL=http://localhost:8083
SEMANTIC_SEARCH_URL=http://localhost:8081

# Timeouts
CODE_ORCHESTRATOR_TIMEOUT=30.0
SEMANTIC_SEARCH_TIMEOUT=30.0

# Thresholds
MSEP_BASE_THRESHOLD=0.5
MSEP_SAME_TOPIC_BOOST=0.15
MSEP_TOP_K=5
```

### Config Dataclass (ai-agents)

```python
# Located in: ai-agents/src/agents/msep/config.py

@dataclass(frozen=True)
class MSEPConfig:
    """Configuration for Multi-Stage Enrichment Pipeline.
    
    Lives in ai-agents - NOT in llm-document-enhancer.
    """
    
    # Feature flags
    enable_parallel: bool = True
    enable_topic_boost: bool = True
    enable_hybrid_search: bool = True
    use_dynamic_threshold: bool = True
    
    # Service URLs
    code_orchestrator_url: str = "http://localhost:8083"
    semantic_search_url: str = "http://localhost:8081"
    
    # Timeouts
    code_orchestrator_timeout: float = 30.0
    semantic_search_timeout: float = 30.0
    
    # Thresholds
    base_threshold: float = 0.5
    same_topic_boost: float = 0.15
    top_k: int = 5
    
    @classmethod
    def from_env(cls) -> "MSEPConfig":
        """Load config from environment variables."""
        return cls(
            enable_parallel=os.getenv("MSEP_ENABLE_PARALLEL", "true").lower() == "true",
            enable_topic_boost=os.getenv("MSEP_ENABLE_TOPIC_BOOST", "true").lower() == "true",
            enable_hybrid_search=os.getenv("MSEP_ENABLE_HYBRID_SEARCH", "true").lower() == "true",
            use_dynamic_threshold=os.getenv("MSEP_ENABLE_DYNAMIC_THRESHOLD", "true").lower() == "true",
            code_orchestrator_url=os.getenv("CODE_ORCHESTRATOR_URL", "http://localhost:8083"),
            semantic_search_url=os.getenv("SEMANTIC_SEARCH_URL", "http://localhost:8081"),
            base_threshold=float(os.getenv("MSEP_BASE_THRESHOLD", "0.5")),
            same_topic_boost=float(os.getenv("MSEP_SAME_TOPIC_BOOST", "0.15")),
            top_k=int(os.getenv("MSEP_TOP_K", "5")),
        )
```

---

## Schema Definitions

### Input Schemas (ai-agents)

```python
# Located in: ai-agents/src/agents/msep/schemas.py

@dataclass
class ChapterMeta:
    """Metadata for a single chapter."""
    book: str
    chapter: int
    title: str
    id: str = field(default="")
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.book}:ch{self.chapter}"


@dataclass
class MSEPRequest:
    """Request to MSEP endpoint."""
    corpus: list[str]
    chapter_index: list[ChapterMeta]
    config: MSEPConfig = field(default_factory=MSEPConfig)
```

### Output Schemas (ai-agents)

```python
@dataclass
class CrossReference:
    """A cross-reference to another chapter."""
    target: ChapterMeta
    score: float
    base_score: float
    topic_boost: float
    method: str  # "sbert", "sbert+bertopic"


@dataclass
class MergedKeywords:
    """Keywords from multiple sources."""
    tfidf: list[str]
    semantic: list[str]
    merged: list[str]


@dataclass
class Provenance:
    """Tracking which methods contributed to enrichment."""
    methods_used: list[str]
    sbert_score: float
    topic_boost: float
    timestamp: str


@dataclass
class EnrichedChapter:
    """Enriched metadata for a single chapter."""
    book: str
    chapter: int
    title: str
    cross_references: list[CrossReference]
    keywords: MergedKeywords
    topic_id: int | None
    topic_name: str | None
    graph_relationships: list[str]
    provenance: Provenance


@dataclass
class EnrichedMetadata:
    """Complete enriched metadata response."""
    chapters: list[EnrichedChapter]
    total_cross_references: int = field(default=0)
    processing_time_ms: float = field(default=0.0)
    
    def __post_init__(self):
        self.total_cross_references = sum(
            len(ch.cross_references) for ch in self.chapters
        )
```

---

## Anti-Pattern Compliance

Per `CODING_PATTERNS_ANALYSIS.md` and `Comp_Static_Analysis_Report_20251203.md`:

| Rule | Applies To | Mitigation | Location |
|------|-----------|------------|----------|
| S1192 | Constants | All thresholds in `MSEPConfig` or constants module | ai-agents |
| S3776 | Cognitive complexity | Extract helper methods (`merge_results`, `compute_threshold`) | ai-agents |
| #7 | Exception shadowing | Namespaced exceptions (`MSEPError`, `EnrichmentTimeoutError`) | ai-agents |
| #12 | Connection pooling | Single `httpx.AsyncClient` per service client | ai-agents |
| #42/#43 | Async/await | Proper `async with` context managers | ai-agents |
| #2.2 | Type hints | All schemas use `@dataclass` with full type annotations | ai-agents |

---

## Expected Outcomes

| Metric | Before (Sequential) | After (MSEP) |
|--------|--------------------|--------------| 
| **Methods Used** | 1 (fallback cascade) | 3-4 (parallel) |
| **Cross-References** | SBERT-only | SBERT + topic boost + graph |
| **Keywords** | TF-IDF fallback only | TF-IDF always (merged) |
| **Latency** | Sum of all methods | Max of parallel methods |
| **Provenance** | None | Full method tracking |
| **Graceful Degradation** | Cascade failure | Independent failure |

---

## References

| Document | Location | Relevance |
|----------|----------|-----------|
| AI Platform Architecture | `textbooks/pending/platform/AI_CODING_PLATFORM_ARCHITECTURE.md` | Kitchen Brigade model |
| ai-agents Architecture | `ai-agents/docs/ARCHITECTURE.md` | Expeditor role |
| Code-Orchestrator Architecture | `Code-Orchestrator-Service/docs/ARCHITECTURE.md` | SBERT/BERTopic APIs |
| semantic-search Architecture | `semantic-search-service/docs/ARCHITECTURE.md` | Hybrid search |
| llm-document-enhancer Architecture | `llm-document-enhancer/docs/reference/ARCHITECTURE.md` | Customer role |
| TECHNICAL_CHANGE_LOG (Platform) | `textbooks/pending/platform/TECHNICAL_CHANGE_LOG.md` | CL-009, CL-010 |
| CODING_PATTERNS_ANALYSIS | `textbooks/Guidelines/CODING_PATTERNS_ANALYSIS.md` | Anti-patterns |
| Comp_Static_Analysis_Report | `llm-gateway/docs/Comp_Static_Analysis_Report_20251203.md` | Code quality |

---

*Generated: December 16, 2025 | Version: 2.0.0 | Kitchen Brigade Aligned*
