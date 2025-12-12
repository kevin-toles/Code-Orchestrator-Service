# Code-Orchestrator-Service: WBS Implementation Plan

## Executive Summary

This document defines the Work Breakdown Structure (WBS) for implementing the **Code Understanding Orchestrator Service** (Sous Chef) - the intelligent core of the Kitchen Brigade architecture that hosts HuggingFace models for keyword extraction, validation, and ranking.

**Target Outcome**: Enable cross-book metadata enrichment with semantic similarity scores that produce REAL cross-references (threshold 0.3-0.5), replacing the broken TF-IDF approach (threshold 0.7 impossible).

---

## Architecture Context

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          KITCHEN BRIGADE ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  👤 CUSTOMER: llm-document-enhancer                                         │
│     └─→ Calls POST /api/v1/search with chapter text                         │
│                                                                              │
│  👨‍🍳 SOUS CHEF: Code-Orchestrator-Service (THIS SERVICE, Port 8083)          │
│     └─→ Hosts CodeT5+ (Generator)                                           │
│     └─→ Hosts GraphCodeBERT (Validator)                                     │
│     └─→ Hosts CodeBERT (Ranker)                                             │
│     └─→ Extracts keywords → Validates → Ranks → Curates                     │
│                                                                              │
│  📖 COOKBOOK: Semantic Search Service (Port 8081)                           │
│     └─→ DUMB retrieval (takes keywords, returns all matches)                │
│                                                                              │
│  📋 EXPEDITOR: ai-agents (Port 8082)                                        │
│     └─→ High-level workflow orchestration                                   │
│                                                                              │
│  🚪 ROUTER: llm-gateway (Port 8080)                                         │
│     └─→ LLM inference routing                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## WBS Structure

### Phase 1: Project Scaffolding (Sprint 1, Days 1-3)

#### 1.1 Repository Setup
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 1.1.1 | Create project structure | `src/`, `tests/`, `docs/`, `config/` directories exist | `ls -la` shows structure |
| 1.1.2 | Initialize pyproject.toml | Poetry/pip installable, Python 3.11+ | `pip install -e .` succeeds |
| 1.1.3 | Create requirements.txt | transformers, torch, sentence-transformers, fastapi, langgraph | `pip install -r requirements.txt` succeeds |
| 1.1.4 | Create Dockerfile | Multi-stage build, CUDA support for GPU | `docker build -t code-orchestrator .` succeeds |
| 1.1.5 | Create docker-compose.yml | Service on port 8083, GPU passthrough | `docker-compose up` starts service |

#### 1.2 FastAPI Application Shell
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 1.2.1 | Create main.py entry point | FastAPI app with lifespan handler | `uvicorn src.main:app` starts |
| 1.2.2 | Implement /health endpoint | Returns 200 with service info | `curl localhost:8083/health` returns JSON |
| 1.2.3 | Implement /ready endpoint | Returns 503 until models loaded | Readiness probe passes after startup |
| 1.2.4 | Add structured logging | JSON logs with correlation IDs | Logs contain trace_id |
| 1.2.5 | Add OpenTelemetry tracing | Spans for each operation | Jaeger shows traces |

**Phase 1 Integration Test:**
```python
def test_phase1_service_starts():
    """Service starts and responds to health checks."""
    response = httpx.get("http://localhost:8083/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

### Phase 2: Model Loading Infrastructure (Sprint 1, Days 4-7)

#### 2.1 Model Registry
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 2.1.1 | Create ModelRegistry class | Singleton pattern, lazy loading | `registry.get("codet5")` returns model |
| 2.1.2 | Implement model caching | Models cached in memory after first load | Second call instant |
| 2.1.3 | Add model config file | `config/models.json` with HF IDs | Config validated at startup |
| 2.1.4 | Implement graceful degradation | Fallback to smaller models on OOM | Service stays up on GPU OOM |

#### 2.2 CodeT5+ Extractor (Model Wrapper)
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 2.2.1 | Load `Salesforce/codet5p-220m` | Model loads in <30s | `test_codet5_loads` |
| 2.2.2 | Implement term extraction | Input: text → Output: primary_terms[], related_terms[] | `test_codet5_extracts_terms` |
| 2.2.3 | Add batch processing | Process multiple chapters efficiently | `test_codet5_batch_processing` |
| 2.2.4 | Add inference timeout | 30s max per request | `test_codet5_timeout` |

**TDD Test (Write First):**
```python
# tests/unit/test_codet5_extractor.py
def test_codet5_extracts_terms():
    """CodeT5+ extracts meaningful technical terms from chapter text."""
    extractor = CodeT5Extractor()
    result = extractor.extract_terms(
        "This chapter covers multi-stage document chunking with overlap for RAG pipelines"
    )
    
    assert "chunking" in result.primary_terms
    assert "RAG" in result.primary_terms
    assert "overlap" in result.related_terms
    assert len(result.primary_terms) >= 3
    assert len(result.related_terms) >= 2
```

#### 2.3 GraphCodeBERT Validator (Model Wrapper)
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 2.3.1 | Load `microsoft/graphcodebert-base` | Model loads in <20s | `test_graphcodebert_loads` |
| 2.3.2 | Implement term validation | Filters generic terms ("split", "data") | `test_graphcodebert_validates` |
| 2.3.3 | Implement domain classification | Identifies AI/LLM vs C++/systems context | `test_graphcodebert_domain` |
| 2.3.4 | Add semantic expansion | Adds related terms (RAG → semantic_search) | `test_graphcodebert_expansion` |

**TDD Test (Write First):**
```python
# tests/unit/test_graphcodebert_validator.py
def test_graphcodebert_filters_generic_terms():
    """GraphCodeBERT filters out overly generic terms."""
    validator = GraphCodeBERTValidator()
    terms = ["chunking", "RAG", "split", "data", "embedding"]
    
    result = validator.validate_terms(
        terms=terms,
        original_query="LLM document processing",
        domain="ai-ml"
    )
    
    assert "chunking" in result.valid_terms
    assert "RAG" in result.valid_terms
    assert "embedding" in result.valid_terms
    assert "split" in result.rejected_terms  # Too generic
    assert "data" in result.rejected_terms   # Too generic
```

#### 2.4 CodeBERT Ranker (Model Wrapper)
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 2.4.1 | Load `microsoft/codebert-base` | Model loads in <20s | `test_codebert_loads` |
| 2.4.2 | Implement embedding generation | 768-dim embeddings for terms | `test_codebert_embeddings` |
| 2.4.3 | Implement similarity scoring | Cosine similarity with query | `test_codebert_similarity` |
| 2.4.4 | Implement ranking | Sort by relevance score descending | `test_codebert_ranking` |

**TDD Test (Write First):**
```python
# tests/unit/test_codebert_ranker.py
def test_codebert_ranks_by_relevance():
    """CodeBERT ranks terms by semantic similarity to query."""
    ranker = CodeBERTRanker()
    terms = ["chunking", "tokenization", "networking", "HTTP"]
    
    result = ranker.rank_terms(
        terms=terms,
        query="LLM document chunking for RAG"
    )
    
    # Chunking should rank highest for this query
    assert result.ranked_terms[0].term == "chunking"
    assert result.ranked_terms[0].score > 0.8
    # Networking should rank low (irrelevant)
    assert result.ranked_terms[-1].term in ["networking", "HTTP"]
    assert result.ranked_terms[-1].score < 0.3
```

**Phase 2 Integration Test:**
```python
def test_phase2_all_models_load():
    """All three model wrappers load successfully and respond."""
    response = httpx.get("http://localhost:8083/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert data["models"]["codet5"] == "loaded"       # Extractor
    assert data["models"]["graphcodebert"] == "loaded" # Validator
    assert data["models"]["codebert"] == "loaded"      # Ranker
```

---

### Phase 3: LangGraph Orchestration (Sprint 2, Days 1-4)

#### 3.1 State Machine Definition
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 3.1.1 | Define OrchestratorState | Pydantic model with all fields | Schema validated |
| 3.1.2 | Create state graph | 4 nodes: generate, validate, rank, consensus | Graph compiles |
| 3.1.3 | Add conditional edges | Route based on validation results | State transitions correct |
| 3.1.4 | Add retry logic | Max 3 retries on failure | Retries on transient errors |

**TDD Test (Write First):**
```python
# tests/unit/test_orchestrator.py
def test_orchestrator_full_pipeline():
    """Full pipeline: generate → validate → rank → consensus."""
    orchestrator = Orchestrator()
    
    result = orchestrator.run({
        "query": "Multi-stage document chunking with overlap for RAG",
        "domain": "ai-ml",
        "options": {"min_confidence": 0.7, "max_terms": 10}
    })
    
    # Pipeline completed all stages
    assert result.stages_completed == ["generate", "validate", "rank", "consensus"]
    
    # Consensus terms have agreement
    for term in result.search_terms:
        assert term.models_agreed >= 2
        assert term.score >= 0.7
```

#### 3.2 Consensus Algorithm
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 3.2.1 | Implement voting | Terms need ≥2/3 model agreement | `test_consensus_voting` |
| 3.2.2 | Weighted scoring | Final score = weighted avg across models | `test_consensus_scoring` |
| 3.2.3 | Excluded terms tracking | Track why terms were rejected | `test_consensus_excluded` |

**Phase 3 Integration Test:**
```python
def test_phase3_extract_endpoint():
    """POST /api/v1/extract returns consensus terms."""
    response = httpx.post(
        "http://localhost:8083/api/v1/extract",
        json={
            "query": "LLM document chunking with overlap",
            "domain": "ai-ml"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["search_terms"]) > 0
    assert data["search_terms"][0]["models_agreed"] >= 2
    assert "metadata" in data
    assert data["metadata"]["processing_time_ms"] < 5000
```

---

### Phase 4: Search Integration (Sprint 2, Days 5-7)

#### 4.1 Semantic Search Client
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 4.1.1 | Create SemanticSearchClient | HTTP client with retry, timeout | `test_search_client_init` |
| 4.1.2 | Implement search method | POST to semantic-search /v1/search | `test_search_client_search` |
| 4.1.3 | Handle errors gracefully | Return empty on 5xx, raise on 4xx | `test_search_client_errors` |

#### 4.2 Result Curation (Chef de Partie)
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 4.2.1 | Implement domain filter | Remove results from wrong domain | `test_curation_domain_filter` |
| 4.2.2 | Implement relevance ranking | Re-rank by semantic similarity to query | `test_curation_ranking` |
| 4.2.3 | Implement duplicate removal | Dedupe by book+chapter | `test_curation_dedup` |

**TDD Test (Write First):**
```python
# tests/unit/test_curation.py
def test_curation_filters_wrong_domain():
    """Curation filters out results from wrong domain."""
    curator = ResultCurator()
    
    raw_results = [
        {"book": "AI Engineering", "chapter": 5, "score": 0.91, "content": "LLM chunking"},
        {"book": "C++ Concurrency", "chapter": 3, "score": 0.45, "content": "memory chunk"},
        {"book": "Building LLM Apps", "chapter": 8, "score": 0.88, "content": "RAG pipeline"},
    ]
    
    curated = curator.curate(
        results=raw_results,
        query="LLM document chunking",
        domain="ai-ml"
    )
    
    # C++ Concurrency should be filtered (wrong domain)
    assert len(curated) == 2
    assert all(r["book"] != "C++ Concurrency" for r in curated)
```

#### 4.3 Full Search Endpoint
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 4.3.1 | Implement /api/v1/search | Full pipeline: extract → search → curate | `test_search_endpoint` |
| 4.3.2 | Add request validation | Validate query, domain, options | `test_search_validation` |
| 4.3.3 | Add response schema | Pydantic response model | Schema enforced |

**Phase 4 Integration Test:**
```python
def test_phase4_full_search_pipeline():
    """POST /api/v1/search executes full pipeline with curation."""
    response = httpx.post(
        "http://localhost:8083/api/v1/search",
        json={
            "query": "Multi-stage document chunking with overlap for RAG",
            "domain": "ai-ml",
            "options": {"top_k": 10}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Got curated results
    assert len(data["results"]) > 0
    assert len(data["results"]) <= 10
    
    # Results are from correct domain
    for result in data["results"]:
        assert "C++ Concurrency" not in result["book"]  # Wrong domain filtered
        assert result["relevance_score"] >= 0.3  # Semantic threshold achievable
    
    # Metadata present
    assert "pipeline" in data["metadata"]
    assert data["metadata"]["pipeline"]["stages_completed"] == 4
```

---

### Phase 5: Integration with llm-document-enhancer (Sprint 3, Days 1-4)

#### 5.1 Enrichment Script Integration
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 5.1.1 | Add --use-orchestrator flag | CLI arg to enable orchestrator | `test_cli_flag` |
| 5.1.2 | Create OrchestratorClient | HTTP client in enrichment script | `test_orchestrator_client` |
| 5.1.3 | Replace find_related_chapters | Call orchestrator instead of TF-IDF | `test_find_related_semantic` |
| 5.1.4 | Update threshold to 0.3 | Achievable for semantic embeddings | `test_threshold_achievable` |

**TDD Test (Write First):**
```python
# llm-document-enhancer/tests/integration/test_orchestrator_integration.py
def test_enrichment_produces_cross_book_references():
    """Enrichment with orchestrator produces cross-book references."""
    result = subprocess.run([
        "python", "workflows/metadata_enrichment/scripts/enrich_metadata_per_book.py",
        "--input", "workflows/metadata_extraction/output/Architecture_Patterns_with_Python_metadata.json",
        "--taxonomy", "workflows/taxonomy_setup/output/comprehensive_taxonomy.json",
        "--output", "/tmp/test_enriched.json",
        "--use-orchestrator",
        "--orchestrator-url", "http://localhost:8083"
    ], capture_output=True)
    
    assert result.returncode == 0
    
    with open("/tmp/test_enriched.json") as f:
        enriched = json.load(f)
    
    # Check for cross-book references (the whole point!)
    cross_book_refs = []
    for chapter in enriched["chapters"]:
        for ref in chapter.get("related_chapters", []):
            if ref["book"] != "Architecture Patterns with Python":
                cross_book_refs.append(ref)
    
    # MUST have cross-book references (currently ZERO)
    assert len(cross_book_refs) > 0, "No cross-book references found!"
    
    # Scores should be achievable (0.3-0.5 range for semantic)
    for ref in cross_book_refs:
        assert ref["relevance_score"] >= 0.3
```

#### 5.2 E2E Validation
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 5.2.1 | Validate cross-book refs | At least 1 cross-book ref per book | `test_cross_book_coverage` |
| 5.2.2 | Validate no false positives | No C++/systems refs in AI books | `test_no_false_positives` |
| 5.2.3 | Validate citation quality | Chicago-style citations generated | `test_citation_format` |

**Phase 5 Integration Test:**
```python
def test_phase5_e2e_enrichment():
    """Full E2E: enrichment produces valid cross-book references."""
    # Run enrichment for multiple books
    books = [
        "Architecture Patterns with Python",
        "Building LLM Powered Applications",
        "AI Engineering Building Applications"
    ]
    
    for book in books:
        run_enrichment(book, use_orchestrator=True)
    
    # Validate cross-references
    for book in books:
        enriched = load_enriched(book)
        
        # Count cross-book references
        cross_refs = count_cross_book_refs(enriched)
        assert cross_refs > 0, f"{book} has no cross-book references"
        
        # Validate domain relevance
        for chapter in enriched["chapters"]:
            for ref in chapter["related_chapters"]:
                assert is_relevant_domain(ref, expected_domain="ai-ml")
```

---

### Phase 6: Performance & Observability (Sprint 3, Days 5-7)

#### 6.1 Performance Optimization
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 6.1.1 | Add model caching | Models stay in memory between requests | `test_model_caching` |
| 6.1.2 | Add result caching | Cache search results for 5 mins | `test_result_caching` |
| 6.1.3 | Add batch inference | Process multiple queries in one call | `test_batch_inference` |
| 6.1.4 | Add GPU optimization | Use torch.cuda for GPU inference | `test_gpu_inference` |

#### 6.2 Observability
| ID | Task | Acceptance Criteria | Test |
|----|------|---------------------|------|
| 6.2.1 | Add Prometheus metrics | `/metrics` endpoint with counters | `test_metrics_endpoint` |
| 6.2.2 | Add trace spans | Spans for each pipeline stage | Traces visible in Jaeger |
| 6.2.3 | Add performance logs | Log inference times per model | `test_performance_logs` |

**Phase 6 Integration Test:**
```python
def test_phase6_performance_requirements():
    """Service meets performance requirements."""
    import time
    
    # Warm-up
    httpx.post("http://localhost:8083/api/v1/extract", json={"query": "test", "domain": "ai-ml"})
    
    # Measure latency
    start = time.time()
    response = httpx.post(
        "http://localhost:8083/api/v1/search",
        json={
            "query": "Multi-stage document chunking with overlap",
            "domain": "ai-ml"
        }
    )
    latency_ms = (time.time() - start) * 1000
    
    assert response.status_code == 200
    assert latency_ms < 5000, f"Latency {latency_ms}ms exceeds 5s SLA"
    
    # Check metrics
    metrics = httpx.get("http://localhost:8083/metrics").text
    assert "orchestrator_requests_total" in metrics
    assert "orchestrator_inference_duration_seconds" in metrics
```

---

## Integration Test Matrix

| Test ID | Services Required | What It Validates |
|---------|-------------------|-------------------|
| IT-001 | Code-Orchestrator | Service starts, models load |
| IT-002 | Code-Orchestrator | /extract returns consensus terms |
| IT-003 | Code-Orchestrator + Semantic-Search | Full search pipeline |
| IT-004 | Code-Orchestrator + Semantic-Search | Domain filtering works |
| IT-005 | All + llm-document-enhancer | Cross-book references produced |
| IT-006 | All | No false positives (C++ filtered) |
| IT-007 | All | Performance SLA met (<5s) |

---

## Acceptance Criteria Summary

### Must Have (MVP)
- [ ] CodeT5+ Extractor, GraphCodeBERT Validator, CodeBERT Ranker model wrappers load and respond
- [ ] /api/v1/extract endpoint returns consensus terms
- [ ] /api/v1/search endpoint returns curated results
- [ ] Domain filtering removes false positives
- [ ] llm-document-enhancer produces cross-book references (currently ZERO)
- [ ] Semantic threshold 0.3 is achievable (vs impossible 0.7 TF-IDF)

### Should Have
- [ ] Result caching for performance
- [ ] Prometheus metrics
- [ ] OpenTelemetry tracing
- [ ] Batch inference support

### Nice to Have
- [ ] GPU inference optimization
- [ ] Model quantization (INT8)
- [ ] Async batch processing

---

## Timeline

| Phase | Sprint | Days | Deliverable |
|-------|--------|------|-------------|
| 1 | Sprint 1 | 1-3 | Service shell, health endpoints |
| 2 | Sprint 1 | 4-7 | All 3 models loading, model wrapper classes |
| 3 | Sprint 2 | 1-4 | LangGraph orchestration, /extract endpoint |
| 4 | Sprint 2 | 5-7 | Search integration, /search endpoint |
| 5 | Sprint 3 | 1-4 | llm-document-enhancer integration |
| 6 | Sprint 3 | 5-7 | Performance, observability, polish |

**Total: 3 Sprints (~3 weeks)**

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Cross-book references per book | 0 | ≥5 | Count in enriched JSON |
| False positive rate | Unknown | <10% | Manual review sample |
| Similarity threshold | 0.7 (impossible) | 0.3 (achievable) | Config value |
| Search latency (p95) | N/A | <5s | Prometheus metrics |
| Model load time | N/A | <60s | Startup logs |

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GPU OOM on large models | High | Medium | Fallback to smaller models, model quantization |
| Slow inference | Medium | Medium | Caching, batch processing, async |
| False positives persist | High | Low | Improve domain classification, add manual review |
| Integration complexity | Medium | High | Start simple, iterate, comprehensive tests |

---

## References

- [Code-Orchestrator-Service ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- [CodeT5+ Paper](https://arxiv.org/abs/2305.07922)
- [GraphCodeBERT Paper](https://arxiv.org/abs/2009.08366)
- [CodeBERT Paper](https://arxiv.org/abs/2002.08155)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
