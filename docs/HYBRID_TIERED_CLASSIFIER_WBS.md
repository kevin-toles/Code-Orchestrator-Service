# HTC-1.0: Hybrid Tiered Classifier - Work Breakdown Structure

**Feature ID**: HTC-1.0  
**Architecture Document**: [HYBRID_TIERED_CLASSIFIER_ARCHITECTURE.md](HYBRID_TIERED_CLASSIFIER_ARCHITECTURE.md)  
**Date**: December 24, 2025  
**Status**: Planning  
**Methodology**: TDD (RED → GREEN → REFACTOR)  
**Repository**: `Code-Orchestrator-Service`

---

## Executive Summary

This WBS implements a **4-tier classification pipeline** for normalizing extracted concepts/keywords into validated taxonomy terms. The classifier operates as part of the Kitchen Brigade architecture (Sous Chef role) with these performance targets:

| Metric | Target |
|--------|--------|
| Accuracy | 98% |
| Avg Latency | 10ms |
| Cost | ~$1-5 per 10K terms |
| Memory | < 500MB |

---

## WBS Organization

This WBS has a **1-to-1 mapping** to each Acceptance Criteria block.
Each WBS section contains ALL tasks required to fully satisfy that AC block end-to-end.

| WBS Section | AC Block | Component | Status |
|-------------|----------|-----------|--------|
| WBS-AC1 | AC-1: Alias Lookup (Tier 1) | `alias_lookup.py` | ⬜ |
| WBS-AC2 | AC-2: Trained Classifier (Tier 2) | `trained_classifier.py` | ⬜ |
| WBS-AC3 | AC-3: Heuristic Filter (Tier 3) | `heuristic_filter.py` | ⬜ |
| WBS-AC4 | AC-4: LLM Fallback (Tier 4) | `llm_fallback.py` | ⬜ |
| WBS-AC5 | AC-5: Orchestrator Pipeline | `orchestrator.py` | ⬜ |
| WBS-AC6 | AC-6: API Endpoint | `src/api/classify.py` | ⬜ |
| WBS-AC7 | AC-7: Training Pipeline | `scripts/train_classifier.py` | ⬜ |
| WBS-AC8 | AC-8: Anti-Pattern Compliance | All files | ⬜ |
| WBS-AC9 | AC-9: Testing Requirements | All tests | ⬜ |

---

## Execution Order (Dependency-Based)

```
WBS-AC1 (Alias Lookup) ─────────────────────────┐
                                                │
WBS-AC2 (Trained Classifier) ───────────────────┤
                                                │
WBS-AC3 (Heuristic Filter) ─────────────────────┼──► WBS-AC5 (Orchestrator)
                                                │            │
WBS-AC4 (LLM Fallback) ─────────────────────────┘            │
                                                             ▼
                                                    WBS-AC6 (API Endpoint)
                                                             │
WBS-AC7 (Training Pipeline) ─────────────────────────────────┤
                                                             │
                                                             ▼
                                                    WBS-AC8 (Anti-Patterns)
                                                             │
                                                             ▼
                                                    WBS-AC9 (Testing)
```

**Parallel Work Possible:**
- WBS-AC1, WBS-AC2, WBS-AC3, WBS-AC4 can all proceed in parallel
- WBS-AC5 depends on ALL of WBS-AC1 through WBS-AC4
- WBS-AC7 can proceed in parallel with WBS-AC5/AC6

---

## WBS-AC1: Alias Lookup (Tier 1)

**AC Block**: AC-1 (AC-1.1 through AC-1.6)  
**Component**: `src/classifiers/alias_lookup.py`  
**Dependencies**: None (first to implement)  
**Pattern**: Hashed Feature (Machine Learning Design Patterns, Ch. 2)

### Acceptance Criteria

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| AC-1.1 | Exact Match | Term exists in lookup | `get(term)` called | Returns `AliasLookupResult` with `confidence=1.0`, `tier_used=1` |
| AC-1.2 | Case Insensitive | Term exists with different case | `get("API Gateway")` called | Returns canonical lowercase match |
| AC-1.3 | Unknown Term | Term not in lookup | `get(unknown)` called | Returns `None` (proceed to Tier 2) |
| AC-1.4 | Alias Resolution | Alias maps to canonical | `get(alias)` called | Returns canonical term, not alias |
| AC-1.5 | Startup Loading | Service starts | AliasLookup instantiated | Loads from `alias_lookup.json` |
| AC-1.6 | O(1) Performance | Lookup called | Any term | Response time < 1ms |

### WBS Tasks

| ID | Task | TDD Phase | AC | File(s) |
|----|------|-----------|-----|---------|
| AC1.1 | RED: Test AliasLookupResult dataclass exists | RED | AC-1.1 | `tests/classifiers/test_alias_lookup.py` |
| AC1.2 | RED: Test exact match returns result | RED | AC-1.1 | `tests/classifiers/test_alias_lookup.py` |
| AC1.3 | RED: Test confidence=1.0 and tier_used=1 | RED | AC-1.1 | `tests/classifiers/test_alias_lookup.py` |
| AC1.4 | RED: Test case insensitive lookup | RED | AC-1.2 | `tests/classifiers/test_alias_lookup.py` |
| AC1.5 | RED: Test unknown term returns None | RED | AC-1.3 | `tests/classifiers/test_alias_lookup.py` |
| AC1.6 | RED: Test alias resolves to canonical | RED | AC-1.4 | `tests/classifiers/test_alias_lookup.py` |
| AC1.7 | RED: Test load from JSON file | RED | AC-1.5 | `tests/classifiers/test_alias_lookup.py` |
| AC1.8 | GREEN: Create AliasLookupResult frozen dataclass | GREEN | AC-1.1 | `src/classifiers/alias_lookup.py` |
| AC1.9 | GREEN: Create AliasLookup class with dict | GREEN | AC-1.1 | `src/classifiers/alias_lookup.py` |
| AC1.10 | GREEN: Implement get() with normalization | GREEN | AC-1.2 | `src/classifiers/alias_lookup.py` |
| AC1.11 | GREEN: Implement _load_lookup() from JSON | GREEN | AC-1.5 | `src/classifiers/alias_lookup.py` |
| AC1.12 | GREEN: Create build_alias_lookup.py script | GREEN | AC-1.5 | `scripts/build_alias_lookup.py` |
| AC1.13 | GREEN: Generate alias_lookup.json from taxonomy | GREEN | AC-1.5 | `config/alias_lookup.json` |
| AC1.14 | REFACTOR: Add type hints, docstrings | REFACTOR | - | `src/classifiers/alias_lookup.py` |
| AC1.15 | QUALITY: Run mypy --strict | GATE | - | - |
| AC1.16 | QUALITY: Run pytest --cov >= 90% | GATE | - | - |

### Exit Criteria for WBS-AC1

- [ ] `AliasLookupResult` dataclass exists with `canonical_term`, `classification`, `confidence`, `tier_used` (AC-1.1)
- [ ] Exact match returns result with `confidence=1.0`, `tier_used=1` (AC-1.1)
- [ ] Case-insensitive lookup works (AC-1.2)
- [ ] Unknown term returns `None` (AC-1.3)
- [ ] Alias resolves to canonical term (AC-1.4)
- [ ] Loads from `alias_lookup.json` at startup (AC-1.5)
- [ ] O(1) lookup performance verified (AC-1.6)
- [ ] All tests pass, coverage >= 90%

---

## WBS-AC2: Trained Classifier (Tier 2)

**AC Block**: AC-2 (AC-2.1 through AC-2.8)  
**Component**: `src/classifiers/trained_classifier.py`  
**Dependencies**: None (can run parallel with WBS-AC1)  
**Pattern**: Embeddings (Machine Learning Design Patterns, Ch. 2)

### Acceptance Criteria

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| AC-2.1 | Protocol Compliance | `ConceptClassifierProtocol` defined | isinstance check | `TrainedClassifier` passes duck typing |
| AC-2.2 | High Confidence | Prediction confidence >= 0.7 | `predict(term)` called | Returns `concept` or `keyword` classification |
| AC-2.3 | Low Confidence | Prediction confidence < 0.7 | `predict(term)` called | Returns `predicted_label="unknown"` |
| AC-2.4 | SBERT Embedding | Term provided | `predict()` called | Uses `all-MiniLM-L6-v2` embedder |
| AC-2.5 | Model Loading | Model file exists | `TrainedClassifier` instantiated | Loads `.joblib` model |
| AC-2.6 | Model Not Loaded | Model file missing | `predict()` called | Raises `ConceptClassifierError` |
| AC-2.7 | Batch Prediction | List of terms provided | `predict_batch()` called | Returns list of results |
| AC-2.8 | Fake Classifier | Testing scenario | `FakeClassifier` used | Returns pre-configured responses |

### WBS Tasks

| ID | Task | TDD Phase | AC | File(s) |
|----|------|-----------|-----|---------|
| AC2.1 | RED: Test Protocol is runtime_checkable | RED | AC-2.1 | `tests/classifiers/test_trained_classifier.py` |
| AC2.2 | RED: Test TrainedClassifier passes Protocol | RED | AC-2.1 | `tests/classifiers/test_trained_classifier.py` |
| AC2.3 | RED: Test high confidence returns classification | RED | AC-2.2 | `tests/classifiers/test_trained_classifier.py` |
| AC2.4 | RED: Test low confidence returns unknown | RED | AC-2.3 | `tests/classifiers/test_trained_classifier.py` |
| AC2.5 | RED: Test model loading from path | RED | AC-2.5 | `tests/classifiers/test_trained_classifier.py` |
| AC2.6 | RED: Test error when model not loaded | RED | AC-2.6 | `tests/classifiers/test_trained_classifier.py` |
| AC2.7 | RED: Test batch prediction | RED | AC-2.7 | `tests/classifiers/test_trained_classifier.py` |
| AC2.8 | RED: Test FakeClassifier passes Protocol | RED | AC-2.8 | `tests/classifiers/test_trained_classifier.py` |
| AC2.9 | RED: Test FakeClassifier returns configured responses | RED | AC-2.8 | `tests/classifiers/test_trained_classifier.py` |
| AC2.10 | GREEN: Create ConceptClassifierProtocol | GREEN | AC-2.1 | `src/classifiers/trained_classifier.py` |
| AC2.11 | GREEN: Create ClassificationResult dataclass | GREEN | AC-2.2 | `src/classifiers/trained_classifier.py` |
| AC2.12 | GREEN: Create TrainedClassifier class | GREEN | AC-2.2 | `src/classifiers/trained_classifier.py` |
| AC2.13 | GREEN: Implement predict() with SBERT | GREEN | AC-2.4 | `src/classifiers/trained_classifier.py` |
| AC2.14 | GREEN: Implement confidence threshold logic | GREEN | AC-2.3 | `src/classifiers/trained_classifier.py` |
| AC2.15 | GREEN: Implement _load_model() | GREEN | AC-2.5 | `src/classifiers/trained_classifier.py` |
| AC2.16 | GREEN: Implement predict_batch() | GREEN | AC-2.7 | `src/classifiers/trained_classifier.py` |
| AC2.17 | GREEN: Create FakeClassifier class | GREEN | AC-2.8 | `src/classifiers/trained_classifier.py` |
| AC2.18 | GREEN: Create ConceptClassifierError exception | GREEN | AC-2.6 | `src/classifiers/exceptions.py` |
| AC2.19 | REFACTOR: Extract constants (threshold, model name) | REFACTOR | - | `src/classifiers/trained_classifier.py` |
| AC2.20 | QUALITY: Run mypy --strict | GATE | - | - |
| AC2.21 | QUALITY: Run pytest --cov >= 90% | GATE | - | - |

### Exit Criteria for WBS-AC2

- [ ] `ConceptClassifierProtocol` exists and is runtime_checkable (AC-2.1)
- [ ] `TrainedClassifier` passes Protocol check (AC-2.1)
- [ ] High confidence (>=0.7) returns `concept` or `keyword` (AC-2.2)
- [ ] Low confidence (<0.7) returns `unknown` (AC-2.3)
- [ ] Uses `all-MiniLM-L6-v2` embedder (AC-2.4)
- [ ] Loads model from `.joblib` file (AC-2.5)
- [ ] Raises `ConceptClassifierError` if model not loaded (AC-2.6)
- [ ] `predict_batch()` works for multiple terms (AC-2.7)
- [ ] `FakeClassifier` passes Protocol and returns configured responses (AC-2.8)
- [ ] All tests pass, coverage >= 90%

---

## WBS-AC3: Heuristic Filter (Tier 3)

**AC Block**: AC-3 (AC-3.1 through AC-3.6)  
**Component**: `src/classifiers/heuristic_filter.py`  
**Dependencies**: None (can run parallel)  
**Config**: `config/noise_terms.yaml` (existing)

### Acceptance Criteria

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| AC-3.1 | Watermark Rejection | Term is OCR watermark | `check(term)` called | Returns `rejection_reason="noise_watermarks"` |
| AC-3.2 | URL Fragment Rejection | Term is broken URL | `check(term)` called | Returns `rejection_reason="noise_url_fragments"` |
| AC-3.3 | Filler Word Rejection | Term is generic filler | `check(term)` called | Returns `rejection_reason="noise_generic_filler"` |
| AC-3.4 | Code Artifact Rejection | Term is Python keyword | `check(term)` called | Returns `rejection_reason="noise_code_artifacts"` |
| AC-3.5 | Valid Term Passes | Term not in noise config | `check(term)` called | Returns `None` (proceed to Tier 4) |
| AC-3.6 | Config Loading | Service starts | `HeuristicFilter` instantiated | Loads from `noise_terms.yaml` |

### WBS Tasks

| ID | Task | TDD Phase | AC | File(s) |
|----|------|-----------|-----|---------|
| AC3.1 | RED: Test watermark detection | RED | AC-3.1 | `tests/classifiers/test_heuristic_filter.py` |
| AC3.2 | RED: Test URL fragment detection | RED | AC-3.2 | `tests/classifiers/test_heuristic_filter.py` |
| AC3.3 | RED: Test filler word detection | RED | AC-3.3 | `tests/classifiers/test_heuristic_filter.py` |
| AC3.4 | RED: Test code artifact detection | RED | AC-3.4 | `tests/classifiers/test_heuristic_filter.py` |
| AC3.5 | RED: Test valid term returns None | RED | AC-3.5 | `tests/classifiers/test_heuristic_filter.py` |
| AC3.6 | RED: Test config loading from YAML | RED | AC-3.6 | `tests/classifiers/test_heuristic_filter.py` |
| AC3.7 | RED: Test regex pattern matching | RED | AC-3.1 | `tests/classifiers/test_heuristic_filter.py` |
| AC3.8 | GREEN: Create HeuristicFilter class | GREEN | AC-3.1 | `src/classifiers/heuristic_filter.py` |
| AC3.9 | GREEN: Implement _load_config() from YAML | GREEN | AC-3.6 | `src/classifiers/heuristic_filter.py` |
| AC3.10 | GREEN: Implement check() with category lookup | GREEN | AC-3.1-3.4 | `src/classifiers/heuristic_filter.py` |
| AC3.11 | GREEN: Implement _compile_patterns() for regex | GREEN | AC-3.1 | `src/classifiers/heuristic_filter.py` |
| AC3.12 | GREEN: Return None for valid terms | GREEN | AC-3.5 | `src/classifiers/heuristic_filter.py` |
| AC3.13 | REFACTOR: Extract category constants | REFACTOR | - | `src/classifiers/heuristic_filter.py` |
| AC3.14 | QUALITY: Run mypy --strict | GATE | - | - |
| AC3.15 | QUALITY: Run pytest --cov >= 90% | GATE | - | - |

### Exit Criteria for WBS-AC3

- [ ] Watermarks rejected with `rejection_reason="noise_watermarks"` (AC-3.1)
- [ ] URL fragments rejected with `rejection_reason="noise_url_fragments"` (AC-3.2)
- [ ] Filler words rejected with `rejection_reason="noise_generic_filler"` (AC-3.3)
- [ ] Code artifacts rejected with `rejection_reason="noise_code_artifacts"` (AC-3.4)
- [ ] Valid terms return `None` (AC-3.5)
- [ ] Config loads from `noise_terms.yaml` (AC-3.6)
- [ ] All 8 noise categories covered
- [ ] All tests pass, coverage >= 90%

---

## WBS-AC4: LLM Fallback (Tier 4)

**AC Block**: AC-4 (AC-4.1 through AC-4.6)  
**Component**: `src/classifiers/llm_fallback.py`  
**Dependencies**: None (can run parallel)  
**Pattern**: Tool Proxy (llm-gateway ARCHITECTURE.md)

### Acceptance Criteria

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| AC-4.1 | Service Call | Unknown term needs classification | `classify(term)` called | POST to `ai-agents:8082/v1/agents/validate-concept` |
| AC-4.2 | Response Parsing | ai-agents returns JSON | Response received | Parses `classification`, `confidence`, `canonical_term` |
| AC-4.3 | Cache High Confidence | Confidence >= 0.9 | Classification succeeds | Result cached in Tier 1 for future lookups |
| AC-4.4 | Async Context Manager | Client instance | Used in async context | `httpx.AsyncClient` properly managed |
| AC-4.5 | Timeout Handling | ai-agents slow/down | Request times out | Raises `LLMFallbackError` with details |
| AC-4.6 | Fake Fallback | Testing scenario | `FakeLLMFallback` used | Returns pre-configured responses |

### WBS Tasks

| ID | Task | TDD Phase | AC | File(s) |
|----|------|-----------|-----|---------|
| AC4.1 | RED: Test POST to ai-agents endpoint | RED | AC-4.1 | `tests/classifiers/test_llm_fallback.py` |
| AC4.2 | RED: Test response parsing | RED | AC-4.2 | `tests/classifiers/test_llm_fallback.py` |
| AC4.3 | RED: Test cache on high confidence | RED | AC-4.3 | `tests/classifiers/test_llm_fallback.py` |
| AC4.4 | RED: Test no cache on low confidence | RED | AC-4.3 | `tests/classifiers/test_llm_fallback.py` |
| AC4.5 | RED: Test timeout raises error | RED | AC-4.5 | `tests/classifiers/test_llm_fallback.py` |
| AC4.6 | RED: Test FakeLLMFallback returns responses | RED | AC-4.6 | `tests/classifiers/test_llm_fallback.py` |
| AC4.7 | GREEN: Create LLMFallback class | GREEN | AC-4.1 | `src/classifiers/llm_fallback.py` |
| AC4.8 | GREEN: Implement classify() with httpx | GREEN | AC-4.1 | `src/classifiers/llm_fallback.py` |
| AC4.9 | GREEN: Implement response parsing | GREEN | AC-4.2 | `src/classifiers/llm_fallback.py` |
| AC4.10 | GREEN: Implement cache logic | GREEN | AC-4.3 | `src/classifiers/llm_fallback.py` |
| AC4.11 | GREEN: Add timeout handling | GREEN | AC-4.5 | `src/classifiers/llm_fallback.py` |
| AC4.12 | GREEN: Create FakeLLMFallback class | GREEN | AC-4.6 | `src/classifiers/llm_fallback.py` |
| AC4.13 | GREEN: Create LLMFallbackError exception | GREEN | AC-4.5 | `src/classifiers/exceptions.py` |
| AC4.14 | REFACTOR: Extract URL constants | REFACTOR | - | `src/classifiers/llm_fallback.py` |
| AC4.15 | QUALITY: Run mypy --strict | GATE | - | - |
| AC4.16 | QUALITY: Run pytest --cov >= 90% | GATE | - | - |

### Exit Criteria for WBS-AC4

- [ ] POST to `ai-agents:8082/v1/agents/validate-concept` works (AC-4.1)
- [ ] Response JSON parsed correctly (AC-4.2)
- [ ] High confidence (>=0.9) results cached (AC-4.3)
- [ ] Low confidence results not cached (AC-4.3)
- [ ] `httpx.AsyncClient` properly managed (AC-4.4)
- [ ] Timeout raises `LLMFallbackError` (AC-4.5)
- [ ] `FakeLLMFallback` returns pre-configured responses (AC-4.6)
- [ ] All tests pass, coverage >= 90%

---

## WBS-AC5: Orchestrator Pipeline

**AC Block**: AC-5 (AC-5.1 through AC-5.7)  
**Component**: `src/classifiers/orchestrator.py`  
**Dependencies**: WBS-AC1, WBS-AC2, WBS-AC3, WBS-AC4

### Acceptance Criteria

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| AC-5.1 | Tier 1 Short-Circuit | Term in alias lookup | `classify(term)` called | Returns immediately with `tier_used=1` |
| AC-5.2 | Tier 2 Acceptance | Tier 1 miss, Tier 2 confident | `classify(term)` called | Returns with `tier_used=2` |
| AC-5.3 | Tier 3 Rejection | Tier 1+2 miss, noise term | `classify(term)` called | Returns rejected with `tier_used=3` |
| AC-5.4 | Tier 4 Fallback | Tiers 1-3 all miss | `classify(term)` called | Calls LLM fallback with `tier_used=4` |
| AC-5.5 | Cascade Logic | Unknown term | Classification proceeds | Tiers checked in order 1→2→3→4 |
| AC-5.6 | Dependency Injection | Components injected | `HybridTieredClassifier` instantiated | All 4 tier components configurable |
| AC-5.7 | Batch Classification | List of terms | `classify_batch()` called | Processes all terms, returns list |

### WBS Tasks

| ID | Task | TDD Phase | AC | File(s) |
|----|------|-----------|-----|---------|
| AC5.1 | RED: Test Tier 1 short-circuit | RED | AC-5.1 | `tests/classifiers/test_orchestrator.py` |
| AC5.2 | RED: Test Tier 2 acceptance | RED | AC-5.2 | `tests/classifiers/test_orchestrator.py` |
| AC5.3 | RED: Test Tier 3 rejection | RED | AC-5.3 | `tests/classifiers/test_orchestrator.py` |
| AC5.4 | RED: Test Tier 4 fallback | RED | AC-5.4 | `tests/classifiers/test_orchestrator.py` |
| AC5.5 | RED: Test full cascade 1→2→3→4 | RED | AC-5.5 | `tests/classifiers/test_orchestrator.py` |
| AC5.6 | RED: Test dependency injection | RED | AC-5.6 | `tests/classifiers/test_orchestrator.py` |
| AC5.7 | RED: Test batch classification | RED | AC-5.7 | `tests/classifiers/test_orchestrator.py` |
| AC5.8 | GREEN: Create HybridTieredClassifier class | GREEN | AC-5.6 | `src/classifiers/orchestrator.py` |
| AC5.9 | GREEN: Implement __init__ with DI | GREEN | AC-5.6 | `src/classifiers/orchestrator.py` |
| AC5.10 | GREEN: Implement classify() pipeline | GREEN | AC-5.5 | `src/classifiers/orchestrator.py` |
| AC5.11 | GREEN: Implement _check_tier1() | GREEN | AC-5.1 | `src/classifiers/orchestrator.py` |
| AC5.12 | GREEN: Implement _check_tier2() | GREEN | AC-5.2 | `src/classifiers/orchestrator.py` |
| AC5.13 | GREEN: Implement _check_tier3() | GREEN | AC-5.3 | `src/classifiers/orchestrator.py` |
| AC5.14 | GREEN: Implement _check_tier4() | GREEN | AC-5.4 | `src/classifiers/orchestrator.py` |
| AC5.15 | GREEN: Implement classify_batch() | GREEN | AC-5.7 | `src/classifiers/orchestrator.py` |
| AC5.16 | REFACTOR: Add logging for tier transitions | REFACTOR | - | `src/classifiers/orchestrator.py` |
| AC5.17 | QUALITY: Run mypy --strict | GATE | - | - |
| AC5.18 | QUALITY: Run pytest --cov >= 90% | GATE | - | - |

### Exit Criteria for WBS-AC5

- [ ] Known term short-circuits at Tier 1 (AC-5.1)
- [ ] Confident prediction stops at Tier 2 (AC-5.2)
- [ ] Noise term rejected at Tier 3 (AC-5.3)
- [ ] Unknown term falls through to Tier 4 (AC-5.4)
- [ ] Cascade logic verified end-to-end (AC-5.5)
- [ ] All components injectable via constructor (AC-5.6)
- [ ] Batch classification works (AC-5.7)
- [ ] All tests pass, coverage >= 90%

---

## WBS-AC6: API Endpoint

**AC Block**: AC-6 (AC-6.1 through AC-6.6)  
**Component**: `src/api/classify.py`  
**Dependencies**: WBS-AC5 (Orchestrator must exist)

### Acceptance Criteria

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| AC-6.1 | Endpoint Registration | Service started | GET /docs accessed | POST `/api/v1/classify` in OpenAPI |
| AC-6.2 | Valid Request | POST with `{"term": "microservice"}` | Request processed | Returns `ClassifyResponse` with classification |
| AC-6.3 | Empty Term Error | POST with `{"term": ""}` | Request processed | 422 Validation Error |
| AC-6.4 | Optional Domain | POST with `{"term": "...", "domain": "devops"}` | Request processed | Domain passed to classifier |
| AC-6.5 | Batch Endpoint | POST `/api/v1/classify/batch` | Multiple terms | Returns list of results |
| AC-6.6 | Dependency Injection | Endpoint called | Request processed | `HybridTieredClassifier` injected via `Depends()` |

### WBS Tasks

| ID | Task | TDD Phase | AC | File(s) |
|----|------|-----------|-----|---------|
| AC6.1 | RED: Test endpoint registration in OpenAPI | RED | AC-6.1 | `tests/api/test_classify.py` |
| AC6.2 | RED: Test valid request returns classification | RED | AC-6.2 | `tests/api/test_classify.py` |
| AC6.3 | RED: Test empty term returns 422 | RED | AC-6.3 | `tests/api/test_classify.py` |
| AC6.4 | RED: Test domain parameter passed | RED | AC-6.4 | `tests/api/test_classify.py` |
| AC6.5 | RED: Test batch endpoint | RED | AC-6.5 | `tests/api/test_classify.py` |
| AC6.6 | RED: Test classifier injected | RED | AC-6.6 | `tests/api/test_classify.py` |
| AC6.7 | GREEN: Create ClassifyRequest model | GREEN | AC-6.2 | `src/api/classify.py` |
| AC6.8 | GREEN: Create ClassifyResponse model | GREEN | AC-6.2 | `src/api/classify.py` |
| AC6.9 | GREEN: Create classify_router | GREEN | AC-6.1 | `src/api/classify.py` |
| AC6.10 | GREEN: Implement POST /classify endpoint | GREEN | AC-6.2 | `src/api/classify.py` |
| AC6.11 | GREEN: Add term validation (min_length=1) | GREEN | AC-6.3 | `src/api/classify.py` |
| AC6.12 | GREEN: Implement POST /classify/batch | GREEN | AC-6.5 | `src/api/classify.py` |
| AC6.13 | GREEN: Create get_classifier() dependency | GREEN | AC-6.6 | `src/api/classify.py` |
| AC6.14 | GREEN: Register router in main.py | GREEN | AC-6.1 | `src/main.py` |
| AC6.15 | QUALITY: Run mypy --strict | GATE | - | - |
| AC6.16 | QUALITY: Run pytest --cov >= 90% | GATE | - | - |

### Exit Criteria for WBS-AC6

- [ ] POST `/api/v1/classify` in OpenAPI docs (AC-6.1)
- [ ] Valid request returns `ClassifyResponse` (AC-6.2)
- [ ] Empty term returns 422 (AC-6.3)
- [ ] Domain parameter passed to classifier (AC-6.4)
- [ ] Batch endpoint works (AC-6.5)
- [ ] Classifier injected via `Depends()` (AC-6.6)
- [ ] All tests pass, coverage >= 90%

---

## WBS-AC7: Training Pipeline

**AC Block**: AC-7 (AC-7.1 through AC-7.5)  
**Component**: `scripts/train_classifier.py`, `scripts/build_alias_lookup.py`  
**Dependencies**: WBS-AC1 (alias lookup format), WBS-AC2 (classifier format)

### Acceptance Criteria

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| AC-7.1 | Data Loading | `FINAL_AGGREGATED_RESULTS.json` exists | `prepare_training_data()` called | Returns concepts (label=0) and keywords (label=1) |
| AC-7.2 | Model Training | Training data prepared | `train_classifier()` called | Trains LogisticRegression on SBERT embeddings |
| AC-7.3 | Model Evaluation | Model trained | Evaluation runs | Reports accuracy, precision, recall, F1 |
| AC-7.4 | Model Export | Training complete | Model saved | `.joblib` file created |
| AC-7.5 | Alias Lookup Generation | Taxonomy loaded | `build_alias_lookup.py` run | `alias_lookup.json` generated |

### WBS Tasks

| ID | Task | TDD Phase | AC | File(s) |
|----|------|-----------|-----|---------|
| AC7.1 | RED: Test prepare_training_data loads taxonomy | RED | AC-7.1 | `tests/scripts/test_train_classifier.py` |
| AC7.2 | RED: Test labels assigned correctly | RED | AC-7.1 | `tests/scripts/test_train_classifier.py` |
| AC7.3 | RED: Test train_classifier returns model | RED | AC-7.2 | `tests/scripts/test_train_classifier.py` |
| AC7.4 | RED: Test model saved to path | RED | AC-7.4 | `tests/scripts/test_train_classifier.py` |
| AC7.5 | RED: Test alias lookup generation | RED | AC-7.5 | `tests/scripts/test_build_alias_lookup.py` |
| AC7.6 | GREEN: Implement prepare_training_data() | GREEN | AC-7.1 | `scripts/train_classifier.py` |
| AC7.7 | GREEN: Implement train_classifier() | GREEN | AC-7.2 | `scripts/train_classifier.py` |
| AC7.8 | GREEN: Implement evaluate_model() | GREEN | AC-7.3 | `scripts/train_classifier.py` |
| AC7.9 | GREEN: Save model with joblib.dump() | GREEN | AC-7.4 | `scripts/train_classifier.py` |
| AC7.10 | GREEN: Implement build_alias_lookup.py | GREEN | AC-7.5 | `scripts/build_alias_lookup.py` |
| AC7.11 | GREEN: Generate alias_lookup.json | GREEN | AC-7.5 | `config/alias_lookup.json` |
| AC7.12 | GREEN: Train and save production model | GREEN | AC-7.4 | `models/concept_classifier.joblib` |
| AC7.13 | QUALITY: Verify accuracy >= 98% | GATE | AC-7.3 | - |

### Exit Criteria for WBS-AC7

- [ ] Training data loads from `FINAL_AGGREGATED_RESULTS.json` (AC-7.1)
- [ ] Model trained on SBERT embeddings (AC-7.2)
- [ ] Evaluation reports accuracy >= 98% (AC-7.3)
- [ ] Model exported to `models/concept_classifier.joblib` (AC-7.4)
- [ ] `config/alias_lookup.json` generated (AC-7.5)
- [ ] All tests pass

---

## WBS-AC8: Anti-Pattern Compliance

**AC Block**: AC-8 (AC-8.1 through AC-8.5)  
**Repository**: All new files  
**Dependencies**: All WBS-AC1 through WBS-AC7  
**Reference**: `textbooks/Guidelines/CODING_PATTERNS_ANALYSIS.md`

### Acceptance Criteria

| ID | Rule | Requirement |
|----|------|-------------|
| AC-8.1 | S1192 | No duplicated string literals (extract to constants) |
| AC-8.2 | S3776 | All functions cognitive complexity < 15 |
| AC-8.3 | Anti-Pattern #7/#13 | Exception classes end in "Error", no shadowing |
| AC-8.4 | Anti-Pattern #12 | Protocol-based fakes for testing (no real connections in tests) |
| AC-8.5 | Type Annotations | mypy --strict passes with 0 errors |

### WBS Tasks

| ID | Task | AC | Tool |
|----|------|-----|------|
| AC8.1 | Audit for duplicated string literals | AC-8.1 | SonarLint |
| AC8.2 | Extract duplicates to constants | AC-8.1 | Manual |
| AC8.3 | Audit function complexity | AC-8.2 | SonarLint |
| AC8.4 | Refactor functions with CC >= 15 | AC-8.2 | Manual |
| AC8.5 | Verify exception names end in "Error" | AC-8.3 | Manual |
| AC8.6 | Verify no shadowing of builtins | AC-8.3 | SonarLint |
| AC8.7 | Verify FakeClassifier/FakeLLMFallback used | AC-8.4 | Code review |
| AC8.8 | Run mypy --strict on all new files | AC-8.5 | mypy |
| AC8.9 | Fix all mypy errors | AC-8.5 | Manual |

### Exit Criteria for WBS-AC8

- [ ] SonarLint: 0 S1192 issues (AC-8.1)
- [ ] SonarLint: 0 S3776 issues, all CC < 15 (AC-8.2)
- [ ] All exceptions end in "Error" (AC-8.3)
- [ ] Fakes used in all unit tests (AC-8.4)
- [ ] mypy --strict: 0 errors (AC-8.5)

---

## WBS-AC9: Testing Requirements

**AC Block**: AC-9 (AC-9.1 through AC-9.3)  
**Repository**: All test files  
**Dependencies**: All WBS-AC1 through WBS-AC8

### Acceptance Criteria

| ID | Requirement | Metric |
|----|-------------|--------|
| AC-9.1 | Unit Test Coverage | >= 90% on all classifier modules |
| AC-9.2 | TDD Compliance | Test written FIRST for each AC |
| AC-9.3 | Integration Tests | Full pipeline tested with real components |

### WBS Tasks

| ID | Task | AC | Command/Tool |
|----|------|-----|--------------|
| AC9.1 | Run coverage on classifiers/ | AC-9.1 | `pytest --cov=src/classifiers` |
| AC9.2 | Run coverage on api/classify | AC-9.1 | `pytest --cov=src/api` |
| AC9.3 | Verify >= 90% coverage | AC-9.1 | Coverage report |
| AC9.4 | Audit git history for TDD compliance | AC-9.2 | `git log` - tests before impl |
| AC9.5 | Write integration test: full pipeline | AC-9.3 | `tests/integration/test_classifier_pipeline.py` |
| AC9.6 | Write integration test: API end-to-end | AC-9.3 | `tests/integration/test_classify_api.py` |
| AC9.7 | Run all integration tests | AC-9.3 | `pytest tests/integration/` |

### Exit Criteria for WBS-AC9

- [ ] Classifier modules coverage >= 90% (AC-9.1)
- [ ] API modules coverage >= 90% (AC-9.1)
- [ ] Git history shows RED before GREEN (AC-9.2)
- [ ] All integration tests pass (AC-9.3)

---

## Summary: Status Tracking

| WBS Section | AC Block | Status | Tests | Coverage |
|-------------|----------|--------|-------|----------|
| WBS-AC1 | AC-1: Alias Lookup | ⬜ NOT STARTED | 0/0 | - |
| WBS-AC2 | AC-2: Trained Classifier | ⬜ NOT STARTED | 0/0 | - |
| WBS-AC3 | AC-3: Heuristic Filter | ⬜ NOT STARTED | 0/0 | - |
| WBS-AC4 | AC-4: LLM Fallback | ⬜ NOT STARTED | 0/0 | - |
| WBS-AC5 | AC-5: Orchestrator | ⬜ NOT STARTED | 0/0 | - |
| WBS-AC6 | AC-6: API Endpoint | ⬜ NOT STARTED | 0/0 | - |
| WBS-AC7 | AC-7: Training Pipeline | ⬜ NOT STARTED | 0/0 | - |
| WBS-AC8 | AC-8: Anti-Patterns | ⬜ NOT STARTED | N/A | N/A |
| WBS-AC9 | AC-9: Testing | ⬜ NOT STARTED | 0/0 | - |

**Target Totals:**
- **~120 unit tests** across all modules
- **~10 integration tests** for end-to-end validation
- **98%+ accuracy** on concept/keyword classification
- **10ms avg latency** (weighted across tiers)

---

## Files to Create (TDD Order)

### Phase 1: RED (Tests First)
```
tests/classifiers/test_alias_lookup.py
tests/classifiers/test_trained_classifier.py
tests/classifiers/test_heuristic_filter.py
tests/classifiers/test_llm_fallback.py
tests/classifiers/test_orchestrator.py
tests/api/test_classify.py
tests/scripts/test_train_classifier.py
tests/scripts/test_build_alias_lookup.py
```

### Phase 2: GREEN (Implementation)
```
src/classifiers/__init__.py
src/classifiers/exceptions.py
src/classifiers/alias_lookup.py
src/classifiers/trained_classifier.py
src/classifiers/heuristic_filter.py
src/classifiers/llm_fallback.py
src/classifiers/orchestrator.py
src/api/classify.py
scripts/train_classifier.py
scripts/build_alias_lookup.py
```

### Phase 3: REFACTOR (Configuration & Artifacts)
```
config/alias_lookup.json (generated)
config/classifier_config.yaml
models/concept_classifier.joblib (trained)
```

---

*WBS organized by Acceptance Criteria blocks with 1-to-1 mapping. Each AC block is independently verifiable end-to-end.*
