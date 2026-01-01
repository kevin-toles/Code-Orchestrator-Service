# Technical Change Log - Code-Orchestrator-Service

This document tracks all implementation changes, their rationale, and git commit correlations.

---

## Change Log Format

| Field | Description |
|-------|-------------|
| **Date/Time** | When the change was made |
| **WBS Item** | Related WBS task number |
| **Change Type** | Feature, Fix, Refactor, Documentation |
| **Summary** | Brief description of the change |
| **Files Changed** | List of affected files |
| **Rationale** | Why the change was made |
| **Git Commit** | Commit hash (if committed) |

---

## 2026-01-01

### CL-025: Graceful Model Degradation & Think Tag Stripping

| Field | Value |
|-------|-------|
| **Date/Time** | 2026-01-01 |
| **WBS Item** | WBS-KB10 (Summarization Pipeline Support) |
| **Change Type** | Feature |
| **Summary** | Implemented graceful model degradation in InferenceClient - system now uses whatever LLM is loaded rather than failing. Added think tag stripping for DeepSeek-R1 output. Increased max_tokens for thinking models. |
| **Files Changed** | `src/clients/inference_client.py` |
| **Rationale** | Hardcoded model preference caused failures when requested model wasn't loaded. Thinking models (DeepSeek-R1) include `<think>...</think>` reasoning blocks that shouldn't appear in final output. 500 max_tokens was insufficient for thinking models that spend 80% of tokens on reasoning. |
| **Git Commit** | Pending |

**Changes Implemented**:

| Component | Change | Purpose |
|-----------|--------|---------|
| `DEFAULT_MODEL` | Changed from `"qwen2.5-7b"` to `None` | No hardcoded model preference |
| `get_loaded_models()` | New method | Query inference-service for loaded models |
| `_resolve_model()` | New method | Graceful degradation - use any loaded model |
| `_strip_think_tags()` | New method | Remove `<think>...</think>` blocks from output |
| `max_tokens` | Increased from 500 to 1500 | Thinking models need more output tokens |

**Graceful Degradation Logic**:
```python
def _resolve_model(self, requested_model: str | None) -> str:
    """Use requested model if loaded, else use any loaded model."""
    loaded = self.get_loaded_models()
    if not loaded:
        raise RuntimeError("No models loaded in inference-service")
    if requested_model and requested_model in loaded:
        return requested_model
    # Fallback to first loaded model
    logger.warning(f"Model '{requested_model}' not loaded, using '{loaded[0]}'")
    return loaded[0]
```

**Think Tag Stripping** (regex-based):
```python
def _strip_think_tags(self, text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> reasoning blocks."""
    # Complete tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Incomplete opening tags (streaming edge case)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()
```

**Architecture Alignment**:
- ✅ Kitchen Brigade: Code-Orchestrator (Line Cook) adapts to available resources
- ✅ Follows WBS-KB10: Supports summarization pipeline with any loaded model
- ✅ Anti-Pattern #12: Graceful degradation prevents hard failures

---

## 2025-12-27

### CL-024: WBS-AC4 Complete - LLM Fallback (Tier 4) ✅

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-27 |
| **WBS Item** | WBS-AC4 (LLM Fallback Tier 4) |
| **Change Type** | Feature |
| **Summary** | Completed Tier 4 LLM Fallback implementation. 44 tests, 96% coverage, mypy --strict passing. Implements Tool Proxy pattern calling ai-agents:8082/v1/agents/validate-concept. |
| **Files Changed** | `src/classifiers/llm_fallback.py`, `tests/classifiers/test_llm_fallback.py`, `src/classifiers/__init__.py` |
| **Rationale** | Final tier of 4-tier classification pipeline for unknown terms requiring LLM validation |
| **Git Commit** | Pending |

**Components Implemented**:
- `LLMFallbackResult` - Frozen dataclass with classification, confidence, canonical_term, tier_used
- `LLMFallbackProtocol` - Runtime-checkable Protocol for DI
- `AliasCacheProtocol` - Protocol for high-confidence result caching
- `LLMFallback` - Main implementation using httpx.AsyncClient
- `FakeLLMFallback` - Test double for protocol-based testing

**Test Summary**:
| Test Class | Count | Coverage |
|------------|-------|----------|
| TestLLMFallbackResult | 6 | Dataclass validation |
| TestLLMFallbackProtocol | 3 | Protocol conformance |
| TestServiceCall | 4 | HTTP POST to ai-agents |
| TestResponseParsing | 6 | JSON response handling |
| TestCacheHighConfidence | 3 | Results >= 0.9 cached |
| TestCacheLowConfidence | 3 | Results < 0.9 not cached |
| TestAsyncContextManager | 2 | httpx.AsyncClient lifecycle |
| TestTimeoutHandling | 4 | Timeout/error handling |
| TestFakeLLMFallback | 6 | Test double behavior |
| TestBatchClassification | 2 | Batch processing |
| TestConstants | 5 | Module constants |
| **TOTAL** | **44** | **96%** |

**Quality Gates**:
- ✅ mypy --strict: Passed
- ✅ pytest: 44/44 passed
- ✅ Coverage: 96% (target: 90%)

**All 4 Tiers Now Complete**:
| Tier | WBS | Tests | Coverage |
|------|-----|-------|----------|
| 1 | WBS-AC1 Alias Lookup | 26 | 94% |
| 2 | WBS-AC2 Trained Classifier | 40 | 95% |
| 3 | WBS-AC3 Heuristic Filter | 70 | 97% |
| 4 | WBS-AC4 LLM Fallback | 44 | 96% |
| **Total** | - | **180** | **~95%** |

**Next**: WBS-AC5 (Orchestrator Pipeline) - orchestrates all 4 tiers

---

## 2025-12-24

### CL-021: HTC-1.0 WBS Creation - Hybrid Tiered Classifier Planning

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-24 |
| **WBS Item** | HTC-1.0 (Planning Phase) |
| **Change Type** | Documentation |
| **Summary** | Created comprehensive WBS for Hybrid Tiered Classifier feature. 9 AC blocks mapped to WBS sections. Architecture document and WBS both created. |
| **Files Changed** | `docs/HYBRID_TIERED_CLASSIFIER_ARCHITECTURE.md`, `docs/HYBRID_TIERED_CLASSIFIER_WBS.md` |
| **Rationale** | Define 4-tier classification pipeline for normalizing extracted concepts/keywords into validated taxonomy terms |
| **Git Commit** | Pending |

**Architecture Summary**:
- **Tier 1**: Alias Lookup (O(1) hash, 0ms, confidence=1.0)
- **Tier 2**: Trained Classifier (SBERT + LogisticRegression, 5ms avg, threshold 0.7)
- **Tier 3**: Heuristic Filter (noise_terms.yaml, 8 categories, 1ms)
- **Tier 4**: LLM Fallback (ai-agents call, <1% of terms, caches results)

**Performance Targets**:
| Metric | Target |
|--------|--------|
| Accuracy | 98% |
| Avg Latency | 10ms |
| Cost | ~$1-5 per 10K terms |
| Memory | < 500MB |

**WBS Structure** (9 AC Blocks):
| WBS Section | AC Block | Component |
|-------------|----------|-----------|
| WBS-AC1 | AC-1: Alias Lookup | `alias_lookup.py` |
| WBS-AC2 | AC-2: Trained Classifier | `trained_classifier.py` |
| WBS-AC3 | AC-3: Heuristic Filter | `heuristic_filter.py` |
| WBS-AC4 | AC-4: LLM Fallback | `llm_fallback.py` |
| WBS-AC5 | AC-5: Orchestrator Pipeline | `orchestrator.py` |
| WBS-AC6 | AC-6: API Endpoint | `src/api/classify.py` |
| WBS-AC7 | AC-7: Training Pipeline | `scripts/train_classifier.py` |
| WBS-AC8 | AC-8: Anti-Pattern Compliance | All files |
| WBS-AC9 | AC-9: Testing Requirements | All tests |

**Training Data**: validated_term_filter.json (3,255 concepts + 6,981 keywords = 10,236 validated examples)

**Methodology**: TDD (RED → GREEN → REFACTOR)

---

### CL-020: Concept Discovery Module (BERTopic)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-23 |
| **WBS Item** | Data-Driven Concept Discovery |
| **Change Type** | Feature |
| **Summary** | Added BERTopic-based concept discovery module for automatic concept extraction from metadata. Replaces hardcoded seed concepts with data-driven discovery. |
| **Files Changed** | `src/nlp/concept_discovery.py`, `scripts/generate_concept_labels.py` |
| **Rationale** | Enable automatic discovery of abstract concepts from extracted terms using topic modeling |
| **Git Commit** | Pending |

**Key Features**:
- BERTopic clustering with SBERT embeddings
- Quality scoring for concept vs keyword classification
- Noise pattern filtering
- JSON export for runtime use

---

## 2025-12-20

### CL-019: CME-1.0 Complete - Configurable Metadata Extraction Feature ✅

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-20 |
| **WBS Item** | CME-1.0 (WBS-AC2) |
| **Change Type** | Feature |
| **Summary** | CME-1.0 Phase 1 COMPLETE. POST /api/v1/metadata/extract endpoint with TF-IDF keyword extraction, concept extraction, noise filtering (8 categories), domain detection, and quality scoring. Full TDD implementation with 96 tests passing. |
| **Files Changed** | See CL-018 below |
| **Rationale** | Enable llm-document-enhancer to call orchestrator for high-quality metadata extraction instead of local StatisticalExtractor |
| **Git Commit** | Pending |

**Architecture Reference**: [CME_ARCHITECTURE.md](../../textbooks/pending/platform/CME_ARCHITECTURE.md)

**Integration Status**:
- ✅ Endpoint: POST /api/v1/metadata/extract responding on :8083
- ✅ Request/Response schemas match architecture specification
- ✅ Noise filtering: 8 categories (watermarks, contractions, URL fragments, etc.)
- ✅ Quality scoring: 0.0-1.0 based on keyword/concept confidence + noise ratio
- ✅ Domain detection: Returns detected_domain and domain_confidence

**Consumer Integration**:
- llm-document-enhancer `MetadataExtractionClient` connects successfully
- Toggle via `--use-orchestrator` CLI flag or `EXTRACTION_USE_ORCHESTRATOR_EXTRACTION=true`

---

### CL-018: CME-1.0 Phase 1 - Configurable Metadata Extraction Endpoint

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-01-XX |
| **WBS Item** | CME-1.1 through CME-1.5 |
| **Change Type** | Feature |
| **Summary** | New POST /api/v1/metadata/extract endpoint with configurable options for keyword extraction, concept extraction, noise filtering, domain detection, and quality scoring. Full TDD implementation with 96 tests. |
| **Files Changed** | See below |
| **Rationale** | Provide unified metadata extraction with caller-controllable parameters (AC-2.1 through AC-2.8) |
| **Git Commit** | Pending |

**New Files Created:**
- `src/models/metadata_models.py` - Pydantic request/response models (15 tests)
- `src/validators/noise_filter.py` - Noise filtering for 8 categories (33 tests)
- `src/validators/__init__.py` - Package exports
- `src/extractors/metadata_extractor.py` - Unified extraction pipeline (25 tests)
- `src/extractors/__init__.py` - Package exports
- `src/api/metadata.py` - FastAPI router with POST /extract endpoint (23 tests)
- `tests/unit/models/test_metadata_models.py` - Model tests
- `tests/unit/validators/test_noise_filter.py` - Noise filter tests
- `tests/unit/extractors/test_metadata_extractor.py` - Extractor tests
- `tests/unit/api/test_metadata_extract.py` - API endpoint tests

**Modified Files:**
- `src/main.py` - Register metadata_router

**Acceptance Criteria Fulfilled:**
| AC | Description | Status |
|----|-------------|--------|
| AC-2.1 | POST /api/v1/metadata/extract registered | ✅ |
| AC-2.2 | Keywords with term, score, is_technical | ✅ |
| AC-2.3 | Concepts with name, confidence, domain, tier | ✅ |
| AC-2.4 | Rejected keywords with reasons | ✅ |
| AC-2.5 | Quality score 0.0-1.0 | ✅ |
| AC-2.6 | Domain detection with confidence | ✅ |
| AC-2.7 | Empty text validation (422) | ✅ |
| AC-2.8 | Invalid options validation (422) | ✅ |

**Anti-Patterns Applied:**
- S1192: Constants at module level (no duplicate strings)
- S3776: Cognitive complexity < 15 via helper methods
- Anti-Pattern #12: Singleton pattern for extractors
- AC-6.5: Full type annotations (mypy --strict)

**Test Summary:**
- WBS-1.1: 15 tests (metadata_models)
- WBS-1.2: 33 tests (noise_filter)
- WBS-1.3: 25 tests (metadata_extractor)
- WBS-1.4: 23 tests (API endpoint)
- **Total: 96 tests passing**

**Coverage:**
- metadata_models.py: 98%
- noise_filter.py: 99%

---

## 2025-12-19

### CL-017: Gateway-First Communication Pattern Documentation

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-19 |
| **WBS Item** | Architecture Documentation |
| **Change Type** | Documentation |
| **Summary** | Added Gateway-First Communication Pattern section to ARCHITECTURE.md. External applications MUST route through Gateway:8080 to access Code-Orchestrator-Service. |
| **Files Changed** | `docs/ARCHITECTURE.md` |
| **Rationale** | Explicitly document that external apps cannot call Code-Orchestrator:8083 directly. All external access must go through Gateway. Internal platform services (ai-agents, Gateway) may call directly. |
| **Git Commit** | Pending |

**Communication Pattern:**

| Source | Target | Route | Status |
|--------|--------|-------|--------|
| External app | Code-Orchestrator | Via Gateway:8080 | ✅ REQUIRED |
| Platform service (ai-agents) | Code-Orchestrator | Direct:8083 | ✅ Allowed |
| Platform service (Gateway) | Code-Orchestrator | Direct:8083 | ✅ Allowed |

---

## 2025-12-18

### CL-016: EEP-6 Diagram Similarity (Enhanced Enrichment Pipeline Phase 6)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-18 |
| **WBS Item** | ENHANCED_ENRICHMENT_PIPELINE_WBS.md - Phase EEP-6 |
| **Change Type** | Feature |
| **Summary** | Diagram extraction and similarity computation for detecting related architecture diagrams across chapters. Detects Figure/Diagram/Architecture references and ASCII art, uses SBERT for semantic similarity. Full TDD RED→GREEN→REFACTOR cycle. |
| **Files Changed** | `src/models/diagram_extractor.py` (NEW), `tests/unit/models/test_eep6_diagram_similarity.py` (NEW) |
| **Rationale** | Flag chapters with similar architecture diagrams to improve cross-referencing quality |
| **Git Commit** | `509e681` |

**Implementation Details:**

| Phase | Status | Details |
|-------|--------|---------|
| Document Analysis | ✅ Complete | Steps 1-3: Hierarchy, Guideline, Conflict review |
| Anti-Pattern Audit | ✅ Complete | S1192, S3776, S1172, #12 all verified |
| RED Phase | ✅ Complete | 50 tests written before implementation |
| GREEN Phase | ✅ Complete | All 50 tests pass |
| REFACTOR Phase | ✅ Complete | ruff check passes, 0 SonarQube issues |

**Acceptance Criteria Met:**

| AC | Description | Status |
|----|-------------|--------|
| AC-6.1.1 | Detect "Figure X", "Diagram X", "Architecture diagram" patterns | ✅ |
| AC-6.1.2 | Detect ASCII art diagrams (box drawing characters) | ✅ |
| AC-6.1.3 | Return `DiagramReference(type, caption, context)` | ✅ |
| AC-6.2.1 | Extract caption text | ✅ |
| AC-6.2.2 | Extract surrounding context (paragraph before/after) | ✅ |
| AC-6.2.3 | Use SBERT to embed description | ✅ |
| AC-6.3.1 | Compare diagram descriptions using SBERT | ✅ |
| AC-6.3.2 | Flag chapters with similar architecture diagrams | ✅ |
| AC-6.4.1 | TDD cycle with 15+ tests | ✅ (50 tests) |

**New Files:**

| File | Purpose |
|------|---------|
| `src/models/diagram_extractor.py` | DiagramExtractor class, Protocol, dataclasses |
| `tests/unit/models/test_eep6_diagram_similarity.py` | 50 unit tests for diagram similarity |

**New Dataclasses:**

| Dataclass | Fields | Purpose |
|-----------|--------|---------|
| `DiagramType` (Enum) | FIGURE, DIAGRAM, ARCHITECTURE, ASCII_ART | Type of diagram detected |
| `DiagramReference` | diagram_type, caption, context, line_number | Reference to diagram in text |
| `DiagramExtractorConfig` | context_lines_before/after, ascii_art_threshold, sbert_model_name | Configuration |
| `DiagramSimilarityResult` | score, source_diagram_index, target_diagram_index | Comparison result |

**DiagramExtractor Methods:**

| Method | Description |
|--------|-------------|
| `extract_diagrams(text)` | Extract all diagram references from text |
| `embed_diagram(diagram)` | SBERT embed diagram description |
| `compute_similarity(d1, d2)` | Cosine similarity between diagrams |
| `compare_chapter_diagrams(src, tgt)` | Compare all diagrams between chapters |
| `get_max_diagram_similarity(src, tgt)` | Get max similarity score |

**Detection Patterns:**

| Pattern | Regex | Example Match |
|---------|-------|---------------|
| Figure | `(?i)figure\s+[\d.]+` | "Figure 3.1: Architecture" |
| Diagram | `(?i)diagram\s+[\d.]+` | "Diagram 2.5.1: Data Flow" |
| Architecture | `(?i)architecture\s+diagram` | "Architecture Diagram: Service Mesh" |
| ASCII Art | Box chars density ≥ 5% | `┌───┐`, `+----+` |

**Patterns Applied:**

- Protocol pattern (CODING_PATTERNS_ANALYSIS.md line 130)
- FakeDiagramExtractor for testing
- Dataclasses for structured output
- Embedding cached via `_embedding_cache` dict (Anti-Pattern #12 prevention)
- Lazy-loaded SBERT model (Anti-Pattern #12 prevention)
- Constants for string literals and thresholds (S1192 compliance)

---

## 2025-01-14

### CL-015: EEP-2 Concept Extraction Layer (Enhanced Enrichment Pipeline Phase 2)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-01-14 |
| **WBS Item** | ENHANCED_ENRICHMENT_PIPELINE_WBS.md - Phase EEP-2 |
| **Change Type** | Feature |
| **Summary** | Concept extraction layer that matches EEP-1 filtered keywords against domain taxonomy to extract domain concepts with confidence scores. Includes hierarchical concept relationships and domain classification. Full TDD RED→GREEN→REFACTOR cycle. |
| **Files Changed** | `src/models/concept_extractor.py` (NEW), `src/api/concepts.py` (NEW), `src/main.py` (MODIFIED), `tests/unit/models/test_eep2_concept_extraction.py` (NEW), `tests/unit/api/test_eep2_concepts_api.py` (NEW) |
| **Rationale** | Extract domain concepts from text/keywords by matching against taxonomy keywords for enrichment pipeline |
| **Git Commit** | Pending |

**Implementation Details:**

| Phase | Status | Details |
|-------|--------|---------|
| Document Analysis | ✅ Complete | Steps 1-3: Hierarchy, Guideline, Conflict review |
| Anti-Pattern Audit | ✅ Complete | S1192, S3776, S1172, S3457, #7, #12 all verified |
| RED Phase | ✅ Complete | 61 tests written (46 model, 15 API) |
| GREEN Phase | ✅ Complete | All 61 tests pass |
| REFACTOR Phase | ✅ Complete | ruff check passes, 0 SonarQube issues |

**New Files:**

| File | Purpose |
|------|---------|
| `src/models/concept_extractor.py` | ConceptExtractor class, Protocol, dataclasses |
| `src/api/concepts.py` | POST /api/v1/concepts endpoint |
| `tests/unit/models/test_eep2_concept_extraction.py` | 46 unit tests for ConceptExtractor |
| `tests/unit/api/test_eep2_concepts_api.py` | 15 API endpoint tests |

**New Dataclasses:**

| Dataclass | Fields | Purpose |
|-----------|--------|---------|
| `ExtractedConcept` | name, confidence, domain, tier, parent_concept | Single extracted concept |
| `ConceptExtractionResult` | concepts, domain_scores, primary_domain, total_matches | Extraction result container |
| `ConceptExtractorConfig` | domain_taxonomy_path, tier_taxonomy_path, enable_hierarchical, min_confidence | Configuration |

**ConceptExtractor Methods:**

| Method | Description |
|--------|-------------|
| `extract_concepts(text)` | Extract concepts from text |
| `extract_concepts_from_keywords(keywords)` | Extract from EEP-1 keywords |
| `get_domain_concepts(domain)` | Get all concepts for a domain |
| `get_tier_concepts(tier)` | Get all concepts for a tier |
| `classify_domain(text)` | Classify text into primary domain |
| `get_concept_hierarchy()` | Get hierarchical concept tree |

**API Schema (AC-2.4.2, AC-2.4.3 Compliant):**

| Field | Type | WBS Ref | Description |
|-------|------|---------|-------------|
| Request: `text` | string | AC-2.4.2 | Text to extract concepts from |
| Request: `domain` | string | AC-2.4.2 | Domain context for filtering |
| Response: `concepts` | array | AC-2.4.3 | List of extracted concepts |
| Response: `domain_score` | float | AC-2.4.3 | Primary domain confidence (0.0-1.0) |
| Response: `domain_scores` | dict | Extended | All domain scores (backward compat) |

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/concepts` | POST | Extract concepts from text/keywords |
| `/api/v1/concepts/domains` | GET | Get all domains with concepts |
| `/api/v1/concepts/domains/{domain}` | GET | Get concepts for specific domain |

**Patterns Applied:**

- Protocol pattern (CODING_PATTERNS_ANALYSIS.md line 130)
- FakeConceptExtractor for testing
- Dataclasses for structured output
- Taxonomy cached at init (Anti-Pattern #12 prevention)
- Constants for string literals (S1192 compliance)

---

## 2025-12-17

### CL-014: EEP-1 Domain-Aware Keyword Filtering (Enhanced Enrichment Pipeline Phase 1)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-17 |
| **WBS Item** | ENHANCED_ENRICHMENT_PIPELINE_WBS.md - Phase EEP-1 |
| **Change Type** | Feature |
| **Summary** | Domain-aware keyword filtering with custom technical stopwords and domain taxonomy integration. Full TDD RED→GREEN→REFACTOR cycle. |
| **Files Changed** | `src/models/tfidf_extractor.py` (EXTENDED), `config/technical_stopwords.json` (NEW), `tests/unit/models/test_eep1_stopword_filtering.py` (NEW) |
| **Rationale** | Filter technical book artifacts (chapter, figure, table) and support domain-specific keyword boosting for enrichment pipeline |
| **Git Commit** | Pending |

**Implementation Details:**

| Phase | Status | Details |
|-------|--------|---------|
| Document Analysis | ✅ Complete | Steps 1-3: Hierarchy, Guideline, Conflict review |
| Anti-Pattern Audit | ✅ Complete | S1192, S3776, S1172, S3457, #7, #12 all verified |
| RED Phase | ✅ Complete | 34 tests written before implementation |
| GREEN Phase | ✅ Complete | All 34 tests pass, 28 legacy tests pass |
| REFACTOR Phase | ✅ Complete | SonarQube issues fixed (S1700, S5795) |

**New Files:**

| File | Purpose |
|------|---------|
| `config/technical_stopwords.json` | Technical book stopwords by category |
| `tests/unit/models/test_eep1_stopword_filtering.py` | 34 unit tests for EEP-1 features |

**Config Extensions (KeywordExtractorConfig):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `custom_stopwords_path` | `Path \| None` | `None` | Path to custom stopwords JSON |
| `merge_stopwords` | `bool` | `True` | Merge with sklearn English stopwords |
| `domain_taxonomy_path` | `Path \| None` | `None` | Path to domain taxonomy JSON |
| `active_domain` | `str \| None` | `None` | Active domain for filtering rules |

**Technical Stopword Categories:**

| Category | Example Terms | Count |
|----------|---------------|-------|
| `document_structure` | chapter, section, figure, table | 26 |
| `meta_words` | isbn, copyright, edition, publisher | 21 |
| `programming_noise` | example, listing, code, output | 30 |
| `common_filler` | following, previous, shown, given | 36 |

**New Methods (TfidfKeywordExtractor):**

| Method | Description |
|--------|-------------|
| `get_effective_stopwords()` | Returns cached frozenset of all effective stopwords |
| `_load_custom_stopwords()` | Load and flatten stopwords from JSON |
| `_load_domain_taxonomy()` | Load and validate domain taxonomy |

**Backward Compatibility:**

- ✅ Default config unchanged (4 tests verify)
- ✅ `stop_words="english"` still works
- ✅ `extract_keywords()` signature unchanged
- ✅ `extract_keywords_with_scores()` signature unchanged

---

## 2025-12-18

### CL-013: BERTopic Cluster Endpoint (Phase B2.2)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-16 |
| **WBS Item** | BERTOPIC_INTEGRATION_WBS.md - Phase B2.2 |
| **Change Type** | Feature |
| **Summary** | POST /api/v1/cluster endpoint for document clustering with topic assignments. Full TDD RED→GREEN→REFACTOR cycle. |
| **Files Changed** | `src/api/topics.py` (EXTENDED), `tests/unit/api/test_wbs_b2_2_cluster_endpoint.py` (NEW) |
| **Rationale** | API endpoint to cluster documents and return per-document topic assignments with chapter metadata |
| **Git Commit** | Pending |

**Implementation Details:**

| Phase | Status | Details |
|-------|--------|---------|
| Document Analysis | ✅ Complete | CODING_PATTERNS_ANALYSIS.md Anti-Pattern 2.2 (dataclasses) reviewed |
| Anti-Pattern Audit | ✅ Complete | 0 SonarQube issues |
| RED Phase | ✅ Complete | 19 tests written before implementation |
| GREEN Phase | ✅ Complete | cluster endpoint added, all 19 tests pass |
| REFACTOR Phase | ✅ Complete | Helper method `_build_cluster_assignments()` extracted |

**New Files:**

| File | Purpose |
|------|---------|
| `tests/unit/api/test_wbs_b2_2_cluster_endpoint.py` | 19 unit tests for cluster endpoint |

**Endpoint Details:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/cluster` | POST | Cluster documents with chapter metadata |

**Request/Response Schemas:**

Request:
```json
{
  "corpus": ["Chapter 1 text...", "Chapter 2 text..."],
  "chapter_index": [{"book": "arch.json", "chapter": 1, "title": "Ch 1"}],
  "embeddings": null,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

Response:
```json
{
  "assignments": [{"book": "arch.json", "chapter": 1, "title": "Ch 1", "topic_id": 0, "topic_name": "...", "confidence": 0.87}],
  "topics": [...],
  "topic_count": 15,
  "processing_time_ms": 1234.5
}
```

---

### CL-012: BERTopic Topics Endpoint (Phase B2.1)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-16 |
| **WBS Item** | BERTOPIC_INTEGRATION_WBS.md - Phase B2.1 |
| **Change Type** | Feature |
| **Summary** | POST /api/v1/topics endpoint for topic discovery. Full TDD RED→GREEN→REFACTOR cycle. |
| **Files Changed** | `src/api/topics.py` (NEW), `src/main.py`, `tests/unit/api/test_wbs_b2_1_topics_endpoint.py` (NEW) |
| **Rationale** | API endpoint to expose BERTopic topic clustering for corpus analysis |
| **Git Commit** | Pending |

**Implementation Details:**

| Phase | Status | Details |
|-------|--------|---------|
| Document Analysis | ✅ Complete | CODING_PATTERNS_ANALYSIS.md, BERTOPIC_INTEGRATION_WBS.md reviewed |
| Anti-Pattern Audit | ✅ Complete | 0 SonarQube issues |
| RED Phase | ✅ Complete | 16 tests written before implementation |
| GREEN Phase | ✅ Complete | `topics.py` created, all 16 tests pass |
| REFACTOR Phase | ✅ Complete | Helper method extracted for complexity |

**New Files:**

| File | Purpose |
|------|---------|
| `src/api/topics.py` | POST /api/v1/topics endpoint |
| `tests/unit/api/test_wbs_b2_1_topics_endpoint.py` | 16 unit tests for endpoint |

**Endpoint Details:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/topics` | POST | Discover topics from corpus |

**Request/Response Schemas:**

Request:
```json
{
  "corpus": ["doc1", "doc2", ...],
  "min_topic_size": 2,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

Response:
```json
{
  "topics": [{"topic_id": 0, "name": "...", "keywords": [...], "size": 5}],
  "topic_count": 15,
  "model_info": {"embedding_model": "...", "bertopic_version": "..."},
  "processing_time_ms": 1234.5
}
```

---

### CL-011: BERTopic Integration Complete (Topic Clustering)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-18 |
| **WBS Item** | BERTOPIC_INTEGRATION_WBS.md - Phase B1 |
| **Change Type** | Feature |
| **Summary** | BERTopic topic clustering integrated with TDD methodology. Full RED→GREEN→REFACTOR cycle complete. |
| **Files Changed** | `src/models/bertopic_clusterer.py` (NEW), `src/models/protocols.py`, `requirements.txt`, `tests/unit/models/test_bertopic_clusterer.py` (NEW) |
| **Rationale** | Enable topic-based chapter clustering for cross-referencing per BERTOPIC_SENTENCE_TRANSFORMERS_DESIGN.md |
| **Git Commit** | Pending |

**Implementation Details:**

| Phase | Status | Details |
|-------|--------|---------|
| Document Analysis | ✅ Complete | AI_CODING_PLATFORM_ARCHITECTURE.md, BERTOPIC_SENTENCE_TRANSFORMERS_DESIGN.md reviewed |
| Anti-Pattern Audit | ✅ Complete | S1172, S3776, SonarQube issues addressed |
| RED Phase | ✅ Complete | 24 tests written before implementation |
| GREEN Phase | ✅ Complete | `bertopic_clusterer.py` created, all 24 tests pass |
| REFACTOR Phase | ✅ Complete | Cognitive complexity reduced, helper methods extracted |

**New Files:**

| File | Purpose |
|------|---------|
| `src/models/bertopic_clusterer.py` | BERTopic wrapper with KMeans fallback |
| `tests/unit/models/test_bertopic_clusterer.py` | 24 unit tests for topic clustering |

**Protocol Added:**

| Protocol | Methods |
|----------|---------|
| `TopicClustererProtocol` | `cluster()`, `get_topic_info()`, `topics`, `embedding_model`, `is_using_fallback` |

**Dependencies Added:**

| Package | Version | Purpose |
|---------|---------|---------|
| `bertopic` | >=0.16.0 | Topic modeling |
| `hdbscan` | >=0.8.29 | Density clustering (BERTopic dependency) |
| `umap-learn` | >=0.5.3 | Dimensionality reduction (BERTopic dependency) |

---

## 2025-12-15

### CL-010: SBERT Migration Complete (M5 Documentation & Rollout)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-15 |
| **WBS Item** | SBERT_EXTRACTION_MIGRATION_WBS.md - M5 Documentation & Rollout |
| **Change Type** | Documentation |
| **Summary** | SBERT migration complete. Updated README with API endpoint documentation. |
| **Files Changed** | `README.md` |
| **Rationale** | Final phase of SBERT extraction/migration per Kitchen Brigade architecture |
| **Git Commit** | Pending |

**Migration Summary:**

| Phase | Status | Tests |
|-------|--------|-------|
| M1 Code Migration | ✅ Complete | 45 tests |
| M2 API Endpoint Layer | ✅ Complete | - |
| M3 API Client Refactor | ✅ Complete | - |
| M4 Test Migration | ✅ Complete | - |
| M5 Documentation | ✅ Complete | - |

**SBERT API Endpoints (Kitchen Brigade - Sous Chef):**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/embeddings` | POST | Generate 384-dim SBERT embeddings |
| `/api/v1/similarity` | POST | Compute cosine similarity |
| `/api/v1/similar-chapters` | POST | Find top-k similar chapters |

---

## 2025-12-14

### CL-009: Phase M2 API Endpoint Layer Complete

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-14 |
| **WBS Item** | WBS 5.2 Phase M2 (API Endpoint Layer) |
| **Change Type** | Feature |
| **Summary** | Complete SBERT API with similarity, embeddings, batch, and similar-chapters endpoints |
| **Files Changed** | `src/api/similarity.py`, `src/main.py`, `src/models/sbert/__init__.py`, `tests/integration/test_wbs_5_2_similarity_endpoint.py` |
| **Rationale** | TDD implementation per SBERT_EXTRACTION_MIGRATION_WBS.md M2.1-M2.4 |
| **Git Commit** | Pending |

**Endpoints Implemented:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/similarity` | POST | Compute cosine similarity between two texts |
| `/v1/embeddings` | POST | Generate 384-dim embeddings for batch of texts |
| `/v1/similarity/batch` | POST | Compute similarity for multiple pairs |
| `/v1/similar-chapters` | POST | Find top-k similar chapters with threshold |

**Pydantic Models (M2.4.7 REFACTOR):**

| Model | Fields |
|-------|--------|
| `SimilarityRequest` | `text1: str`, `text2: str` |
| `SimilarityResponse` | `score: float`, `model: str`, `processing_time_ms: float` |
| `EmbeddingsRequest` | `texts: list[str]` |
| `EmbeddingsResponse` | `embeddings: list[list[float]]`, `model: str`, `processing_time_ms: float` |
| `BatchSimilarityRequest` | `pairs: list[SimilarityPair]` |
| `BatchSimilarityResponse` | `scores: list[float]`, `model: str`, `processing_time_ms: float` |
| `SimilarChaptersRequest` | `query: str`, `chapters: list[ChapterInput]`, `top_k: int`, `threshold: float` |
| `SimilarChaptersResponse` | `chapters: list[SimilarChapterResult]`, `method: str`, `model: str`, `processing_time_ms: float` |

**Test Summary:**

| Phase | Tests | Description |
|-------|-------|-------------|
| M1 (Code Migration) | 29 | SemanticSimilarityEngine, dependencies, anti-pattern compliance |
| M2.1 (Model Loading) | 16 | Singleton, thread safety, graceful degradation |
| M2.2 (Embeddings) | 26 | Embeddings endpoint, validation, batch processing |
| M2.3 (Batch Similarity) | 11 | Similarity symmetry (3), batch similarity (8) |
| M2.4 (Similar Chapters) | 13 | Similar chapters (5), threshold (4), method metadata (4) |
| **Total** | **95** | All passing |

**Anti-Pattern Compliance:**

| Anti-Pattern | Resolution |
|--------------|------------|
| S1192 (magic values) | `DEFAULT_MODEL_NAME`, `EMBEDDING_DIMENSIONS` exported from `__init__.py` |
| #6 (duplicate code) | Singleton `SBERTModelLoader` shared across endpoints |
| #7 (exception handling) | `SBERTModelError` with proper inheritance |
| #9 (API design) | FastAPI router pattern with Pydantic models |
| #10 (state mutation) | `asyncio.Lock` for thread safety |
| #12 (connection pooling) | Cached model instance via singleton |

**Quality Gates:**
- ✅ pytest: 88 WBS 5.2 tests passing
- ✅ ruff check: 0 errors
- ✅ mypy: 0 errors

---

## 2025-01-20

### CL-008: Phase M2.2 API Routes Complete

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-01-20 |
| **WBS Item** | WBS 5.2 Phase M2.2 (API Routes) |
| **Change Type** | Feature |
| **Summary** | Similarity and embeddings API endpoints with full TDD |
| **Files Changed** | `src/api/similarity.py`, `src/main.py`, `tests/integration/test_wbs_5_2_similarity_endpoint.py` |
| **Rationale** | TDD implementation for SBERT-powered semantic similarity API |
| **Git Commit** | Pending |

**Endpoints Implemented:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/similarity` | POST | Compute cosine similarity between two texts |
| `/v1/embeddings` | POST | Generate 384-dim embeddings for batch of texts |

**Request/Response Models:**

| Model | Fields |
|-------|--------|
| `SimilarityRequest` | `text1: str`, `text2: str` |
| `SimilarityResponse` | `score: float`, `model: str`, `processing_time_ms: float` |
| `EmbeddingsRequest` | `texts: list[str]` |
| `EmbeddingsResponse` | `embeddings: list[list[float]]`, `model: str`, `processing_time_ms: float` |

**Tests Added (M2.2):**

| Test Class | Tests | Purpose |
|------------|-------|---------|
| TestSimilarityEndpointExists | 5 | Endpoint existence and response format |
| TestSimilarityRequestValidation | 4 | 422 responses for invalid input |
| TestSimilarityResponseMetadata | 3 | Model info and timing |
| TestEmbeddingsEndpointExists | 5 | Endpoint existence and 384-dim vectors |
| TestEmbeddingsRequestValidation | 3 | 422 responses for invalid input |
| TestEmbeddingsBatchProcessing | 4 | Batch processing and timing |
| TestSimilarityAntiPatternCompliance | 2 | Singleton and error structure |

**Test Count:** 71 tests passing (M1: 29, M2.1: 16, M2.2: 26)

**Quality Gates:**
- ✅ pytest: 71 WBS 5.2 tests passing
- ✅ ruff check: 0 errors
- ✅ mypy: 0 errors

---

### CL-007: Phase M2.1 Model Loading Infrastructure Complete

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-01-20 |
| **WBS Item** | WBS 5.2 Phase M2.1 (Model Loading Infrastructure) |
| **Change Type** | Feature |
| **Summary** | SBERTModelLoader singleton with thread safety and graceful degradation |
| **Files Changed** | `src/models/sbert/model_loader.py`, `src/core/exceptions.py`, `tests/unit/models/test_wbs_5_2_sbert_model_loader.py` |
| **Rationale** | TDD implementation following RED → GREEN → REFACTOR cycle per M2 API Endpoint Layer |
| **Git Commit** | Pending |

**Implementation Details:**

| Component | Description |
|-----------|-------------|
| `SBERTModelLoader` | Singleton wrapper for SemanticSimilarityEngine with thread-safe initialization |
| `SBERTModelProtocol` | Duck-typing Protocol per CODING_PATTERNS_ANALYSIS.md line 130 |
| `SBERTModelError` | Namespaced exception (Anti-Pattern #7 compliance) |
| `get_sbert_model()` | Factory function for singleton access |
| `reset_sbert_model()` | Test helper for singleton reset |

**Anti-Pattern Compliance:**

| Anti-Pattern | Resolution |
|--------------|------------|
| #6 Duplicate Code | Single SBERTModelLoader instance (singleton pattern) |
| #7 Exception Shadowing | SBERTModelError inherits CodeOrchestratorError |
| #10 State Mutation | asyncio.Lock() protects concurrent embedding computation |
| #12 Connection Pooling | Cached model instance, lazy initialization |

**Tests Added (M2.1):**

| Test Class | Tests | Purpose |
|------------|-------|---------|
| TestSBERTModelSingleton | 5 | Singleton pattern verification |
| TestSBERTModelThreadSafety | 3 | asyncio.Lock and concurrent access |
| TestSBERTGracefulDegradation | 5 | TF-IDF fallback and error handling |
| TestSBERTModelLoaderAntiPatternCompliance | 3 | Exception inheritance and protocol adherence |

**Test Count:** 45 tests passing (M1: 29, M2.1: 16)

**Quality Gates:**
- ✅ pytest: 45 tests passing
- ✅ ruff check: 0 errors
- ✅ mypy: 0 errors

---

### CL-006: Phase M1.3 Anti-Pattern Audit Complete

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-01-20 |
| **WBS Item** | WBS 5.2 Phase M1.3 (Anti-Pattern Audit) |
| **Change Type** | Fix, Refactor |
| **Summary** | SBERT engine audited and fixed for anti-pattern compliance |
| **Files Changed** | `src/models/sbert/semantic_similarity_engine.py`, `tests/unit/models/test_wbs_5_2_sbert_engine.py` |
| **Rationale** | TDD compliance for Phase M1 - ensure code quality before Phase M2 integration |
| **Git Commit** | Pending |

**Anti-Pattern Fixes Applied:**

| Issue | Resolution |
|-------|------------|
| E402 (ruff) | Added `# noqa: E402` to intentional post-try/except sklearn imports |
| unused type: ignore | Removed unnecessary `type: ignore[assignment]` comment on line 124 |
| S1192 | Verified - `DEFAULT_MODEL_NAME` constant already extracted (2 occurrences allowed) |
| #7 exception shadowing | Verified - no bare except clauses found |
| S3776 | Verified - all functions have cognitive complexity < 15 |
| mypy | Passed with zero errors after unused ignore removal |

**Tests Added (M1.3):**

| Test Class | Test Method | Purpose |
|------------|-------------|---------|
| TestImportStructureCompliance | test_ruff_e402_resolved | Validates noqa comments present on sklearn imports |
| TestImportStructureCompliance | test_import_order_is_intentional | Validates graceful degradation pattern maintained |
| TestAntiPatternCompliance | test_s1192_no_magic_strings_tripled | Validates S1192 compliance (no 3+ duplicate literals) |
| TestAntiPatternCompliance | test_s3776_cognitive_complexity_under_limit | Validates S3776 compliance (CC < 15) |

**Test Count:** 29 tests passing (M1.1: 19, M1.2: 6, M1.3: 4)

---

### CL-005: Phase M1.1-M1.2 Code Migration & Dependency Setup

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-01-20 |
| **WBS Item** | WBS 5.2 Phase M1.1 (Code Migration), M1.2 (Dependency Setup) |
| **Change Type** | Feature |
| **Summary** | SBERT SemanticSimilarityEngine migrated from llm-document-enhancer to Code-Orchestrator-Service |
| **Files Changed** | See list below |
| **Rationale** | Kitchen Brigade architecture - Sous Chef (Code-Orchestrator) hosts all understanding models |
| **Git Commit** | Pending |

**Files Created:**
- `src/models/sbert/__init__.py` - Package initialization
- `src/models/sbert/semantic_similarity_engine.py` - Full engine with 384-dim SBERT embeddings
- `tests/unit/models/__init__.py` - Test package init
- `tests/unit/models/test_wbs_5_2_sbert_engine.py` - TDD test suite (29 tests)

**Files Modified:**
- `src/models/__init__.py` - Fixed broken imports (ExtractionResult, removed ValidatedTerm)
- `requirements.txt` - Added `scikit-learn~=1.3.0` for TF-IDF fallback
- `README.md` - Added SBERT dependencies section

**Engine API Modifications:**
- Constructor: `__init__(config, *, model_name)` - config-first signature
- Added `_tfidf_vectorizer` alias for backward compatibility
- Added `compute_similarity_matrix(texts: list[str] | NDArray)` - accepts text lists
- Added `find_similar(query='text', candidates=[...])` - text-based convenience API

---

## 2025-12-14

### CL-004: SBERT Integration Planning (Architecture Update)

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-14 |
| **WBS Item** | WBS 5.2 (SBERT Integration) |
| **Change Type** | Documentation |
| **Summary** | Architecture updated to include SBERT as a core model hosted by Code-Orchestrator-Service |
| **Files Changed** | `docs/ARCHITECTURE.md` |
| **Rationale** | SBERT is a participant in Kitchen Brigade - helps translate NL requirements from LLM gateway before CodeBERT/CodeT5+ processing |
| **Git Commit** | Pending |

**Architecture Changes:**

| Section | Update |
|---------|--------|
| Executive Summary | Added SBERT to model list alongside CodeT5+, GraphCodeBERT, CodeBERT |
| Kitchen Brigade Diagram | Added "SBERT: Translates NL requirements, finds similar chapters" to Sous Chef role |
| Model List | SBERT (`all-MiniLM-L6-v2`) - Text/NL semantic similarity |

**New API Endpoints (Planned - WBS 5.2):**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/v1/similarity` | Compare texts semantically |
| POST | `/v1/embeddings` | Generate text embeddings |
| POST | `/v1/similar-chapters` | Find similar chapters across corpus |

**Migration Impact:**
- `llm-document-enhancer` will migrate from local SBERT to API calls
- `SemanticSimilarityEngine` will become an API client
- No breaking changes - fallback to local SBERT if API unavailable

**Kitchen Brigade Alignment:**
- ✅ Sous Chef hosts ALL understanding models (SBERT, CodeBERT, GraphCodeBERT, CodeT5+)
- ✅ SBERT helps translate NL requirements before code-specific processing
- ✅ Service separation maintained (processing vs intelligence)

**Cross-References:**
- Platform TECHNICAL_CHANGE_LOG.md: CL-009 (SBERT Migration)
- AI_CODING_PLATFORM_WBS.md: Phase 5.2 (SBERT Integration)
- SBERT_EXTRACTION_MIGRATION_WBS.md: Detailed TDD migration plan

---

## 2025-12-11

### CL-003: Rename Agent to Extractor/Validator/Ranker

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-11 |
| **WBS Item** | Architecture Clarification |
| **Change Type** | Refactor |
| **Summary** | Renamed "Agent" classes to Extractor/Validator/Ranker for clarity - these are model wrappers, not autonomous agents |
| **Files Changed** | Multiple src/ files |
| **Rationale** | Per Kitchen Brigade architecture: Sous Chef hosts CodeBERT/GraphCodeBERT/CodeT5+ models. These are model wrappers that extract, validate, and rank - not autonomous agents that make decisions. Naming clarification prevents confusion with ai-agents (Expeditor) |
| **Git Commit** | `9a47143` |

**Terminology Changes:**

| Before | After | Purpose |
|--------|-------|---------|
| CodeBERTAgent | CodeBERTExtractor | Extracts NL↔code relationships |
| GraphCodeBERTAgent | GraphCodeBERTValidator | Validates code structure, data flow |
| CodeT5Agent | CodeT5Ranker | Ranks and generates draft code |

**Kitchen Brigade Alignment:**
- Sous Chef (this service) = SMART - hosts models, generates drafts
- Expeditor (ai-agents) = Orchestration - coordinates workflows
- This change clarifies that Sous Chef has "model wrappers" not "agents"

---

## 2025-12-10

### CL-002: Phase 4 - Search Integration Complete

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-10 |
| **WBS Item** | Phase 4: Search Integration |
| **Change Type** | Feature |
| **Summary** | Complete search integration with 57 new tests (270 total) |
| **Files Changed** | See table below |
| **Rationale** | WBS Phase 4 requires CodeBERT search integration with semantic-search-service |
| **Git Commit** | `b8ccc1e` |

**Implementation Details:**

| Component | Tests | Description |
|-----------|-------|-------------|
| CodeBERT Search | 15 | NL↔code search via embeddings |
| GraphCodeBERT Analysis | 12 | Structure analysis, data flow detection |
| CodeT5+ Generation | 18 | Draft code generation (decoder mode) |
| API Routes | 12 | REST endpoints for model access |

**API Endpoints Added:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/search/code` | Search codebase via CodeBERT |
| POST | `/v1/analyze/structure` | Analyze code structure via GraphCodeBERT |
| POST | `/v1/generate/draft` | Generate draft code via CodeT5+ |
| GET | `/health` | Health check |

**Test Summary:**
- New tests: 57
- Total tests: 270
- Coverage: 85%

---

## 2025-12-09

### CL-001: Initial Architecture Documentation

| Field | Value |
|-------|-------|
| **Date/Time** | 2025-12-09 |
| **WBS Item** | Project Setup |
| **Change Type** | Documentation |
| **Summary** | Initial commit with architecture docs defining Sous Chef role |
| **Files Changed** | `docs/ARCHITECTURE.md`, `README.md` |
| **Rationale** | Establish service role in Kitchen Brigade architecture |
| **Git Commit** | `1ed8354` |

**Service Definition:**

| Attribute | Value |
|-----------|-------|
| **Role** | Sous Chef |
| **Port** | 8083 |
| **Intelligence** | SMART |
| **Models** | CodeBERT, GraphCodeBERT, CodeT5+ |
| **Responsibility** | Search codebase, understand structure, GENERATE draft code |

**Kitchen Brigade Positioning:**
- Router (llm-gateway:8080) → routes requests
- Expeditor (ai-agents:8082) → orchestrates workflow
- Cookbook (semantic-search:8081) → DUMB retrieval
- **Sous Chef (this:8083) → SMART generation**
- Auditor (audit-service:8084) → validation only

---

## Cross-Repo References

| Related Repo | Document | Purpose |
|--------------|----------|---------|
| `textbooks` | `pending/platform/AI_CODING_PLATFORM_ARCHITECTURE.md` | Platform architecture |
| `textbooks` | `pending/platform/AI_CODING_PLATFORM_WBS.md` | Implementation phases |
| `llm-gateway` | `docs/Comp_Static_Analysis_Report_20251203.md` | Anti-pattern reference |
| `ai-agents` | `docs/TECHNICAL_CHANGE_LOG.md` | Expeditor changes |
