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

## 2025-01-20

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
