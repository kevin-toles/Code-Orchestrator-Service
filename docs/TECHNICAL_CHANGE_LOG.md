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
