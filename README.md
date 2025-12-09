# Code Understanding Orchestrator Service

> 🧑‍🍳 The **Sous Chef** in the Kitchen Brigade Architecture

A standalone microservice that coordinates multiple specialized code understanding models to dynamically extract, validate, and rank search terms from natural language queries.

## 🎯 Purpose

This service solves the **false positive problem** in cross-reference systems. Instead of hardcoded keyword mappings that match "chunk" to C++ memory allocation, it uses AI models to understand context and extract semantically relevant terms.

**Before:**
```python
# Hardcoded, brittle
FOCUS_SEARCH_TERMS = {
    "chunking": ["chunk", "split", "segment"]  # Matches C++ memory allocation!
}
```

**After:**
```python
# Dynamic, context-aware
response = orchestrator.extract(
    query="LLM document chunking with overlap",
    domain="ai-ml"
)
# Returns: ["chunking", "RAG", "text_splitter", "embedding"] ✅
# Excludes: ["chunk of memory", "memory allocation"] ✅
```

## 🏗️ Architecture

```
Customer (Claude/GPT) 
    ↓
Sous Chef (This Service) ← Extracts keywords, curates results
    ↓
Cookbook (Semantic Search) ← Dumb retrieval
    ↓
Sous Chef (Curation Phase) ← Filters irrelevant results
    ↓
Line Cook (Code Generator) ← Generates code
    ↓
Customer receives working code
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

## 🧠 Models Used

### Keyword Extraction Pipeline
| Model | Role | HuggingFace ID |
|-------|------|----------------|
| **CodeT5+** | Generator | `Salesforce/codet5p-220m` |
| **GraphCodeBERT** | Validator | `microsoft/graphcodebert-base` |
| **CodeBERT** | Ranker | `microsoft/codebert-base` |

### Code Generation (Line Cook)
| Model | Parameters | Best For |
|-------|------------|----------|
| **Qwen2.5-Coder-32B** | 32B | Primary - Production |
| **Qwen2.5-Coder-7B** | 7.6B | Fallback - Development |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/kevin-toles/Code-Orchestrator-Service.git
cd Code-Orchestrator-Service

# Install dependencies
pip install -r requirements.txt

# Download models (first run)
python scripts/download_models.py

# Run
uvicorn src.main:app --reload --port 8080
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/extract` | POST | Extract search terms from query |
| `/api/v1/validate` | POST | Validate terms against domain |
| `/api/v1/search` | POST | Full pipeline: extract + search |
| `/api/v1/generate` | POST | Generate code from context |
| `/health` | GET | Health check |

### Example Request

```bash
curl -X POST http://localhost:8080/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "query": "LLM document chunking with overlap for RAG",
    "domain": "ai-ml",
    "options": {
      "min_confidence": 0.7,
      "max_terms": 10
    }
  }'
```

### Example Response

```json
{
  "search_terms": [
    {"term": "chunking", "score": 0.95, "models_agreed": 3},
    {"term": "RAG", "score": 0.92, "models_agreed": 3},
    {"term": "embedding", "score": 0.89, "models_agreed": 3},
    {"term": "text_splitter", "score": 0.85, "models_agreed": 2}
  ],
  "excluded_terms": [
    {"term": "split", "reason": "Too generic", "models_agreed": 1}
  ],
  "metadata": {
    "processing_time_ms": 245,
    "models_used": ["codet5", "graphcodebert", "codebert"]
  }
}
```

## 📁 Project Structure

```
Code-Orchestrator-Service/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── main.py                 # FastAPI entry point
│   ├── api/                    # REST endpoints
│   ├── agents/                 # Model agents (CodeT5+, etc.)
│   ├── orchestrator/           # LangGraph state machine
│   ├── models/                 # Model loading/inference
│   └── config/                 # Settings
│
├── tests/
├── scripts/
└── docs/
    └── ARCHITECTURE.md         # Full architecture docs
```

## 🔗 Related Services

| Service | Purpose |
|---------|---------|
| [semantic-search-service](../semantic-search-service) | Vector DB queries (Cookbook) |
| [llm-gateway](../llm-gateway) | Model routing |
| [ai-agents](../ai-agents) | Main orchestration |

## 📋 Development Status

- [ ] Phase 1: Basic FastAPI structure
- [ ] Phase 2: CodeT5+ generator agent
- [ ] Phase 3: GraphCodeBERT validator agent
- [ ] Phase 4: CodeBERT ranker agent
- [ ] Phase 5: LangGraph orchestration
- [ ] Phase 6: Line Cook integration
- [ ] Phase 7: Integration tests
- [ ] Phase 8: Docker deployment

## 📄 License

MIT
