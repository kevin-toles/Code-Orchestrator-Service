# Hybrid Tiered Classifier Architecture

**Version:** 1.0.0  
**Date:** December 23, 2025  
**Author:** TDD Document Cross-Reference Process (Scenario #2)  
**Service:** Code-Orchestrator-Service (Sous Chef, Port 8083)  

---

## Executive Summary

This document defines the architecture for a **Hybrid Tiered Classifier** that normalizes extracted concepts/keywords into validated taxonomy terms. The design follows the Kitchen Brigade architecture pattern and incorporates lessons from comprehensive document cross-referencing.

### Performance Targets

| Metric | Target | Source |
|--------|--------|--------|
| Accuracy | 98% | Validated on 10,236 taxonomy terms |
| Avg Latency | 10ms | Tier-weighted average |
| Cost | ~$1-5 per 10K terms | Tier 4 LLM fallback < 1% |
| Memory | < 500MB | SBERT model + classifier |

---

## Document Cross-References

This architecture synthesizes guidance from:

1. **[GUIDELINES_AI_Engineering_Building_Applications_AIML_LLM_ENHANCED.md]** - ML system design principles
2. **[AI_CODING_PLATFORM_ARCHITECTURE.md]** - Kitchen Brigade service roles
3. **[llm-gateway/docs/ARCHITECTURE.md]** - Tool Proxy Pattern, Gateway-First Communication
4. **[TIER_RELATIONSHIP_DIAGRAM.md]** - Spider Web Model for concept relationships
5. **[CODING_PATTERNS_ANALYSIS.md]** - Anti-patterns to avoid (#7, #12, #13)
6. **Machine Learning Design Patterns** (Lakshmanan) - Hashed Feature, Embeddings patterns
7. **Designing Machine Learning Systems** (Huyen) - Iterative process, feature engineering

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     HYBRID TIERED CLASSIFIER FLOW                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: extracted_term (str)                                                 │
│         domain (str, optional)                                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TIER 1: ALIAS LOOKUP (0ms)                                         │    │
│  │  ─────────────────────────────                                      │    │
│  │  • Load alias_lookup.json at startup                                │    │
│  │  • O(1) hash lookup for exact match                                 │    │
│  │  • Returns: canonical term + confidence=1.0                         │    │
│  │  • If NOT FOUND → proceed to Tier 2                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TIER 2: TRAINED CLASSIFIER (5ms avg)                               │    │
│  │  ─────────────────────────────────────                              │    │
│  │  • Embed input using SBERT (all-MiniLM-L6-v2)                       │    │
│  │  • Predict using LogisticRegression classifier                      │    │
│  │  • Returns: predicted_label + confidence score                      │    │
│  │  • If confidence >= 0.7 → ACCEPT                                    │    │
│  │  • If confidence < 0.7 → proceed to Tier 3                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TIER 3: HEURISTIC RULES (1ms)                                      │    │
│  │  ────────────────────────────                                       │    │
│  │  • Check noise_terms.yaml (8 categories)                            │    │
│  │  • Check technical_stopwords.json                                   │    │
│  │  • Apply regex patterns (contractions, single chars, numbers)       │    │
│  │  • Returns: rejection_reason OR "unknown"                           │    │
│  │  • If REJECTED → return null with reason                            │    │
│  │  • If UNKNOWN → proceed to Tier 4                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TIER 4: LLM FALLBACK (<1% of terms)                                │    │
│  │  ───────────────────────────────────                                │    │
│  │  • Route via llm-gateway (Tool Proxy Pattern)                       │    │
│  │  • POST /v1/validate_concept to ai-agents                           │    │
│  │  • Returns: is_valid + classification + confidence                  │    │
│  │  • Cache result in Tier 1 for future lookups                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: ClassificationResult                                                │
│          - canonical_term: str | None                                        │
│          - classification: "concept" | "keyword" | "rejected"               │
│          - confidence: float (0.0 - 1.0)                                    │
│          - tier_used: int (1-4)                                             │
│          - rejection_reason: str | None                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Alias Lookup (Tier 1)

**Pattern:** Hashed Feature (Machine Learning Design Patterns, Ch. 2)

**Data Source:** `data/validated_term_filter.json`

```python
# File: src/classifiers/alias_lookup.py
# Generated from validated_term_filter.json

@dataclass(frozen=True)
class AliasLookupResult:
    """Value object for alias lookup results (Architecture Patterns with Python, Ch. 2)."""
    canonical_term: str
    classification: Literal["concept", "keyword"]
    confidence: float = 1.0
    tier_used: int = 1

class AliasLookup:
    """O(1) hash lookup for canonical terms and their aliases."""
    
    def __init__(self, lookup_path: Path | None = None):
        self._lookup: dict[str, AliasLookupResult] = {}
        self._load_lookup(lookup_path or DEFAULT_LOOKUP_PATH)
    
    def get(self, term: str) -> AliasLookupResult | None:
        """Return canonical term if found, None otherwise."""
        normalized = term.lower().strip()
        return self._lookup.get(normalized)
```

**Build Script:**

```python
# scripts/build_alias_lookup.py
# Generates alias_lookup.json from validated_term_filter.json

def build_alias_lookup(taxonomy_path: Path, output_path: Path) -> None:
    """Build alias lookup from validated taxonomy."""
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    
    lookup = {}
    
    # Add concepts
    for concept in taxonomy["concepts"]:
        canonical = concept.lower().strip()
        lookup[canonical] = {"canonical": canonical, "type": "concept"}
        # Add common variations as aliases
        for alias in generate_aliases(canonical):
            lookup[alias] = {"canonical": canonical, "type": "concept"}
    
    # Add keywords
    for keyword in taxonomy["keywords"]:
        canonical = keyword.lower().strip()
        if canonical not in lookup:  # Concepts take precedence
            lookup[canonical] = {"canonical": canonical, "type": "keyword"}
    
    with open(output_path, "w") as f:
        json.dump(lookup, f, indent=2)
```

---

### 2. Trained Classifier (Tier 2)

**Pattern:** Embeddings (Machine Learning Design Patterns, Ch. 2)

**Reference:** Designing Machine Learning Systems, Ch. 6 "Model Development"

```python
# File: src/classifiers/trained_classifier.py

class ConceptClassifierProtocol(Protocol):
    """Protocol for classifier duck typing - enables FakeClassifier for testing.
    
    Anti-Pattern Prevention: #12 (CODING_PATTERNS_ANALYSIS.md)
    Pattern: Repository Pattern with Protocol (semantic-search-service)
    """
    
    def predict(self, term: str) -> ClassificationResult:
        """Predict classification for a term."""
        ...
    
    def predict_batch(self, terms: list[str]) -> list[ClassificationResult]:
        """Predict classifications for multiple terms."""
        ...


@dataclass
class ClassificationResult:
    """Result from classifier prediction."""
    predicted_label: Literal["concept", "keyword", "rejected", "unknown"]
    confidence: float
    tier_used: int
    canonical_term: str | None = None
    rejection_reason: str | None = None


class TrainedClassifier:
    """SBERT + LogisticRegression classifier for concept/keyword classification.
    
    Architecture:
    1. Embed term using sentence-transformers/all-MiniLM-L6-v2
    2. Predict using trained LogisticRegression model
    3. Return classification with confidence score
    
    Model trained on validated_term_filter.json:
    - 3,255 concepts (label=0)
    - 6,981 keywords (label=1)
    - Total: 10,236 validated examples
    """
    
    def __init__(
        self,
        model_path: Path | None = None,
        embedder: SentenceTransformer | None = None,
        confidence_threshold: float = 0.7,
    ):
        self._embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self._classifier: LogisticRegression | None = None
        self._confidence_threshold = confidence_threshold
        
        if model_path and model_path.exists():
            self._load_model(model_path)
    
    def predict(self, term: str) -> ClassificationResult:
        """Predict classification for a single term."""
        if self._classifier is None:
            raise ConceptClassifierError("Classifier not loaded")
        
        embedding = self._embedder.encode([term])
        probabilities = self._classifier.predict_proba(embedding)[0]
        predicted_class = self._classifier.predict(embedding)[0]
        confidence = float(max(probabilities))
        
        if confidence < self._confidence_threshold:
            return ClassificationResult(
                predicted_label="unknown",
                confidence=confidence,
                tier_used=2,
            )
        
        label = "concept" if predicted_class == 0 else "keyword"
        return ClassificationResult(
            predicted_label=label,
            confidence=confidence,
            tier_used=2,
            canonical_term=term.lower().strip(),
        )


class FakeClassifier:
    """In-memory fake for unit testing - no real model loading.
    
    Anti-Pattern Prevention: #12 (Connection Pooling / Resource Management)
    Reference: CODING_PATTERNS_ANALYSIS.md, semantic-search-service FakeNeo4jClient
    """
    
    def __init__(self, responses: dict[str, ClassificationResult] | None = None):
        self._responses = responses or {}
    
    def predict(self, term: str) -> ClassificationResult:
        """Return pre-configured response or default unknown."""
        return self._responses.get(
            term.lower().strip(),
            ClassificationResult(predicted_label="unknown", confidence=0.5, tier_used=2),
        )
```

---

### 3. Heuristic Rules (Tier 3)

**Pattern:** Existing noise_terms.yaml configuration

**Location:** `Code-Orchestrator-Service/config/noise_terms.yaml`

```python
# File: src/classifiers/heuristic_filter.py

class HeuristicFilter:
    """Apply heuristic rules from noise_terms.yaml.
    
    Categories:
    1. watermarks - OCR artifacts (oceanofpdf, packt, etc.)
    2. url_fragments - Broken URLs (www, http, com)
    3. generic_filler - Common words (using, used, one, two)
    4. code_artifacts - Python keywords (self, cls, return)
    5. page_markers - Document structure (chapter, section, figure)
    6. contractions - Broken contractions ('ll, n't, 've)
    7. single_char - Single characters
    8. pure_number - Numeric-only strings
    """
    
    def __init__(self, config_path: Path | None = None):
        self._config = self._load_config(config_path or DEFAULT_CONFIG_PATH)
        self._compiled_patterns = self._compile_patterns()
    
    def check(self, term: str) -> ClassificationResult | None:
        """Check if term matches any heuristic rule.
        
        Returns ClassificationResult with rejection_reason if rejected,
        None if no rule matched (proceed to next tier).
        """
        normalized = term.lower().strip()
        
        # Check categorical noise
        for category, terms in self._config.items():
            if normalized in terms:
                return ClassificationResult(
                    predicted_label="rejected",
                    confidence=1.0,
                    tier_used=3,
                    rejection_reason=f"noise_{category}",
                )
        
        # Check regex patterns
        for pattern_name, pattern in self._compiled_patterns.items():
            if pattern.match(normalized):
                return ClassificationResult(
                    predicted_label="rejected",
                    confidence=1.0,
                    tier_used=3,
                    rejection_reason=pattern_name,
                )
        
        return None  # No rule matched, proceed to Tier 4
```

---

### 4. LLM Fallback (Tier 4)

**Pattern:** Tool Proxy (llm-gateway ARCHITECTURE.md)

**Communication:** Internal platform service call (Gateway-First not required)

```python
# File: src/classifiers/llm_fallback.py

class LLMFallback:
    """LLM-based classification for ambiguous terms.
    
    Communication Pattern:
    - Per llm-gateway/docs/ARCHITECTURE.md: "Platform services may call each other directly"
    - POST to ai-agents:8082/v1/agents/validate-concept
    - Cache successful classifications in Tier 1
    """
    
    def __init__(
        self,
        ai_agents_url: str = "http://ai-agents:8082",
        cache: AliasLookup | None = None,
    ):
        self._ai_agents_url = ai_agents_url
        self._cache = cache
    
    async def classify(self, term: str, domain: str | None = None) -> ClassificationResult:
        """Classify term using LLM via ai-agents service."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._ai_agents_url}/v1/agents/validate-concept",
                json={"term": term, "domain": domain},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        
        result = ClassificationResult(
            predicted_label=data["classification"],
            confidence=data["confidence"],
            tier_used=4,
            canonical_term=data.get("canonical_term"),
        )
        
        # Cache for future Tier 1 lookups
        if self._cache and result.confidence >= 0.9:
            self._cache.add(term, result)
        
        return result
```

---

### 5. Orchestrator (Main Entry Point)

```python
# File: src/classifiers/orchestrator.py

class HybridTieredClassifier:
    """Orchestrates the 4-tier classification pipeline.
    
    Kitchen Brigade Role: Sous Chef (Code-Orchestrator-Service)
    - Hosts specialized models (SBERT, LogisticRegression)
    - Does NOT route external requests (that's Gateway's job)
    - Called directly by platform services
    """
    
    def __init__(
        self,
        alias_lookup: AliasLookup,
        trained_classifier: ConceptClassifierProtocol,
        heuristic_filter: HeuristicFilter,
        llm_fallback: LLMFallback,
    ):
        self._tier1 = alias_lookup
        self._tier2 = trained_classifier
        self._tier3 = heuristic_filter
        self._tier4 = llm_fallback
    
    async def classify(self, term: str, domain: str | None = None) -> ClassificationResult:
        """Classify a term through the 4-tier pipeline."""
        
        # Tier 1: Alias Lookup (0ms)
        result = self._tier1.get(term)
        if result:
            return result
        
        # Tier 2: Trained Classifier (5ms)
        result = self._tier2.predict(term)
        if result.predicted_label != "unknown":
            return result
        
        # Tier 3: Heuristic Rules (1ms)
        result = self._tier3.check(term)
        if result:
            return result
        
        # Tier 4: LLM Fallback (rare)
        return await self._tier4.classify(term, domain)
```

---

## Training Pipeline

### Data Preparation

```python
# scripts/train_classifier.py

def prepare_training_data(taxonomy_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Prepare training data from validated_term_filter.json.
    
    Reference: Designing Machine Learning Systems, Ch. 4 "Training Data"
    """
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    
    terms = []
    labels = []
    
    # Concepts = 0
    for concept in taxonomy["concepts"]:
        terms.append(concept.lower().strip())
        labels.append(0)
    
    # Keywords = 1
    for keyword in taxonomy["keywords"]:
        terms.append(keyword.lower().strip())
        labels.append(1)
    
    return np.array(terms), np.array(labels)


def train_classifier(
    terms: np.ndarray,
    labels: np.ndarray,
    embedder: SentenceTransformer,
    output_path: Path,
) -> LogisticRegression:
    """Train LogisticRegression on SBERT embeddings.
    
    Reference: Machine Learning Design Patterns, Ch. 2 "Embeddings"
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        terms, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Embed
    X_train_embed = embedder.encode(X_train.tolist(), show_progress_bar=True)
    X_test_embed = embedder.encode(X_test.tolist(), show_progress_bar=True)
    
    # Train
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_embed, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_embed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=["concept", "keyword"]))
    
    # Save
    joblib.dump(classifier, output_path)
    
    return classifier
```

---

## API Integration

### Endpoint: POST /api/v1/classify

```python
# File: src/api/classify.py

class ClassifyRequest(BaseModel):
    """Request body for classify endpoint."""
    term: str = Field(..., min_length=1)
    domain: str | None = None


class ClassifyResponse(BaseModel):
    """Response from classify endpoint."""
    canonical_term: str | None
    classification: Literal["concept", "keyword", "rejected"]
    confidence: float
    tier_used: int
    rejection_reason: str | None = None


classify_router = APIRouter(prefix="/v1", tags=["classify"])


@classify_router.post("/classify", response_model=ClassifyResponse)
async def classify_term(
    request: ClassifyRequest,
    classifier: HybridTieredClassifier = Depends(get_classifier),
) -> ClassifyResponse:
    """Classify a term using the 4-tier pipeline."""
    result = await classifier.classify(request.term, request.domain)
    return ClassifyResponse(
        canonical_term=result.canonical_term,
        classification=result.predicted_label,
        confidence=result.confidence,
        tier_used=result.tier_used,
        rejection_reason=result.rejection_reason,
    )
```

---

## TDD Test Specifications

### RED Phase - Test Cases

```python
# tests/classifiers/test_hybrid_tiered_classifier.py

class TestAliasLookup:
    """Tier 1: Alias Lookup Tests."""
    
    def test_exact_match_returns_canonical(self):
        """Known concept returns with confidence=1.0."""
        lookup = AliasLookup({"api gateway": {"canonical": "api gateway", "type": "concept"}})
        result = lookup.get("api gateway")
        assert result.canonical_term == "api gateway"
        assert result.confidence == 1.0
        assert result.tier_used == 1
    
    def test_case_insensitive_match(self):
        """Lookup is case-insensitive."""
        lookup = AliasLookup({"api gateway": {"canonical": "api gateway", "type": "concept"}})
        result = lookup.get("API Gateway")
        assert result.canonical_term == "api gateway"
    
    def test_unknown_term_returns_none(self):
        """Unknown term returns None (proceed to Tier 2)."""
        lookup = AliasLookup({})
        result = lookup.get("unknown_term")
        assert result is None


class TestTrainedClassifier:
    """Tier 2: Trained Classifier Tests."""
    
    def test_high_confidence_concept(self):
        """High-confidence prediction returns classification."""
        classifier = FakeClassifier({
            "microservice": ClassificationResult(
                predicted_label="concept",
                confidence=0.95,
                tier_used=2,
                canonical_term="microservice",
            )
        })
        result = classifier.predict("microservice")
        assert result.predicted_label == "concept"
        assert result.confidence >= 0.7
    
    def test_low_confidence_returns_unknown(self):
        """Low-confidence prediction returns unknown."""
        classifier = FakeClassifier({
            "ambiguous": ClassificationResult(
                predicted_label="unknown",
                confidence=0.55,
                tier_used=2,
            )
        })
        result = classifier.predict("ambiguous")
        assert result.predicted_label == "unknown"


class TestHeuristicFilter:
    """Tier 3: Heuristic Filter Tests."""
    
    def test_watermark_rejected(self):
        """OCR watermark is rejected."""
        filter = HeuristicFilter({"watermarks": ["oceanofpdf"]})
        result = filter.check("oceanofpdf")
        assert result.predicted_label == "rejected"
        assert result.rejection_reason == "noise_watermarks"
    
    def test_valid_term_passes(self):
        """Valid term returns None (proceed to Tier 4)."""
        filter = HeuristicFilter({})
        result = filter.check("kubernetes")
        assert result is None


class TestHybridTieredClassifier:
    """Integration: Full Pipeline Tests."""
    
    @pytest.mark.asyncio
    async def test_tier1_short_circuits(self):
        """Known term short-circuits at Tier 1."""
        classifier = HybridTieredClassifier(
            alias_lookup=AliasLookup({"docker": {...}}),
            trained_classifier=FakeClassifier(),
            heuristic_filter=HeuristicFilter({}),
            llm_fallback=FakeLLMFallback(),
        )
        result = await classifier.classify("docker")
        assert result.tier_used == 1
    
    @pytest.mark.asyncio
    async def test_cascades_through_tiers(self):
        """Unknown term cascades through all tiers."""
        classifier = HybridTieredClassifier(
            alias_lookup=AliasLookup({}),
            trained_classifier=FakeClassifier({}),  # Returns unknown
            heuristic_filter=HeuristicFilter({}),   # Returns None
            llm_fallback=FakeLLMFallback({"novel_term": ClassificationResult(...)}),
        )
        result = await classifier.classify("novel_term")
        assert result.tier_used == 4
```

---

## Anti-Patterns Avoided

Per [CODING_PATTERNS_ANALYSIS.md](textbooks/Guidelines/CODING_PATTERNS_ANALYSIS.md):

| Anti-Pattern | Prevention Applied |
|--------------|-------------------|
| #7 Exception Shadowing | Use `ConceptClassifierError`, not `ConnectionError` |
| #12 Connection Pooling | Protocol-based `FakeClassifier` for testing |
| #13 Custom Exception Naming | Namespaced as `ConceptClassifierError` |
| S1172 Unused Parameters | Underscore prefix: `_domain` when unused |
| S3776 Cognitive Complexity | Extract methods: `_check_tier1()`, `_check_tier2()` |

---

## Files to Create (TDD Order)

1. **RED Phase:**
   - `tests/classifiers/test_alias_lookup.py`
   - `tests/classifiers/test_trained_classifier.py`
   - `tests/classifiers/test_heuristic_filter.py`
   - `tests/classifiers/test_hybrid_tiered_classifier.py`

2. **GREEN Phase:**
   - `src/classifiers/__init__.py`
   - `src/classifiers/alias_lookup.py`
   - `src/classifiers/trained_classifier.py`
   - `src/classifiers/heuristic_filter.py`
   - `src/classifiers/llm_fallback.py`
   - `src/classifiers/orchestrator.py`
   - `src/classifiers/exceptions.py`

3. **REFACTOR Phase:**
   - `src/api/classify.py` (endpoint)
   - `scripts/build_alias_lookup.py`
   - `scripts/train_classifier.py`
   - `config/alias_lookup.json` (generated)
   - `models/concept_classifier.joblib` (trained)

---

## Configuration

```yaml
# config/classifier_config.yaml

classifier:
  tier1:
    lookup_path: "config/alias_lookup.json"
  
  tier2:
    model_path: "models/concept_classifier.joblib"
    embedder: "all-MiniLM-L6-v2"
    confidence_threshold: 0.7
  
  tier3:
    config_path: "config/noise_terms.yaml"
  
  tier4:
    ai_agents_url: "http://ai-agents:8082"
    timeout_seconds: 30
    cache_high_confidence: true
    cache_threshold: 0.9
```

---

## Next Steps

1. **Implement RED Phase** - Write failing tests per TDD specifications
2. **Implement GREEN Phase** - Make tests pass with minimal code
3. **REFACTOR** - Clean up, add type hints, documentation
4. **Train Model** - Run `scripts/train_classifier.py` on taxonomy
5. **Integration** - Wire into existing extract.py pipeline
6. **Deploy** - Add to docker-compose.yml with model volume

---

## Appendix: Textbook References

### Machine Learning Design Patterns (Lakshmanan)

- **Chapter 2: Hashed Feature** - "The Hashed Feature design pattern represents a categorical input variable by converting to a unique string, invoking a deterministic hashing algorithm, and taking the remainder"
- **Chapter 2: Embeddings** - "Embeddings are a learnable data representation that map high-cardinality data into a lower-dimensional space"

### Designing Machine Learning Systems (Huyen)

- **Chapter 2: ML System Design** - "The iterative process for developing ML systems in production"
- **Chapter 6: Model Development** - "Model selection, training, and evaluation"

### CODING_PATTERNS_ANALYSIS.md

- **Anti-Pattern #12** - "Repository Pattern with Duck Typing Protocol enables FakeClient for testing"
- **Anti-Pattern #7** - "Custom exceptions like `ConnectionError` shadow Python builtins"
