"""FastAPI routes and endpoints.

Endpoints:
- GET /health: Service health status
- GET /ready: Readiness probe (models loaded)
- POST /v1/search: Semantic search with curation
- POST /v1/codet5/summarize: Code summarization
- POST /v1/codet5/generate: Code generation from NL
- POST /v1/codet5/translate: Code translation between languages
- POST /v1/codet5/complete: Code completion
- POST /v1/codet5/understand: Code semantic analysis
- POST /v1/codet5/detect-defects: Bug detection
- POST /v1/codet5/detect-clones: Clone detection

Patterns applied from GUIDELINES_AI_Engineering:
- Dependency injection for model registry
- Statelessness principle for horizontal scaling
"""
