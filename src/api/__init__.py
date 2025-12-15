"""FastAPI routes and endpoints.

Endpoints:
- GET /health: Service health status
- GET /ready: Readiness probe (models loaded)
- POST /v1/search: Semantic search with curation

Patterns applied from GUIDELINES_AI_Engineering:
- Dependency injection for model registry
- Statelessness principle for horizontal scaling
"""
