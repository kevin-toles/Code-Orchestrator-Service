"""
Code-Orchestrator-Service - Health API Routes

WBS 1.2.2: Implement /health endpoint
WBS 1.2.3: Implement /ready endpoint

Patterns Applied:
- Health Check Pattern (CODING_PATTERNS_ANALYSIS.md line 90)
- HealthService class with Repository pattern (llm-gateway/src/api/routes/health.py)
- Pydantic response models per FastAPI best practices

Anti-Patterns Avoided:
- Bare except clauses
- Missing response models
"""

from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.logging import get_logger

# Initialize router
router = APIRouter(tags=["health"])

# Get logger
logger = get_logger(__name__)


# =============================================================================
# Response Models - Pydantic patterns (per llm-gateway health.py)
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response model.

    WBS 1.2.2: Returns 200 with service info
    """
    status: str
    version: str
    service: str


class ReadinessResponse(BaseModel):
    """Readiness check response model.

    WBS 1.2.3: Returns 503 until models loaded
    WBS 2.x Integration Test: Returns model-specific status
    """
    status: str
    checks: dict[str, bool]
    models: dict[str, str] | None = None  # Phase 2: model-specific status


# =============================================================================
# Health Service - Repository Pattern (per CODING_PATTERNS_ANALYSIS.md)
# =============================================================================


class HealthService:
    """Service class for health check operations.

    Pattern: Health Check Pattern per CODING_PATTERNS_ANALYSIS.md line 90
    Returns structured {"status": "healthy", ...} response
    """

    def __init__(self, version: str = "0.1.0"):
        """Initialize health service.

        Args:
            version: Service version string
        """
        self._version = version
        self._models_loaded = False

    def check_health(self) -> dict[str, Any]:
        """Check basic service health.

        Returns:
            Health status dictionary with status, version, service
        """
        return {
            "status": "healthy",
            "version": self._version,
            "service": "code-orchestrator-service",
        }

    def check_readiness(self) -> tuple[dict[str, Any], bool]:
        """Check if service is ready to accept requests.

        WBS 1.2.3: Returns 503 until models loaded
        Checks SBERT model status (all-MiniLM-L6-v2)

        Returns:
            Tuple of (readiness dict, is_ready bool)
        """
        models_loaded = self._models_loaded

        checks = {
            "models_loaded": models_loaded,
        }

        is_ready = all(checks.values())
        status = "ready" if is_ready else "not_ready"

        result: dict[str, Any] = {
            "status": status,
            "checks": checks,
        }

        return result, is_ready

    def set_models_loaded(self, loaded: bool) -> None:
        """Set model loading status.

        Called by lifespan handler when models are loaded.

        Args:
            loaded: Whether models are loaded
        """
        self._models_loaded = loaded


# Global health service instance
# In production, this would be injected via dependency injection
_health_service = HealthService()


def get_health_service() -> HealthService:
    """Get health service instance.

    Pattern: Dependency injection per FastAPI patterns
    """
    return _health_service


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Basic health check endpoint for liveness probe",
)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    WBS 1.2.2: Returns 200 with service info
    Kubernetes liveness probe endpoint.
    """
    service = get_health_service()
    data = service.check_health()
    logger.debug("health_check", status=data["status"])
    return HealthResponse(**data)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
    summary="Readiness Check",
    description="Readiness check endpoint for Kubernetes readiness probe",
)
async def readiness_check() -> JSONResponse:
    """Readiness check endpoint.

    WBS 1.2.3: Returns 503 until models loaded
    Kubernetes readiness probe endpoint.

    Returns:
        200 if ready, 503 if not ready
    """
    service = get_health_service()
    data, is_ready = service.check_readiness()

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    logger.debug("readiness_check", status=data["status"], is_ready=is_ready)
    return JSONResponse(content=data, status_code=status_code)
