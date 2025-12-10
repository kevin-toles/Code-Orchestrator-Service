"""
Code-Orchestrator-Service - Main Application Entry Point

WBS 1.2.1: Create main.py entry point
- FastAPI app with lifespan handler
- uvicorn src.main:app starts successfully

Patterns Applied:
- Lifespan context manager (CODING_PATTERNS_ANALYSIS.md line 1940)
- One-time configure_logging() at startup (Comp_Static_Analysis_Report #16)

Anti-Patterns Avoided:
- Deprecated @app.on_event - using modern lifespan pattern
- structlog.configure() per request - one-time at startup
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.extract import extract_router
from src.api.health import router as health_router
from src.api.search import search_router
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.core.tracing import configure_tracing

# Get settings
settings = get_settings()

# Configure logging ONCE at module load (per Comp_Static_Analysis_Report #16)
configure_logging(
    log_level=settings.log_level,
    json_output=settings.log_json,
)

# Get logger after configuration
logger = get_logger(__name__)


# =============================================================================
# Lifespan Context Manager - WBS 1.2.1
# Pattern: Modern lifespan instead of deprecated @app.on_event
# Reference: CODING_PATTERNS_ANALYSIS.md line 1940-1954
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup/shutdown events.

    WBS 1.2.1: FastAPI app with lifespan handler

    Uses the modern lifespan pattern instead of deprecated @app.on_event.
    Pattern: CODING_PATTERNS_ANALYSIS.md line 1940-1954
    """
    # =========================================================================
    # STARTUP
    # =========================================================================
    logger.info(
        "startup",
        service=settings.service_name,
        version=settings.version,
        environment=settings.environment,
    )

    # Configure tracing (one-time)
    if settings.tracing_enabled:
        configure_tracing(
            service_name=settings.service_name,
            console_export=settings.tracing_console_export,
        )
        logger.info("tracing_configured")

    # Initialize app state
    app.state.initialized = True
    app.state.environment = settings.environment

    # NOTE: Model loading deferred to Phase 2 (WBS 2.x)
    # When implemented: load models and call get_health_service().set_models_loaded(True)
    # For now, models_loaded remains False, so /ready returns 503

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("shutdown", service=settings.service_name)

    # Clean up resources
    app.state.initialized = False

    # NOTE: Model cleanup deferred to Phase 2 (WBS 2.x)


# =============================================================================
# FastAPI Application - WBS 1.2.1
# =============================================================================


app = FastAPI(
    title="Code-Orchestrator-Service",
    description="Sous Chef - HuggingFace model hosting for Kitchen Brigade architecture",
    version=settings.version,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(extract_router)
app.include_router(search_router)


# =============================================================================
# Root endpoint
# =============================================================================


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint redirecting to docs."""
    return {
        "service": settings.service_name,
        "version": settings.version,
        "docs": "/docs",
    }
