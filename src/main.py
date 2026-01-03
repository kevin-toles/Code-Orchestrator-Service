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

from src.api.classify import classify_router
from src.api.codebert import router as codebert_router
from src.api.concepts import concepts_router
from src.api.embed import router as embed_router
from src.api.extract import extract_router
from src.api.health import router as health_router
from src.api.keywords import keywords_router
from src.api.metadata import metadata_router
from src.api.search import search_router
from src.api.similarity import similarity_router
from src.api.topics import router as topics_router
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
    Kitchen Brigade: Load all BERT models at startup for healthy service

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

    # =========================================================================
    # MODEL LOADING - Load SBERT model at startup
    # =========================================================================
    from src.api.health import get_health_service
    from src.models.sbert.model_loader import get_sbert_model

    health_service = get_health_service()
    models_loaded = True

    try:
        # Load SBERT model (all-MiniLM-L6-v2) for similarity/extraction/validation
        logger.info("loading_sbert_model")
        sbert_loader = get_sbert_model()
        app.state.sbert_model = sbert_loader
        logger.info(
            "sbert_model_loaded",
            model="all-MiniLM-L6-v2",
            using_fallback=sbert_loader.using_fallback,
        )
        
        # Pre-warm the model wrappers
        # Note: CodeT5+ removed - it's a code generation model, not keyword extractor
        from src.models.codebert_ranker import CodeBERTRanker
        from src.models.graphcodebert_validator import GraphCodeBERTValidator
        
        logger.info("initializing_model_wrappers")
        _ = CodeBERTRanker()   # Warms up ranker
        _ = GraphCodeBERTValidator()  # Warms up validator
        logger.info("model_wrappers_initialized")
        
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        models_loaded = False

    if models_loaded:
        health_service.set_models_loaded(True)
        logger.info("all_models_ready")

    yield

    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    logger.info("shutdown", service=settings.service_name)

    # Clean up resources
    app.state.initialized = False

    # Reset model loaders
    from src.models.sbert.model_loader import reset_sbert_model
    reset_sbert_model()


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
app.include_router(keywords_router, prefix="/api")  # MSE-1.4: Keywords endpoint
app.include_router(search_router)
app.include_router(similarity_router)
app.include_router(topics_router)
app.include_router(embed_router)  # EEP-1.5.7: Multi-modal embedding endpoints
app.include_router(concepts_router, prefix="/api")  # EEP-2.4: Concept extraction endpoint
app.include_router(codebert_router)  # EEP-5.2: CodeBERT embedding endpoint
app.include_router(metadata_router, prefix="/api")  # CME-1.4: Metadata extraction endpoint
app.include_router(classify_router, prefix="/api")  # HTC-1.0: Classification endpoint


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
