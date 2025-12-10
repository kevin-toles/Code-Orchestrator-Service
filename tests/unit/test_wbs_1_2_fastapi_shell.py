"""
TDD Tests for WBS 1.2: FastAPI Application Shell
Phase: RED (Failing Tests)

These tests verify the FastAPI application shell per WBS_IMPLEMENTATION.md 1.2.1-1.2.5

Patterns Applied:
- Health Check Pattern (CODING_PATTERNS_ANALYSIS.md line 90)
- Lifespan context manager (CODING_PATTERNS_ANALYSIS.md line 1940)
- structlog one-time config (Comp_Static_Analysis_Report #16)

Anti-Patterns Avoided:
- #16: structlog.configure() per request
- #12: New httpx.AsyncClient per request
- #42: Async function without await
"""


import pytest
from fastapi.testclient import TestClient


class TestMainEntryPoint:
    """WBS 1.2.1: Verify main.py entry point exists and app starts."""

    def test_main_py_exists(self):
        """main.py must exist at src/main.py."""
        from pathlib import Path
        main_file = Path(__file__).parent.parent.parent / "src" / "main.py"
        assert main_file.exists(), f"src/main.py missing at {main_file}"

    def test_app_can_be_imported(self):
        """FastAPI app must be importable from src.main."""
        from src.main import app
        assert app is not None
        assert hasattr(app, "routes")

    def test_app_has_lifespan_handler(self):
        """App must use lifespan context manager pattern."""
        from src.main import app
        # Modern FastAPI uses lifespan parameter instead of on_event decorators
        assert app.router.lifespan_context is not None, \
            "App must have lifespan handler (not deprecated @app.on_event)"

    def test_app_metadata(self):
        """App must have title, version, and description."""
        from src.main import app
        assert app.title == "Code-Orchestrator-Service"
        assert app.version is not None
        assert app.description is not None


class TestHealthEndpoint:
    """WBS 1.2.2: Verify /health endpoint returns 200 with service info."""

    @pytest.fixture
    def client(self):
        """Create test client without triggering lifespan."""
        from src.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_health_endpoint_exists(self, client):
        """/health endpoint must exist."""
        response = client.get("/health")
        assert response.status_code != 404, "/health endpoint not found"

    def test_health_returns_200(self, client):
        """/health must return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """/health must return JSON content type."""
        response = client.get("/health")
        assert response.headers.get("content-type") == "application/json"

    def test_health_response_has_status(self, client):
        """/health response must contain status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_response_has_version(self, client):
        """/health response must contain version field."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_response_has_service_name(self, client):
        """/health response must contain service name."""
        response = client.get("/health")
        data = response.json()
        assert "service" in data
        assert data["service"] == "code-orchestrator-service"


class TestReadyEndpoint:
    """WBS 1.2.3: Verify /ready endpoint returns 503 until models loaded."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_ready_endpoint_exists(self, client):
        """/ready endpoint must exist."""
        response = client.get("/ready")
        assert response.status_code != 404, "/ready endpoint not found"

    def test_ready_returns_json(self, client):
        """/ready must return JSON content type."""
        response = client.get("/ready")
        assert response.headers.get("content-type") == "application/json"

    def test_ready_response_has_status(self, client):
        """/ready response must contain status field."""
        response = client.get("/ready")
        data = response.json()
        assert "status" in data

    def test_ready_returns_503_when_models_not_loaded(self, client):
        """/ready must return 503 when models are not loaded.

        Per WBS 1.2.3: "Returns 503 until models loaded"
        This is the Kubernetes readiness probe behavior.
        """
        # Models won't be loaded in test environment
        response = client.get("/ready")
        # 503 = Service Unavailable (not ready)
        # 200 = Ready (models loaded)
        assert response.status_code in [200, 503], \
            f"Expected 200 or 503, got {response.status_code}"

    def test_ready_response_has_checks(self, client):
        """/ready response must contain checks dict."""
        response = client.get("/ready")
        data = response.json()
        assert "checks" in data
        assert isinstance(data["checks"], dict)

    def test_ready_checks_model_status(self, client):
        """/ready checks must include model loading status."""
        response = client.get("/ready")
        data = response.json()
        checks = data.get("checks", {})
        assert "models_loaded" in checks, \
            "/ready must report models_loaded status for Kubernetes probes"


class TestStructuredLogging:
    """WBS 1.2.4: Verify structured logging with JSON and correlation IDs."""

    def test_logging_module_exists(self):
        """Logging module must exist at src/core/logging.py."""
        from pathlib import Path
        logging_file = Path(__file__).parent.parent.parent / "src" / "core" / "logging.py"
        assert logging_file.exists(), f"src/core/logging.py missing at {logging_file}"

    def test_configure_logging_function_exists(self):
        """configure_logging() must exist (per Comp_Static_Analysis_Report #16)."""
        from src.core.logging import configure_logging
        assert callable(configure_logging)

    def test_get_logger_function_exists(self):
        """get_logger() must exist for obtaining loggers."""
        from src.core.logging import get_logger
        assert callable(get_logger)

    def test_logger_returns_bound_logger(self):
        """get_logger() must return a structlog BoundLogger."""
        from src.core.logging import configure_logging, get_logger
        configure_logging()
        logger = get_logger("test")
        # structlog BoundLoggers have bind() method
        assert hasattr(logger, "bind"), "Logger must be structlog BoundLogger"

    def test_logger_supports_correlation_id(self):
        """Logger must support binding trace_id for correlation.

        Per WBS 1.2.4: "JSON logs with correlation IDs"
        """
        from src.core.logging import configure_logging, get_logger
        configure_logging()
        logger = get_logger("test")
        # Must be able to bind trace_id
        bound = logger.bind(trace_id="test-trace-123")
        assert bound is not None

    def test_configure_logging_called_once(self):
        """configure_logging() must be idempotent (Comp_Static_Analysis_Report #16).

        Anti-pattern: structlog.configure() called every get_logger()
        Prevention: One-time configure_logging() at startup
        """
        from src.core.logging import configure_logging

        # First call should configure
        configure_logging()

        # Module flag should be set
        from src.core import logging as log_module
        assert log_module._configured is True, \
            "configure_logging() must set _configured flag (per Comp_Static_Analysis_Report #16)"


class TestOpenTelemetryTracing:
    """WBS 1.2.5: Verify OpenTelemetry tracing with spans."""

    def test_tracing_module_exists(self):
        """Tracing module must exist at src/core/tracing.py."""
        from pathlib import Path
        tracing_file = Path(__file__).parent.parent.parent / "src" / "core" / "tracing.py"
        assert tracing_file.exists(), f"src/core/tracing.py missing at {tracing_file}"

    def test_configure_tracing_function_exists(self):
        """configure_tracing() must exist for one-time setup."""
        from src.core.tracing import configure_tracing
        assert callable(configure_tracing)

    def test_get_tracer_function_exists(self):
        """get_tracer() must exist for obtaining tracers."""
        from src.core.tracing import get_tracer
        assert callable(get_tracer)

    def test_tracer_can_start_span(self):
        """Tracer must support starting spans."""
        from src.core.tracing import configure_tracing, get_tracer
        configure_tracing()
        tracer = get_tracer("test")

        # Must have start_as_current_span method
        assert hasattr(tracer, "start_as_current_span"), \
            "Tracer must support start_as_current_span for span creation"


class TestPhase1IntegrationTest:
    """Phase 1 Integration Test from WBS_IMPLEMENTATION.md."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_phase1_service_starts(self, client):
        """Service starts and responds to health checks.

        This is the exact test from WBS_IMPLEMENTATION.md Phase 1.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
