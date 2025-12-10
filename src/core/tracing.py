"""
Code-Orchestrator-Service - OpenTelemetry Tracing Module

WBS 1.2.5: Add OpenTelemetry tracing with spans for each operation

Patterns Applied:
- One-time configure_tracing() at startup
- Minimal manual instrumentation (defer auto-instrumentation to later phase)

Reference:
- CODING_PATTERNS_ANALYSIS.md line 2994: OpenTelemetry in tech stack
- GUIDELINES line 2309: Observability practices
"""

from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Module-level flag for one-time configuration
_configured: bool = False

# Service resource info
SERVICE_NAME = "code-orchestrator-service"


def configure_tracing(
    service_name: str = SERVICE_NAME,
    console_export: bool = True,
) -> None:
    """Configure OpenTelemetry tracing for the application.

    This function must be called exactly ONCE at application startup.

    Args:
        service_name: Name of the service for trace attribution
        console_export: Whether to export spans to console (for development)
    """
    global _configured

    # Idempotent - only configure once
    if _configured:
        return

    # Create resource with service info
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
        }
    )

    # Create and set tracer provider
    provider = TracerProvider(resource=resource)

    # Add console exporter for development (Jaeger exporter can be added later)
    if console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(console_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    _configured = True


def get_tracer(name: str) -> Any:
    """Get a tracer instance for creating spans.

    Args:
        name: Tracer name (typically module name)

    Returns:
        OpenTelemetry Tracer instance
    """
    return trace.get_tracer(name)


def reset_tracing() -> None:
    """Reset tracing configuration for testing."""
    global _configured
    _configured = False
