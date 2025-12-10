"""
Code-Orchestrator-Service - Structured Logging Module

WBS 1.2.4: Add structured logging with JSON logs and correlation IDs

Patterns Applied:
- One-time configure_logging() at startup (Comp_Static_Analysis_Report #16)
- structlog BoundLogger with JSON output

Anti-Patterns Avoided:
- #16: structlog.configure() called per get_logger() - PREVENTED via _configured flag

Reference:
- CODING_PATTERNS_ANALYSIS.md line 1434: structlog processor functions
- Comp_Static_Analysis_Report_20251203.md line 264: One-time configuration pattern
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import EventDict

# Module-level flag for one-time configuration (per Comp_Static_Analysis_Report #16)
_configured: bool = False


def add_service_info(
    logger: logging.Logger,  # noqa: ARG001 - Required by structlog interface
    method_name: str,  # noqa: ARG001 - Required by structlog interface
    event_dict: EventDict,
) -> EventDict:
    """Add service metadata to every log entry.

    Args:
        logger: The logger instance (unused but required by structlog interface)
        method_name: The log method name (unused but required by structlog interface)
        event_dict: The event dictionary to modify

    Returns:
        Modified event dictionary with service info
    """
    event_dict["service"] = "code-orchestrator-service"
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_output: bool = True,
) -> None:
    """Configure structlog for the application.

    This function must be called exactly ONCE at application startup.
    Per Comp_Static_Analysis_Report #16: "Application calls configure_logging() once at startup"

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Whether to use JSON renderer (True for production)
    """
    global _configured

    # Idempotent - only configure once (per Comp_Static_Analysis_Report #16)
    if _configured:
        return

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Choose renderer based on output format
    renderer: Any
    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure structlog with processors
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_service_info,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str) -> Any:
    """Get a structured logger instance.

    Per Comp_Static_Analysis_Report #16: "get_logger() no longer calls structlog.configure()"

    Args:
        name: Logger name (typically module name)

    Returns:
        structlog BoundLogger instance
    """
    return structlog.get_logger(name)


def reset_logging() -> None:
    """Reset logging configuration for testing.

    Per Comp_Static_Analysis_Report #16: "Added reset_logging() for testing"
    """
    global _configured
    _configured = False
    structlog.reset_defaults()
