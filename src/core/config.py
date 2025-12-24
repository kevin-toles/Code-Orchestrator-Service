"""
Code-Orchestrator-Service - Application Configuration

WBS 1.2: FastAPI Application Shell Configuration

Patterns Applied:
- Pydantic Settings with SettingsConfigDict (CODING_PATTERNS_ANALYSIS.md Phase 1)
- Environment variable prefix COS_ for Code-Orchestrator-Service

Anti-Patterns Avoided:
- #287: Missing pydantic-settings (now in requirements.txt)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables with COS_ prefix.
    Example: COS_HOST=0.0.0.0, COS_PORT=8083

    Pattern: Pydantic Settings per CODING_PATTERNS_ANALYSIS.md Phase 1
    """

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8083

    # Application metadata
    service_name: str = "code-orchestrator-service"
    version: str = "0.1.0"
    environment: str = "development"

    # Logging configuration
    log_level: str = "INFO"
    log_json: bool = True

    # Model configuration (Phase 2)
    model_cache_dir: str = "./cache/models"
    codet5_model: str = "Salesforce/codet5p-220m"
    graphcodebert_model: str = "microsoft/graphcodebert-base"
    codebert_model: str = "microsoft/codebert-base"

    # Tracing configuration
    tracing_enabled: bool = True
    tracing_console_export: bool = True

    model_config = SettingsConfigDict(
        env_prefix="COS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars like OPENROUTER_API_KEY
    )


def get_settings() -> Settings:
    """Get application settings instance.

    Returns:
        Settings instance with values from environment
    """
    return Settings()
