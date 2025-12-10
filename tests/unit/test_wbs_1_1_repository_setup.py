"""
TDD Tests for WBS 1.1: Repository Setup
Phase: RED (Failing Tests)

These tests verify the project structure exists per WBS_IMPLEMENTATION.md
Anti-patterns checked:
- #1.1: Missing Optional type annotations (Mypy)
- Pydantic Settings pattern from CODING_PATTERNS_ANALYSIS.md Phase 1
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestProjectStructure:
    """WBS 1.1.1: Verify project directory structure exists."""

    def test_src_directory_exists(self):
        """src/ directory must exist for source code."""
        src_dir = PROJECT_ROOT / "src"
        assert src_dir.exists(), f"src/ directory missing at {src_dir}"
        assert src_dir.is_dir(), "src/ must be a directory"

    def test_tests_directory_exists(self):
        """tests/ directory must exist for test code."""
        tests_dir = PROJECT_ROOT / "tests"
        assert tests_dir.exists(), f"tests/ directory missing at {tests_dir}"
        assert tests_dir.is_dir(), "tests/ must be a directory"

    def test_docs_directory_exists(self):
        """docs/ directory must exist for documentation."""
        docs_dir = PROJECT_ROOT / "docs"
        assert docs_dir.exists(), f"docs/ directory missing at {docs_dir}"
        assert docs_dir.is_dir(), "docs/ must be a directory"

    def test_config_directory_exists(self):
        """config/ directory must exist for configuration files."""
        config_dir = PROJECT_ROOT / "config"
        assert config_dir.exists(), f"config/ directory missing at {config_dir}"
        assert config_dir.is_dir(), "config/ must be a directory"

    def test_src_has_init(self):
        """src/ must be a Python package with __init__.py."""
        init_file = PROJECT_ROOT / "src" / "__init__.py"
        assert init_file.exists(), f"src/__init__.py missing at {init_file}"

    def test_src_core_exists(self):
        """src/core/ must exist for core components (config, exceptions)."""
        core_dir = PROJECT_ROOT / "src" / "core"
        assert core_dir.exists(), f"src/core/ directory missing at {core_dir}"
        assert core_dir.is_dir(), "src/core/ must be a directory"

    def test_src_agents_exists(self):
        """src/agents/ must exist for model agent implementations."""
        agents_dir = PROJECT_ROOT / "src" / "agents"
        assert agents_dir.exists(), f"src/agents/ directory missing at {agents_dir}"
        assert agents_dir.is_dir(), "src/agents/ must be a directory"

    def test_src_api_exists(self):
        """src/api/ must exist for FastAPI routes."""
        api_dir = PROJECT_ROOT / "src" / "api"
        assert api_dir.exists(), f"src/api/ directory missing at {api_dir}"
        assert api_dir.is_dir(), "src/api/ must be a directory"


class TestPyprojectToml:
    """WBS 1.1.2: Verify pyproject.toml exists and is valid."""

    def test_pyproject_toml_exists(self):
        """pyproject.toml must exist for modern Python packaging."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists(), f"pyproject.toml missing at {pyproject}"

    def test_pyproject_has_project_name(self):
        """pyproject.toml must define project name."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "[project]" in content, "pyproject.toml must have [project] section"
        assert "code-orchestrator" in content.lower() or "code_orchestrator" in content.lower(), \
            "Project name must include 'code-orchestrator'"

    def test_pyproject_has_python_version(self):
        """pyproject.toml must require Python 3.11+."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "requires-python" in content, "pyproject.toml must specify Python version"
        # Should require 3.11 or higher
        assert "3.11" in content or "3.12" in content, "Must require Python 3.11+"


class TestRequirements:
    """WBS 1.1.3: Verify requirements.txt contains required dependencies."""

    def test_requirements_txt_exists(self):
        """requirements.txt must exist."""
        requirements = PROJECT_ROOT / "requirements.txt"
        assert requirements.exists(), f"requirements.txt missing at {requirements}"

    def test_requirements_has_transformers(self):
        """requirements.txt must include transformers for HuggingFace models."""
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text().lower()
        assert "transformers" in content, "transformers package required for HuggingFace models"

    def test_requirements_has_torch(self):
        """requirements.txt must include torch for model inference."""
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text().lower()
        assert "torch" in content, "torch package required for model inference"

    def test_requirements_has_sentence_transformers(self):
        """requirements.txt must include sentence-transformers for embeddings."""
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text().lower()
        assert "sentence-transformers" in content, "sentence-transformers required for embeddings"

    def test_requirements_has_fastapi(self):
        """requirements.txt must include fastapi for API framework."""
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text().lower()
        assert "fastapi" in content, "fastapi required for API framework"

    def test_requirements_has_pydantic_settings(self):
        """requirements.txt must include pydantic-settings per CODING_PATTERNS_ANALYSIS.md.

        Anti-pattern #287: Missing pydantic-settings (CodeRabbit finding)
        """
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text().lower()
        assert "pydantic-settings" in content, \
            "pydantic-settings required per CODING_PATTERNS_ANALYSIS.md anti-pattern #287"

    def test_requirements_has_uvicorn(self):
        """requirements.txt must include uvicorn for ASGI server."""
        requirements = PROJECT_ROOT / "requirements.txt"
        content = requirements.read_text().lower()
        assert "uvicorn" in content, "uvicorn required for ASGI server"


class TestDockerfile:
    """WBS 1.1.4: Verify Dockerfile exists and follows best practices."""

    def test_dockerfile_exists(self):
        """Dockerfile must exist for containerization."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), f"Dockerfile missing at {dockerfile}"

    def test_dockerfile_has_python_base(self):
        """Dockerfile must use Python base image."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        assert "FROM" in content, "Dockerfile must have FROM instruction"
        assert "python" in content.lower(), "Dockerfile must use Python base image"

    def test_dockerfile_has_multistage_build(self):
        """Dockerfile should use multi-stage build for smaller image size."""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        # Count FROM instructions - multi-stage has more than one
        from_count = content.count("FROM")
        assert from_count >= 2, "Dockerfile should use multi-stage build (2+ FROM instructions)"


class TestDockerCompose:
    """WBS 1.1.5: Verify docker-compose.yml exists."""

    def test_docker_compose_exists(self):
        """docker-compose.yml must exist."""
        compose = PROJECT_ROOT / "docker-compose.yml"
        assert compose.exists(), f"docker-compose.yml missing at {compose}"

    def test_docker_compose_has_service(self):
        """docker-compose.yml must define code-orchestrator service."""
        compose = PROJECT_ROOT / "docker-compose.yml"
        content = compose.read_text()
        assert "services:" in content, "docker-compose.yml must define services"
        assert "code-orchestrator" in content or "code_orchestrator" in content, \
            "docker-compose.yml must define code-orchestrator service"

    def test_docker_compose_has_port_8083(self):
        """docker-compose.yml must expose port 8083 per Kitchen Brigade architecture."""
        compose = PROJECT_ROOT / "docker-compose.yml"
        content = compose.read_text()
        assert "8083" in content, "docker-compose.yml must expose port 8083"
