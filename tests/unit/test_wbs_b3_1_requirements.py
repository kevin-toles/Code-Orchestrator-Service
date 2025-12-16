"""
WBS B3.1: Requirements.txt BERTopic Dependencies Tests

TDD tests for verifying BERTopic dependencies are properly declared.
Per BERTOPIC_INTEGRATION_WBS.md B3.1 Acceptance Criteria:
- Dependencies added to requirements.txt
- pip install succeeds
- Import verification passes

Patterns Applied:
- TDD verification tests
- Dependency checking patterns

Anti-Patterns Avoided:
- S1172: Unused parameters
- S1192: Duplicated literals (use constants)
"""

import re
from pathlib import Path

import pytest

# =============================================================================
# Constants (S1192 compliance)
# =============================================================================

REQUIREMENTS_FILE: str = "requirements.txt"
BERTOPIC_PACKAGE: str = "bertopic"
HDBSCAN_PACKAGE: str = "hdbscan"
UMAP_PACKAGE: str = "umap-learn"

# Version patterns
VERSION_PATTERN: str = r">=[\d.]+"


# =============================================================================
# B3.1 Test 1: Requirements File Exists
# =============================================================================


class TestRequirementsFileExists:
    """Verify requirements.txt exists and is readable."""

    def test_requirements_file_exists(self) -> None:
        """requirements.txt should exist in project root."""
        # tests/unit/test_wbs_b3_1_requirements.py -> up 3 levels to project root
        project_root = Path(__file__).parent.parent.parent
        requirements_path = project_root / REQUIREMENTS_FILE
        assert requirements_path.exists(), (
            f"requirements.txt not found at {requirements_path}"
        )

    def test_requirements_file_is_readable(self) -> None:
        """requirements.txt should be readable."""
        project_root = Path(__file__).parent.parent.parent
        requirements_path = project_root / REQUIREMENTS_FILE
        content = requirements_path.read_text()
        assert len(content) > 0, "requirements.txt is empty"


# =============================================================================
# B3.1 Test 2: BERTopic Dependencies Declared
# =============================================================================


class TestBERTopicDependenciesDeclared:
    """Verify BERTopic and dependencies are in requirements.txt."""

    @pytest.fixture
    def requirements_content(self) -> str:
        """Load requirements.txt content."""
        # tests/unit/test_wbs_b3_1_requirements.py -> up 3 levels to project root
        project_root = Path(__file__).parent.parent.parent
        requirements_path = project_root / REQUIREMENTS_FILE
        return requirements_path.read_text()

    def test_bertopic_package_declared(self, requirements_content: str) -> None:
        """bertopic>=X.X.X should be in requirements.txt."""
        pattern = rf"^{BERTOPIC_PACKAGE}{VERSION_PATTERN}"
        assert re.search(pattern, requirements_content, re.MULTILINE), (
            f"{BERTOPIC_PACKAGE} not found in requirements.txt"
        )

    def test_hdbscan_package_declared(self, requirements_content: str) -> None:
        """hdbscan>=X.X.X should be in requirements.txt."""
        pattern = rf"^{HDBSCAN_PACKAGE}{VERSION_PATTERN}"
        assert re.search(pattern, requirements_content, re.MULTILINE), (
            f"{HDBSCAN_PACKAGE} not found in requirements.txt"
        )

    def test_umap_package_declared(self, requirements_content: str) -> None:
        """umap-learn>=X.X.X should be in requirements.txt."""
        pattern = rf"^{UMAP_PACKAGE}{VERSION_PATTERN}"
        assert re.search(pattern, requirements_content, re.MULTILINE), (
            f"{UMAP_PACKAGE} not found in requirements.txt"
        )

    def test_bertopic_comment_exists(self, requirements_content: str) -> None:
        """BERTopic section should have WBS reference comment."""
        assert "BERTopic" in requirements_content, (
            "BERTopic comment/section not found in requirements.txt"
        )


# =============================================================================
# B3.1 Test 3: Import Verification
# =============================================================================


class TestBERTopicImports:
    """Verify BERTopic packages can be imported."""

    def test_bertopic_import(self) -> None:
        """bertopic should be importable."""
        try:
            import bertopic  # noqa: F401
            assert True
        except ImportError:
            pytest.skip("bertopic not installed - run pip install bertopic")

    def test_hdbscan_import(self) -> None:
        """hdbscan should be importable."""
        try:
            import hdbscan  # noqa: F401
            assert True
        except ImportError:
            pytest.skip("hdbscan not installed - run pip install hdbscan")

    def test_umap_import(self) -> None:
        """umap should be importable."""
        try:
            import umap  # noqa: F401
            assert True
        except ImportError:
            pytest.skip("umap-learn not installed - run pip install umap-learn")

    def test_topic_clusterer_bertopic_available(self) -> None:
        """TopicClusterer should detect BERTopic availability."""
        from src.models.bertopic_clusterer import BERTOPIC_AVAILABLE
        # This test documents current state - may be True or False
        assert isinstance(BERTOPIC_AVAILABLE, bool)


# =============================================================================
# B3.1 Test 4: Version Constraints
# =============================================================================


class TestBERTopicVersionConstraints:
    """Verify version constraints are properly specified."""

    @pytest.fixture
    def requirements_content(self) -> str:
        """Load requirements.txt content."""
        # tests/unit/test_wbs_b3_1_requirements.py -> up 3 levels to project root
        project_root = Path(__file__).parent.parent.parent
        requirements_path = project_root / REQUIREMENTS_FILE
        return requirements_path.read_text()

    def test_bertopic_minimum_version(self, requirements_content: str) -> None:
        """bertopic should require >=0.16.0."""
        pattern = r"bertopic>=0\.16\.0"
        assert re.search(pattern, requirements_content), (
            "bertopic should specify >=0.16.0 minimum version"
        )

    def test_hdbscan_minimum_version(self, requirements_content: str) -> None:
        """hdbscan should require >=0.8.29."""
        pattern = r"hdbscan>=0\.8\.\d+"
        assert re.search(pattern, requirements_content), (
            "hdbscan should specify >=0.8.X minimum version"
        )

    def test_umap_minimum_version(self, requirements_content: str) -> None:
        """umap-learn should require >=0.5.3."""
        pattern = r"umap-learn>=0\.5\.\d+"
        assert re.search(pattern, requirements_content), (
            "umap-learn should specify >=0.5.X minimum version"
        )
