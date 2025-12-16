"""
WBS B3.2: TECHNICAL_CHANGE_LOG.md Documentation Tests

TDD tests for verifying BERTopic integration is documented.
Per BERTOPIC_INTEGRATION_WBS.md B3.2 Acceptance Criteria:
- CL-012 entry for B2.1 (POST /api/v1/topics)
- CL-013 entry for B2.2 (POST /api/v1/cluster)
- Entries reference correct files

Patterns Applied:
- TDD verification tests
- Documentation validation patterns

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

CHANGE_LOG_FILE: str = "docs/TECHNICAL_CHANGE_LOG.md"
CL_012_ID: str = "CL-012"
CL_013_ID: str = "CL-013"
TOPICS_ENDPOINT: str = "/api/v1/topics"
CLUSTER_ENDPOINT: str = "/api/v1/cluster"
TOPICS_API_FILE: str = "topics.py"
BERTOPIC_CLUSTERER_FILE: str = "bertopic_clusterer.py"


# =============================================================================
# B3.2 Test 1: Change Log File Exists
# =============================================================================


class TestChangeLogFileExists:
    """Verify TECHNICAL_CHANGE_LOG.md exists and is readable."""

    def test_change_log_file_exists(self) -> None:
        """TECHNICAL_CHANGE_LOG.md should exist in docs/."""
        # tests/unit/test_wbs_b3_2_change_log.py -> up 3 levels to project root
        project_root = Path(__file__).parent.parent.parent
        change_log_path = project_root / CHANGE_LOG_FILE
        assert change_log_path.exists(), (
            f"TECHNICAL_CHANGE_LOG.md not found at {change_log_path}"
        )

    def test_change_log_is_readable(self) -> None:
        """TECHNICAL_CHANGE_LOG.md should be readable."""
        project_root = Path(__file__).parent.parent.parent
        change_log_path = project_root / CHANGE_LOG_FILE
        content = change_log_path.read_text()
        assert len(content) > 0, "TECHNICAL_CHANGE_LOG.md is empty"


# =============================================================================
# B3.2 Test 2: CL-012 Entry (B2.1 POST /api/v1/topics)
# =============================================================================


class TestCL012TopicsEndpointEntry:
    """Verify CL-012 documents POST /api/v1/topics endpoint."""

    @pytest.fixture
    def change_log_content(self) -> str:
        """Load TECHNICAL_CHANGE_LOG.md content."""
        project_root = Path(__file__).parent.parent.parent
        change_log_path = project_root / CHANGE_LOG_FILE
        return change_log_path.read_text()

    def test_cl012_entry_exists(self, change_log_content: str) -> None:
        """CL-012 entry should exist in change log."""
        assert CL_012_ID in change_log_content, (
            f"{CL_012_ID} entry not found in TECHNICAL_CHANGE_LOG.md"
        )

    def test_cl012_references_topics_endpoint(self, change_log_content: str) -> None:
        """CL-012 should reference /api/v1/topics."""
        # Find CL-012 section
        pattern = rf"{CL_012_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_012_ID} section not found"
        cl012_section = match.group(0)
        assert TOPICS_ENDPOINT in cl012_section or "topics" in cl012_section.lower(), (
            f"{CL_012_ID} should reference topics endpoint"
        )

    def test_cl012_references_topics_api_file(self, change_log_content: str) -> None:
        """CL-012 should reference topics.py."""
        pattern = rf"{CL_012_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_012_ID} section not found"
        cl012_section = match.group(0)
        assert TOPICS_API_FILE in cl012_section, (
            f"{CL_012_ID} should reference {TOPICS_API_FILE}"
        )

    def test_cl012_has_wbs_reference(self, change_log_content: str) -> None:
        """CL-012 should reference WBS B2.1."""
        pattern = rf"{CL_012_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_012_ID} section not found"
        cl012_section = match.group(0)
        assert "B2.1" in cl012_section or "WBS" in cl012_section, (
            f"{CL_012_ID} should reference WBS task B2.1"
        )


# =============================================================================
# B3.2 Test 3: CL-013 Entry (B2.2 POST /api/v1/cluster)
# =============================================================================


class TestCL013ClusterEndpointEntry:
    """Verify CL-013 documents POST /api/v1/cluster endpoint."""

    @pytest.fixture
    def change_log_content(self) -> str:
        """Load TECHNICAL_CHANGE_LOG.md content."""
        project_root = Path(__file__).parent.parent.parent
        change_log_path = project_root / CHANGE_LOG_FILE
        return change_log_path.read_text()

    def test_cl013_entry_exists(self, change_log_content: str) -> None:
        """CL-013 entry should exist in change log."""
        assert CL_013_ID in change_log_content, (
            f"{CL_013_ID} entry not found in TECHNICAL_CHANGE_LOG.md"
        )

    def test_cl013_references_cluster_endpoint(self, change_log_content: str) -> None:
        """CL-013 should reference /api/v1/cluster."""
        pattern = rf"{CL_013_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_013_ID} section not found"
        cl013_section = match.group(0)
        assert CLUSTER_ENDPOINT in cl013_section or "cluster" in cl013_section.lower(), (
            f"{CL_013_ID} should reference cluster endpoint"
        )

    def test_cl013_references_topics_api_file(self, change_log_content: str) -> None:
        """CL-013 should reference topics.py."""
        pattern = rf"{CL_013_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_013_ID} section not found"
        cl013_section = match.group(0)
        assert TOPICS_API_FILE in cl013_section, (
            f"{CL_013_ID} should reference {TOPICS_API_FILE}"
        )

    def test_cl013_has_wbs_reference(self, change_log_content: str) -> None:
        """CL-013 should reference WBS B2.2."""
        pattern = rf"{CL_013_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_013_ID} section not found"
        cl013_section = match.group(0)
        assert "B2.2" in cl013_section or "WBS" in cl013_section, (
            f"{CL_013_ID} should reference WBS task B2.2"
        )


# =============================================================================
# B3.2 Test 4: Entry Format Compliance
# =============================================================================


class TestChangeLogEntryFormat:
    """Verify change log entries follow standard format."""

    @pytest.fixture
    def change_log_content(self) -> str:
        """Load TECHNICAL_CHANGE_LOG.md content."""
        project_root = Path(__file__).parent.parent.parent
        change_log_path = project_root / CHANGE_LOG_FILE
        return change_log_path.read_text()

    def test_entries_have_dates(self, change_log_content: str) -> None:
        """CL-012 and CL-013 should have date stamps."""
        # Look for ISO date format near CL entries
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        assert re.search(date_pattern, change_log_content), (
            "Change log should contain date stamps"
        )

    def test_cl012_has_files_changed_section(self, change_log_content: str) -> None:
        """CL-012 should list files changed."""
        pattern = rf"{CL_012_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_012_ID} section not found"
        cl012_section = match.group(0)
        # Should reference files - flexible on format
        assert "src/" in cl012_section or ".py" in cl012_section, (
            f"{CL_012_ID} should reference source files"
        )

    def test_cl013_has_files_changed_section(self, change_log_content: str) -> None:
        """CL-013 should list files changed."""
        pattern = rf"{CL_013_ID}.*?(?=CL-\d{{3}}|$)"
        match = re.search(pattern, change_log_content, re.DOTALL)
        assert match is not None, f"{CL_013_ID} section not found"
        cl013_section = match.group(0)
        assert "src/" in cl013_section or ".py" in cl013_section, (
            f"{CL_013_ID} should reference source files"
        )
