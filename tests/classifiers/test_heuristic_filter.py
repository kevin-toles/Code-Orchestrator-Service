"""
TDD RED Phase Tests for HeuristicFilter (Tier 3).

WBS: WBS-AC3 - Heuristic Filter
AC Block: AC-3.1 through AC-3.6

Tests organized by acceptance criteria:
- TestHeuristicFilterResult: Result dataclass tests
- TestWatermarkDetection: AC-3.1
- TestURLFragmentDetection: AC-3.2
- TestFillerWordDetection: AC-3.3
- TestCodeArtifactDetection: AC-3.4
- TestPageMarkerDetection: Additional category
- TestRegexPatternDetection: AC-3.1 (pattern-based)
- TestValidTermPasses: AC-3.5
- TestConfigLoading: AC-3.6
- TestAllNoiseCategories: Comprehensive category coverage
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def noise_config_path() -> Path:
    """Return path to noise_terms.yaml config."""
    return Path(__file__).parent.parent.parent / "config" / "noise_terms.yaml"


@pytest.fixture
def heuristic_filter(noise_config_path: Path) -> "HeuristicFilter":
    """Create HeuristicFilter with real config."""
    from src.classifiers.heuristic_filter import HeuristicFilter

    return HeuristicFilter(config_path=noise_config_path)


@pytest.fixture
def heuristic_filter_default() -> "HeuristicFilter":
    """Create HeuristicFilter with default config path."""
    from src.classifiers.heuristic_filter import HeuristicFilter

    return HeuristicFilter()


@pytest.fixture
def temp_config(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary config file for testing."""
    config_content = """
watermarks:
  - testpublisher
  - fakepdf
url_fragments:
  - www
  - com
generic_filler:
  - using
  - also
code_artifacts:
  - self
  - return
page_markers:
  - chapter
  - section
"""
    config_file = tmp_path / "test_noise_terms.yaml"
    config_file.write_text(config_content)
    yield config_file


# =============================================================================
# TestHeuristicFilterResult: Result dataclass tests
# =============================================================================


class TestHeuristicFilterResult:
    """Test HeuristicFilterResult dataclass structure."""

    def test_result_dataclass_exists(self) -> None:
        """HeuristicFilterResult dataclass should exist."""
        from src.classifiers.heuristic_filter import HeuristicFilterResult

        assert HeuristicFilterResult is not None

    def test_result_has_rejection_reason_field(self) -> None:
        """Result should have rejection_reason field."""
        from src.classifiers.heuristic_filter import HeuristicFilterResult

        result = HeuristicFilterResult(
            rejection_reason="noise_watermarks",
            matched_term="oceanofpdf",
            category="watermarks",
        )
        assert result.rejection_reason == "noise_watermarks"

    def test_result_has_matched_term_field(self) -> None:
        """Result should have matched_term field showing what was matched."""
        from src.classifiers.heuristic_filter import HeuristicFilterResult

        result = HeuristicFilterResult(
            rejection_reason="noise_watermarks",
            matched_term="oceanofpdf",
            category="watermarks",
        )
        assert result.matched_term == "oceanofpdf"

    def test_result_has_category_field(self) -> None:
        """Result should have category field."""
        from src.classifiers.heuristic_filter import HeuristicFilterResult

        result = HeuristicFilterResult(
            rejection_reason="noise_watermarks",
            matched_term="oceanofpdf",
            category="watermarks",
        )
        assert result.category == "watermarks"

    def test_result_is_frozen(self) -> None:
        """Result should be immutable (frozen dataclass)."""
        from src.classifiers.heuristic_filter import HeuristicFilterResult

        result = HeuristicFilterResult(
            rejection_reason="noise_watermarks",
            matched_term="oceanofpdf",
            category="watermarks",
        )
        with pytest.raises(AttributeError):
            result.rejection_reason = "changed"  # type: ignore[misc]


# =============================================================================
# TestWatermarkDetection: AC-3.1
# =============================================================================


class TestWatermarkDetection:
    """Test watermark noise detection (AC-3.1)."""

    def test_detects_oceanofpdf_watermark(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'oceanofpdf' as watermark."""
        result = heuristic_filter.check("oceanofpdf")
        assert result is not None
        assert result.rejection_reason == "noise_watermarks"
        assert result.category == "watermarks"

    def test_detects_ebscohost_watermark(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'ebscohost' as watermark."""
        result = heuristic_filter.check("ebscohost")
        assert result is not None
        assert result.rejection_reason == "noise_watermarks"

    def test_detects_packt_watermark(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'packt' as watermark."""
        result = heuristic_filter.check("packt")
        assert result is not None
        assert result.rejection_reason == "noise_watermarks"

    def test_watermark_case_insensitive(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should detect watermarks case-insensitively."""
        result = heuristic_filter.check("OCEANOFPDF")
        assert result is not None
        assert result.rejection_reason == "noise_watermarks"

    def test_watermark_mixed_case(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should detect watermarks with mixed case."""
        result = heuristic_filter.check("OceanOfPDF")
        assert result is not None
        assert result.rejection_reason == "noise_watermarks"


# =============================================================================
# TestURLFragmentDetection: AC-3.2
# =============================================================================


class TestURLFragmentDetection:
    """Test URL fragment noise detection (AC-3.2)."""

    def test_detects_www_fragment(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'www' as URL fragment."""
        result = heuristic_filter.check("www")
        assert result is not None
        assert result.rejection_reason == "noise_url_fragments"
        assert result.category == "url_fragments"

    def test_detects_http_fragment(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'http' as URL fragment."""
        result = heuristic_filter.check("http")
        assert result is not None
        assert result.rejection_reason == "noise_url_fragments"

    def test_detects_https_fragment(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'https' as URL fragment."""
        result = heuristic_filter.check("https")
        assert result is not None
        assert result.rejection_reason == "noise_url_fragments"

    def test_detects_com_fragment(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'com' as URL fragment."""
        result = heuristic_filter.check("com")
        assert result is not None
        assert result.rejection_reason == "noise_url_fragments"

    def test_detects_html_fragment(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'html' as URL fragment."""
        result = heuristic_filter.check("html")
        assert result is not None
        assert result.rejection_reason == "noise_url_fragments"


# =============================================================================
# TestFillerWordDetection: AC-3.3
# =============================================================================


class TestFillerWordDetection:
    """Test generic filler word detection (AC-3.3)."""

    def test_detects_using_filler(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'using' as generic filler."""
        result = heuristic_filter.check("using")
        assert result is not None
        assert result.rejection_reason == "noise_generic_filler"
        assert result.category == "generic_filler"

    def test_detects_example_filler(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'example' as generic filler."""
        result = heuristic_filter.check("example")
        assert result is not None
        assert result.rejection_reason == "noise_generic_filler"

    def test_detects_also_filler(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'also' as generic filler."""
        result = heuristic_filter.check("also")
        assert result is not None
        assert result.rejection_reason == "noise_generic_filler"

    def test_detects_just_filler(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'just' as generic filler."""
        result = heuristic_filter.check("just")
        assert result is not None
        assert result.rejection_reason == "noise_generic_filler"

    def test_filler_case_insensitive(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should detect filler words case-insensitively."""
        result = heuristic_filter.check("USING")
        assert result is not None
        assert result.rejection_reason == "noise_generic_filler"


# =============================================================================
# TestCodeArtifactDetection: AC-3.4
# =============================================================================


class TestCodeArtifactDetection:
    """Test code artifact detection (AC-3.4)."""

    def test_detects_self_artifact(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'self' as code artifact."""
        result = heuristic_filter.check("self")
        assert result is not None
        assert result.rejection_reason == "noise_code_artifacts"
        assert result.category == "code_artifacts"

    def test_detects_return_artifact(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'return' as code artifact."""
        result = heuristic_filter.check("return")
        assert result is not None
        assert result.rejection_reason == "noise_code_artifacts"

    def test_detects_import_artifact(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'import' as code artifact."""
        result = heuristic_filter.check("import")
        assert result is not None
        assert result.rejection_reason == "noise_code_artifacts"

    def test_detects_lambda_artifact(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'lambda' as code artifact."""
        result = heuristic_filter.check("lambda")
        assert result is not None
        assert result.rejection_reason == "noise_code_artifacts"

    def test_detects_none_artifact(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'none' as code artifact."""
        result = heuristic_filter.check("none")
        assert result is not None
        assert result.rejection_reason == "noise_code_artifacts"


# =============================================================================
# TestPageMarkerDetection: Additional category coverage
# =============================================================================


class TestPageMarkerDetection:
    """Test page marker detection (additional category)."""

    def test_detects_chapter_marker(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'chapter' as page marker."""
        result = heuristic_filter.check("chapter")
        assert result is not None
        assert result.rejection_reason == "noise_page_markers"
        assert result.category == "page_markers"

    def test_detects_section_marker(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'section' as page marker."""
        result = heuristic_filter.check("section")
        assert result is not None
        assert result.rejection_reason == "noise_page_markers"

    def test_detects_figure_marker(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'figure' as page marker."""
        result = heuristic_filter.check("figure")
        assert result is not None
        assert result.rejection_reason == "noise_page_markers"

    def test_detects_appendix_marker(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'appendix' as page marker."""
        result = heuristic_filter.check("appendix")
        assert result is not None
        assert result.rejection_reason == "noise_page_markers"


# =============================================================================
# TestRegexPatternDetection: AC-3.1 (pattern-based)
# =============================================================================


class TestRegexPatternDetection:
    """Test regex-based pattern detection."""

    def test_detects_broken_contraction_ll(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'll' as broken contraction."""
        result = heuristic_filter.check("ll")
        assert result is not None
        assert result.rejection_reason == "noise_broken_contraction"

    def test_detects_broken_contraction_nt(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 'n't' or 'nt' as broken contraction."""
        result = heuristic_filter.check("nt")
        assert result is not None
        assert result.rejection_reason == "noise_broken_contraction"

    def test_detects_broken_contraction_ve(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 've' as broken contraction."""
        result = heuristic_filter.check("ve")
        assert result is not None
        assert result.rejection_reason == "noise_broken_contraction"

    def test_detects_broken_contraction_re(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject 're' as broken contraction."""
        result = heuristic_filter.check("re")
        assert result is not None
        assert result.rejection_reason == "noise_broken_contraction"

    def test_detects_underscore_prefix(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject '_add' as underscore prefix."""
        result = heuristic_filter.check("_add")
        assert result is not None
        assert result.rejection_reason == "noise_underscore_prefix"

    def test_detects_dunder_prefix(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject '__init__' as underscore prefix."""
        result = heuristic_filter.check("__init__")
        assert result is not None
        assert result.rejection_reason == "noise_underscore_prefix"

    def test_detects_single_character(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject single character as too short."""
        result = heuristic_filter.check("a")
        assert result is not None
        assert result.rejection_reason == "noise_single_char"

    def test_detects_pure_number(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject pure numbers."""
        result = heuristic_filter.check("12345")
        assert result is not None
        assert result.rejection_reason == "noise_pure_number"

    def test_detects_pure_number_with_decimals(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should reject decimal numbers."""
        result = heuristic_filter.check("3.14")
        assert result is not None
        assert result.rejection_reason == "noise_pure_number"


# =============================================================================
# TestValidTermPasses: AC-3.5
# =============================================================================


class TestValidTermPasses:
    """Test that valid terms pass through filter (AC-3.5)."""

    def test_valid_term_returns_none(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Valid technical term should return None."""
        result = heuristic_filter.check("microservice")
        assert result is None

    def test_valid_term_api_gateway(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """'api_gateway' should pass through."""
        result = heuristic_filter.check("api_gateway")
        assert result is None

    def test_valid_term_kubernetes(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """'kubernetes' should pass through."""
        result = heuristic_filter.check("kubernetes")
        assert result is None

    def test_valid_term_docker(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """'docker' should pass through."""
        result = heuristic_filter.check("docker")
        assert result is None

    def test_valid_term_machine_learning(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """'machine_learning' should pass through."""
        result = heuristic_filter.check("machine_learning")
        assert result is None

    def test_valid_term_neural_network(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """'neural_network' should pass through."""
        result = heuristic_filter.check("neural_network")
        assert result is None

    def test_valid_alphanumeric_term(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """'python3' (alphanumeric) should pass through."""
        result = heuristic_filter.check("python3")
        assert result is None

    def test_empty_string_rejected(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Empty string should be rejected."""
        result = heuristic_filter.check("")
        assert result is not None
        assert result.rejection_reason == "noise_empty"

    def test_whitespace_only_rejected(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Whitespace-only should be rejected."""
        result = heuristic_filter.check("   ")
        assert result is not None
        assert result.rejection_reason == "noise_empty"


# =============================================================================
# TestConfigLoading: AC-3.6
# =============================================================================


class TestConfigLoading:
    """Test config loading from YAML (AC-3.6)."""

    def test_loads_from_default_path(
        self, heuristic_filter_default: "HeuristicFilter"
    ) -> None:
        """Should load from default config path."""
        # If it loads without error, config loaded successfully
        assert heuristic_filter_default is not None

    def test_loads_from_custom_path(self, temp_config: Path) -> None:
        """Should load from custom config path."""
        from src.classifiers.heuristic_filter import HeuristicFilter

        hf = HeuristicFilter(config_path=temp_config)
        assert hf is not None

    def test_custom_config_uses_custom_terms(self, temp_config: Path) -> None:
        """Custom config should use its own terms."""
        from src.classifiers.heuristic_filter import HeuristicFilter

        hf = HeuristicFilter(config_path=temp_config)
        # testpublisher is in temp config
        result = hf.check("testpublisher")
        assert result is not None
        assert result.rejection_reason == "noise_watermarks"

    def test_missing_config_raises_error(self, tmp_path: Path) -> None:
        """Missing config file should raise ConfigurationError."""
        from src.classifiers.heuristic_filter import (
            HeuristicFilter,
            HeuristicFilterConfigError,
        )

        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(HeuristicFilterConfigError):
            HeuristicFilter(config_path=missing_path)

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Invalid YAML should raise ConfigurationError."""
        from src.classifiers.heuristic_filter import (
            HeuristicFilter,
            HeuristicFilterConfigError,
        )

        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("watermarks: [unclosed bracket")
        with pytest.raises(HeuristicFilterConfigError):
            HeuristicFilter(config_path=invalid_config)

    def test_config_categories_loaded(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should load all expected categories from config."""
        categories = heuristic_filter.get_categories()
        expected = {
            "watermarks",
            "url_fragments",
            "generic_filler",
            "code_artifacts",
            "page_markers",
        }
        assert expected.issubset(set(categories))


# =============================================================================
# TestAllNoiseCategories: Comprehensive category coverage
# =============================================================================


class TestAllNoiseCategories:
    """Test all 8 noise categories are covered."""

    def test_category_count(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Should have at least 5 YAML categories plus 4 regex categories."""
        yaml_categories = heuristic_filter.get_categories()
        assert len(yaml_categories) >= 5

    def test_rejection_reason_format(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Rejection reasons should follow noise_<category> format."""
        result = heuristic_filter.check("oceanofpdf")
        assert result is not None
        assert result.rejection_reason.startswith("noise_")

    def test_all_yaml_categories_have_rejection_reasons(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Each YAML category should map to a rejection reason."""
        test_cases = [
            ("oceanofpdf", "noise_watermarks"),
            ("www", "noise_url_fragments"),
            ("using", "noise_generic_filler"),
            ("self", "noise_code_artifacts"),
            ("chapter", "noise_page_markers"),
        ]
        for term, expected_reason in test_cases:
            result = heuristic_filter.check(term)
            assert result is not None, f"Expected rejection for {term}"
            assert (
                result.rejection_reason == expected_reason
            ), f"Wrong reason for {term}"

    def test_all_regex_categories_have_rejection_reasons(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Each regex pattern should map to a rejection reason."""
        test_cases = [
            ("ll", "noise_broken_contraction"),
            ("_add", "noise_underscore_prefix"),
            ("a", "noise_single_char"),
            ("123", "noise_pure_number"),
            ("", "noise_empty"),
        ]
        for term, expected_reason in test_cases:
            result = heuristic_filter.check(term)
            assert result is not None, f"Expected rejection for '{term}'"
            assert (
                result.rejection_reason == expected_reason
            ), f"Wrong reason for '{term}': {result.rejection_reason}"


# =============================================================================
# TestBatchCheck: Batch processing
# =============================================================================


class TestBatchCheck:
    """Test batch checking of terms."""

    def test_check_batch_returns_dict(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """check_batch should return dict mapping term to result."""
        terms = ["oceanofpdf", "microservice", "www"]
        results = heuristic_filter.check_batch(terms)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_check_batch_mixed_results(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Batch should handle mix of noise and valid terms."""
        terms = ["oceanofpdf", "microservice", "www"]
        results = heuristic_filter.check_batch(terms)
        # oceanofpdf = noise
        assert results["oceanofpdf"] is not None
        # microservice = valid
        assert results["microservice"] is None
        # www = noise
        assert results["www"] is not None

    def test_check_batch_empty_list(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Batch with empty list should return empty dict."""
        results = heuristic_filter.check_batch([])
        assert results == {}

    def test_check_batch_all_valid(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Batch with all valid terms should return all None."""
        terms = ["kubernetes", "docker", "microservice"]
        results = heuristic_filter.check_batch(terms)
        assert all(v is None for v in results.values())

    def test_check_batch_all_noise(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """Batch with all noise terms should return all results."""
        terms = ["oceanofpdf", "www", "self", "chapter"]
        results = heuristic_filter.check_batch(terms)
        assert all(v is not None for v in results.values())


# =============================================================================
# TestHeuristicFilterProtocol: Protocol compliance (AC-8.4)
# =============================================================================


class TestHeuristicFilterProtocol:
    """Test HeuristicFilterProtocol for dependency injection."""

    def test_protocol_exists(self) -> None:
        """HeuristicFilterProtocol should exist."""
        from src.classifiers.heuristic_filter import HeuristicFilterProtocol

        assert HeuristicFilterProtocol is not None

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should be runtime_checkable."""
        from typing import runtime_checkable

        from src.classifiers.heuristic_filter import HeuristicFilterProtocol

        # Protocol should have __runtime_checkable__ marker
        assert hasattr(HeuristicFilterProtocol, "__protocol_attrs__") or hasattr(
            HeuristicFilterProtocol, "_is_runtime_protocol"
        )

    def test_heuristic_filter_passes_protocol(
        self, heuristic_filter: "HeuristicFilter"
    ) -> None:
        """HeuristicFilter should pass Protocol isinstance check."""
        from src.classifiers.heuristic_filter import HeuristicFilterProtocol

        assert isinstance(heuristic_filter, HeuristicFilterProtocol)


# =============================================================================
# TestFakeHeuristicFilter: Test double for AC-8.4
# =============================================================================


class TestFakeHeuristicFilter:
    """Test FakeHeuristicFilter for testing scenarios."""

    def test_fake_exists(self) -> None:
        """FakeHeuristicFilter should exist."""
        from src.classifiers.heuristic_filter import FakeHeuristicFilter

        assert FakeHeuristicFilter is not None

    def test_fake_passes_protocol(self) -> None:
        """FakeHeuristicFilter should pass Protocol check."""
        from src.classifiers.heuristic_filter import (
            FakeHeuristicFilter,
            HeuristicFilterProtocol,
        )

        fake = FakeHeuristicFilter()
        assert isinstance(fake, HeuristicFilterProtocol)

    def test_fake_returns_configured_responses(self) -> None:
        """Fake should return pre-configured responses."""
        from src.classifiers.heuristic_filter import (
            FakeHeuristicFilter,
            HeuristicFilterResult,
        )

        configured_result = HeuristicFilterResult(
            rejection_reason="noise_watermarks",
            matched_term="test",
            category="watermarks",
        )
        fake = FakeHeuristicFilter(responses={"test": configured_result})
        assert fake.check("test") == configured_result

    def test_fake_returns_none_for_unconfigured(self) -> None:
        """Fake should return None for terms not in responses."""
        from src.classifiers.heuristic_filter import FakeHeuristicFilter

        fake = FakeHeuristicFilter(responses={})
        assert fake.check("unknown") is None

    def test_fake_check_batch_works(self) -> None:
        """Fake check_batch should work."""
        from src.classifiers.heuristic_filter import (
            FakeHeuristicFilter,
            HeuristicFilterResult,
        )

        result = HeuristicFilterResult(
            rejection_reason="noise_watermarks",
            matched_term="test",
            category="watermarks",
        )
        fake = FakeHeuristicFilter(responses={"test": result})
        batch_results = fake.check_batch(["test", "other"])
        assert batch_results["test"] == result
        assert batch_results["other"] is None
