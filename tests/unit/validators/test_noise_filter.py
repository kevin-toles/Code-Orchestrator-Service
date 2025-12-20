"""Unit tests for NoiseFilter - WBS-1.2.

Tests noise filtering for metadata extraction per AC-2.4.
TDD Phase: RED - All tests should FAIL initially.

Noise Categories (from CME_ARCHITECTURE.md Section 5.1):
- OCR Watermarks: oceanofpdf, ebscohost, packt
- Broken Contractions: 'll, n't, 's, 've
- URL Fragments: www, http, https, com
- Generic Filler: using, used, one, new, first
- Code Artifacts: _add, __init__, self, cls
- Page Markers: chapter 1, page 42, figure 3.1
- Single Characters: a, b, c, n, o
- Pure Numbers: 123, 45, 2025
"""

import pytest
from src.validators.noise_filter import (
    NoiseFilter,
    FilterResult,
    NOISE_REASON_WATERMARK,
    NOISE_REASON_CONTRACTION,
    NOISE_REASON_URL_FRAGMENT,
    NOISE_REASON_GENERIC_FILLER,
    NOISE_REASON_CODE_ARTIFACT,
    NOISE_REASON_PAGE_MARKER,
    NOISE_REASON_SINGLE_CHAR,
    NOISE_REASON_PURE_NUMBER,
)


# === WBS-1.2.1-1.2.2: Watermark Detection ===

class TestWatermarkDetection:
    """Tests for OCR watermark filtering (AC-2.4)."""

    def test_detects_oceanofpdf_watermark(self) -> None:
        """Watermark 'oceanofpdf' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("oceanofpdf")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_WATERMARK

    def test_detects_ebscohost_watermark(self) -> None:
        """Watermark 'ebscohost' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("ebscohost")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_WATERMARK

    def test_detects_packt_watermark(self) -> None:
        """Watermark 'packt' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("packt")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_WATERMARK

    def test_watermark_case_insensitive(self) -> None:
        """Watermark detection should be case-insensitive."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("OceanOfPDF")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_WATERMARK


# === WBS-1.2.3-1.2.4: Broken Contraction Detection ===

class TestBrokenContractionDetection:
    """Tests for broken contraction filtering (AC-2.4)."""

    def test_detects_broken_ll_contraction(self) -> None:
        """Broken contraction \"'ll\" should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("'ll")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CONTRACTION

    def test_detects_broken_nt_contraction(self) -> None:
        """Broken contraction \"n't\" should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("n't")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CONTRACTION

    def test_detects_broken_s_contraction(self) -> None:
        """Broken contraction \"'s\" should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("'s")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CONTRACTION

    def test_detects_broken_ve_contraction(self) -> None:
        """Broken contraction \"'ve\" should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("'ve")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CONTRACTION


# === WBS-1.2.5-1.2.6: URL Fragment Detection ===

class TestURLFragmentDetection:
    """Tests for URL fragment filtering (AC-2.4)."""

    def test_detects_www_fragment(self) -> None:
        """URL fragment 'www' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("www")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_URL_FRAGMENT

    def test_detects_http_fragment(self) -> None:
        """URL fragment 'http' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("http")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_URL_FRAGMENT

    def test_detects_https_fragment(self) -> None:
        """URL fragment 'https' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("https")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_URL_FRAGMENT

    def test_detects_com_fragment(self) -> None:
        """URL fragment 'com' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("com")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_URL_FRAGMENT


# === WBS-1.2.7-1.2.8: Generic Filler Detection ===

class TestGenericFillerDetection:
    """Tests for generic filler filtering (AC-2.4)."""

    def test_detects_using_filler(self) -> None:
        """Filler word 'using' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("using")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_GENERIC_FILLER

    def test_detects_used_filler(self) -> None:
        """Filler word 'used' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("used")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_GENERIC_FILLER

    def test_detects_one_filler(self) -> None:
        """Filler word 'one' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("one")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_GENERIC_FILLER


# === WBS-1.2.9-1.2.10: Code Artifact Detection ===

class TestCodeArtifactDetection:
    """Tests for code artifact filtering (AC-2.4)."""

    def test_detects_dunder_init(self) -> None:
        """Code artifact '__init__' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("__init__")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CODE_ARTIFACT

    def test_detects_self_keyword(self) -> None:
        """Code artifact 'self' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("self")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CODE_ARTIFACT

    def test_detects_cls_keyword(self) -> None:
        """Code artifact 'cls' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("cls")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CODE_ARTIFACT

    def test_detects_underscore_prefix(self) -> None:
        """Code artifact '_add' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("_add")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_CODE_ARTIFACT


# === Additional Pattern Tests ===

class TestPageMarkerDetection:
    """Tests for page marker filtering (AC-2.4)."""

    def test_detects_chapter_marker(self) -> None:
        """Page marker 'chapter' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("chapter")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_PAGE_MARKER

    def test_detects_figure_marker(self) -> None:
        """Page marker 'figure' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("figure")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_PAGE_MARKER


class TestSingleCharDetection:
    """Tests for single character filtering (AC-2.4)."""

    def test_rejects_single_char_a(self) -> None:
        """Single character 'a' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("a")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_SINGLE_CHAR

    def test_rejects_single_char_x(self) -> None:
        """Single character 'x' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("x")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_SINGLE_CHAR


class TestPureNumberDetection:
    """Tests for pure number filtering (AC-2.4)."""

    def test_rejects_pure_number(self) -> None:
        """Pure number '123' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("123")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_PURE_NUMBER

    def test_rejects_year_number(self) -> None:
        """Year '2025' should be rejected."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("2025")
        assert result.is_rejected is True
        assert result.reason == NOISE_REASON_PURE_NUMBER


# === WBS-1.2.11-1.2.12: Rejection Tracking ===

class TestRejectionTracking:
    """Tests for rejection reasons dictionary (AC-2.4)."""

    def test_filter_batch_returns_accepted_and_rejected(self) -> None:
        """filter_batch() should return both accepted and rejected terms."""
        noise_filter = NoiseFilter()
        terms = ["microservices", "oceanofpdf", "'ll", "api", "www"]
        result = noise_filter.filter_batch(terms)

        assert "microservices" in result.accepted
        assert "api" in result.accepted
        assert "oceanofpdf" in result.rejected
        assert "'ll" in result.rejected
        assert "www" in result.rejected

    def test_filter_batch_provides_rejection_reasons(self) -> None:
        """filter_batch() should provide reasons for each rejection."""
        noise_filter = NoiseFilter()
        terms = ["oceanofpdf", "'ll", "www"]
        result = noise_filter.filter_batch(terms)

        assert result.rejection_reasons["oceanofpdf"] == NOISE_REASON_WATERMARK
        assert result.rejection_reasons["'ll"] == NOISE_REASON_CONTRACTION
        assert result.rejection_reasons["www"] == NOISE_REASON_URL_FRAGMENT

    def test_filter_batch_empty_input(self) -> None:
        """filter_batch() with empty list should return empty results."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_batch([])

        assert result.accepted == []
        assert result.rejected == []
        assert result.rejection_reasons == {}


# === Valid Terms (Should NOT be rejected) ===

class TestValidTermsAccepted:
    """Tests that valid technical terms are NOT rejected."""

    def test_accepts_microservices(self) -> None:
        """Technical term 'microservices' should be accepted."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("microservices")
        assert result.is_rejected is False
        assert result.reason is None

    def test_accepts_api(self) -> None:
        """Technical term 'api' should be accepted."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("api")
        assert result.is_rejected is False
        assert result.reason is None

    def test_accepts_kubernetes(self) -> None:
        """Technical term 'kubernetes' should be accepted."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("kubernetes")
        assert result.is_rejected is False
        assert result.reason is None

    def test_accepts_architecture(self) -> None:
        """Technical term 'architecture' should be accepted."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("architecture")
        assert result.is_rejected is False
        assert result.reason is None

    def test_accepts_machine_learning(self) -> None:
        """Technical term 'machine learning' should be accepted."""
        noise_filter = NoiseFilter()
        result = noise_filter.filter_term("machine learning")
        assert result.is_rejected is False
        assert result.reason is None
