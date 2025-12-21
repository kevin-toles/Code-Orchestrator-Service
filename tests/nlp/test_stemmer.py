"""Tests for Stemmer module - HCE-3.0 Morphological Deduplication.

TDD RED Phase: HCE-3.1 through HCE-3.10, HCE-3.15 through HCE-3.18.

AC Reference:
- AC-3.1: Stemmer module exists and imports
- AC-3.2: Suffix stripping works (ation, ity, ing, ies, es, s)
- AC-3.3: Protected terms preserved (microservices, kubernetes, APIs)
- AC-3.4: deduplicate_by_stem() keeps first occurrence, returns count
"""

from __future__ import annotations

import pytest


# =============================================================================
# HCE-3.1: Test Stemmer can be imported (AC-3.1)
# =============================================================================


class TestStemmerImport:
    """HCE-3.1: Test Stemmer module can be imported."""

    def test_stemmer_module_imports(self) -> None:
        """AC-3.1: Stemmer module can be imported."""
        from src.nlp import stemmer

        assert stemmer is not None

    def test_get_word_stem_function_exists(self) -> None:
        """AC-3.1: get_word_stem function exists."""
        from src.nlp.stemmer import get_word_stem

        assert callable(get_word_stem)

    def test_deduplicate_by_stem_function_exists(self) -> None:
        """AC-3.1: deduplicate_by_stem function exists."""
        from src.nlp.stemmer import deduplicate_by_stem

        assert callable(deduplicate_by_stem)


# =============================================================================
# HCE-3.2: Test SUFFIX_RULES constant exists (AC-3.1)
# =============================================================================


class TestSuffixRulesConstant:
    """HCE-3.2: Test SUFFIX_RULES constant exists."""

    def test_suffix_rules_constant_exists(self) -> None:
        """AC-3.1: SUFFIX_RULES constant exists."""
        from src.nlp.stemmer import SUFFIX_RULES

        assert SUFFIX_RULES is not None

    def test_suffix_rules_is_list_or_tuple(self) -> None:
        """AC-3.1: SUFFIX_RULES is a sequence of rules."""
        from src.nlp.stemmer import SUFFIX_RULES

        assert isinstance(SUFFIX_RULES, (list, tuple))

    def test_suffix_rules_has_ation_rule(self) -> None:
        """AC-3.2: SUFFIX_RULES includes 'ation' suffix."""
        from src.nlp.stemmer import SUFFIX_RULES

        suffixes = [rule[0] for rule in SUFFIX_RULES]
        assert "ation" in suffixes

    def test_suffix_rules_has_ity_rule(self) -> None:
        """AC-3.2: SUFFIX_RULES includes 'ity' suffix."""
        from src.nlp.stemmer import SUFFIX_RULES

        suffixes = [rule[0] for rule in SUFFIX_RULES]
        assert "ity" in suffixes

    def test_suffix_rules_has_ing_rule(self) -> None:
        """AC-3.2: SUFFIX_RULES includes 'ing' suffix."""
        from src.nlp.stemmer import SUFFIX_RULES

        suffixes = [rule[0] for rule in SUFFIX_RULES]
        assert "ing" in suffixes


# =============================================================================
# HCE-3.3: Test PROTECTED_TERMS constant exists (AC-3.1)
# =============================================================================


class TestProtectedTermsConstant:
    """HCE-3.3: Test PROTECTED_TERMS constant exists."""

    def test_protected_terms_constant_exists(self) -> None:
        """AC-3.1: PROTECTED_TERMS constant exists."""
        from src.nlp.stemmer import PROTECTED_TERMS

        assert PROTECTED_TERMS is not None

    def test_protected_terms_is_set_or_frozenset(self) -> None:
        """AC-3.3: PROTECTED_TERMS is a set for O(1) lookup."""
        from src.nlp.stemmer import PROTECTED_TERMS

        assert isinstance(PROTECTED_TERMS, (set, frozenset))

    def test_protected_terms_contains_microservices(self) -> None:
        """AC-3.3: PROTECTED_TERMS includes 'microservices'."""
        from src.nlp.stemmer import PROTECTED_TERMS

        # Check case-insensitive
        lower_terms = {t.lower() for t in PROTECTED_TERMS}
        assert "microservices" in lower_terms

    def test_protected_terms_contains_kubernetes(self) -> None:
        """AC-3.3: PROTECTED_TERMS includes 'kubernetes'."""
        from src.nlp.stemmer import PROTECTED_TERMS

        lower_terms = {t.lower() for t in PROTECTED_TERMS}
        assert "kubernetes" in lower_terms

    def test_protected_terms_contains_apis(self) -> None:
        """AC-3.3: PROTECTED_TERMS includes 'APIs'."""
        from src.nlp.stemmer import PROTECTED_TERMS

        lower_terms = {t.lower() for t in PROTECTED_TERMS}
        assert "apis" in lower_terms


# =============================================================================
# HCE-3.4: Test get_word_stem() strips "ation" (AC-3.2)
# =============================================================================


class TestStripAtionSuffix:
    """HCE-3.4: Test get_word_stem() strips 'ation' suffix."""

    def test_strips_ation_from_implementation(self) -> None:
        """AC-3.2: 'implementation' → 'implement'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("implementation")
        assert result == "implement"

    def test_strips_ation_from_configuration(self) -> None:
        """AC-3.2: 'configuration' → 'configur'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("configuration")
        assert result == "configur"

    def test_strips_ation_from_documentation(self) -> None:
        """AC-3.2: 'documentation' → 'document'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("documentation")
        assert result == "document"


# =============================================================================
# HCE-3.5: Test get_word_stem() strips "ity" (AC-3.2)
# =============================================================================


class TestStripItySuffix:
    """HCE-3.5: Test get_word_stem() strips 'ity' suffix."""

    def test_strips_ity_from_complexity(self) -> None:
        """AC-3.2: 'complexity' → 'complex'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("complexity")
        assert result == "complex"

    def test_strips_ity_from_simplicity(self) -> None:
        """AC-3.2: 'simplicity' → 'simplic'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("simplicity")
        assert result == "simplic"

    def test_strips_ity_from_scalability(self) -> None:
        """AC-3.2: 'scalability' → 'scalabil'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("scalability")
        assert result == "scalabil"


# =============================================================================
# HCE-3.6: Test get_word_stem() strips "ing" (AC-3.2)
# =============================================================================


class TestStripIngSuffix:
    """HCE-3.6: Test get_word_stem() strips 'ing' suffix."""

    def test_strips_ing_from_processing(self) -> None:
        """AC-3.2: 'processing' → 'process'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("processing")
        assert result == "process"

    def test_strips_ing_from_computing(self) -> None:
        """AC-3.2: 'computing' → 'comput'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("computing")
        assert result == "comput"

    def test_strips_ing_from_learning(self) -> None:
        """AC-3.2: 'learning' → 'learn'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("learning")
        assert result == "learn"


# =============================================================================
# HCE-3.7: Test get_word_stem() strips "ies" → "y" (AC-3.2)
# =============================================================================


class TestStripIesSuffix:
    """HCE-3.7: Test get_word_stem() strips 'ies' → 'y'."""

    def test_strips_ies_from_dependencies(self) -> None:
        """AC-3.2: 'dependencies' → 'dependency'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("dependencies")
        assert result == "dependency"

    def test_strips_ies_from_libraries(self) -> None:
        """AC-3.2: 'libraries' → 'library'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("libraries")
        assert result == "library"

    def test_strips_ies_from_categories(self) -> None:
        """AC-3.2: 'categories' → 'category'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("categories")
        assert result == "category"


# =============================================================================
# HCE-3.8: Test get_word_stem() strips "s" (AC-3.2)
# =============================================================================


class TestStripSSuffix:
    """HCE-3.8: Test get_word_stem() strips 's' suffix."""

    def test_strips_s_from_models(self) -> None:
        """AC-3.2: 'models' → 'model'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("models")
        assert result == "model"

    def test_strips_s_from_systems(self) -> None:
        """AC-3.2: 'systems' → 'system'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("systems")
        assert result == "system"

    def test_strips_s_from_patterns(self) -> None:
        """AC-3.2: 'patterns' → 'pattern'."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("patterns")
        assert result == "pattern"

    def test_strips_es_from_services(self) -> None:
        """AC-3.2: 'services' → 'servic' (es suffix)."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("services")
        assert result == "servic"


# =============================================================================
# HCE-3.9: Test get_word_stem() respects min length (AC-3.2)
# =============================================================================


class TestMinStemLength:
    """HCE-3.9: Test get_word_stem() respects minimum stem length."""

    def test_does_not_strip_if_stem_too_short(self) -> None:
        """AC-3.2: Short words unchanged if stem would be too short."""
        from src.nlp.stemmer import get_word_stem

        # "is" should not strip 's' leaving just "i"
        result = get_word_stem("is")
        assert result == "is"

    def test_does_not_strip_short_word_bus(self) -> None:
        """AC-3.2: 'bus' unchanged (stem would be 'bu')."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("bus")
        assert result == "bus"

    def test_preserves_word_api(self) -> None:
        """AC-3.2: 'api' unchanged (too short)."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("api")
        assert result == "api"

    def test_empty_string_returns_empty(self) -> None:
        """AC-3.2: Empty string returns empty."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("")
        assert result == ""


# =============================================================================
# HCE-3.10: Test get_word_stem() protects technical terms (AC-3.3)
# =============================================================================


class TestProtectedTermsStemming:
    """HCE-3.10: Test get_word_stem() protects technical terms."""

    def test_microservices_preserved(self) -> None:
        """AC-3.3: 'microservices' unchanged despite 's' suffix."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("microservices")
        assert result == "microservices"

    def test_kubernetes_preserved(self) -> None:
        """AC-3.3: 'kubernetes' unchanged despite 's' suffix."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("kubernetes")
        assert result == "kubernetes"

    def test_apis_preserved(self) -> None:
        """AC-3.3: 'APIs' unchanged despite 's' suffix."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("APIs")
        assert result == "APIs"

    def test_graphql_preserved(self) -> None:
        """AC-3.3: 'GraphQL' unchanged."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("GraphQL")
        assert result == "GraphQL"

    def test_rest_preserved(self) -> None:
        """AC-3.3: 'REST' unchanged."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("REST")
        assert result == "REST"

    def test_protected_term_case_insensitive(self) -> None:
        """AC-3.3: Protection is case-insensitive."""
        from src.nlp.stemmer import get_word_stem

        result = get_word_stem("MICROSERVICES")
        assert result == "MICROSERVICES"


# =============================================================================
# HCE-3.15: Test deduplicate_by_stem() groups by stem (AC-3.4)
# =============================================================================


class TestDeduplicateByStemGrouping:
    """HCE-3.15: Test deduplicate_by_stem() groups by stem."""

    def test_groups_model_and_models(self) -> None:
        """AC-3.4: 'model' and 'models' grouped by stem."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["model", "models"]
        result, _ = deduplicate_by_stem(terms)
        assert len(result) == 1

    def test_groups_complex_and_complexity(self) -> None:
        """AC-3.4: 'complex' and 'complexity' grouped by stem."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["complex", "complexity"]
        result, _ = deduplicate_by_stem(terms)
        assert len(result) == 1

    def test_different_stems_not_grouped(self) -> None:
        """AC-3.4: Different stems remain separate."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["model", "pattern", "system"]
        result, _ = deduplicate_by_stem(terms)
        assert len(result) == 3


# =============================================================================
# HCE-3.16: Test deduplicate_by_stem() keeps first occurrence (AC-3.4)
# =============================================================================


class TestDeduplicateByStemFirstOccurrence:
    """HCE-3.16: Test deduplicate_by_stem() keeps first occurrence."""

    def test_keeps_first_occurrence_model(self) -> None:
        """AC-3.4: 'model' kept over 'models' (first occurrence)."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["model", "models", "modeling"]
        result, _ = deduplicate_by_stem(terms)
        assert result[0] == "model"

    def test_keeps_first_occurrence_complexity(self) -> None:
        """AC-3.4: 'complexity' kept over 'complex' when first."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["complexity", "complex"]
        result, _ = deduplicate_by_stem(terms)
        assert result[0] == "complexity"

    def test_order_preserved_for_unique_stems(self) -> None:
        """AC-3.4: Order preserved for terms with unique stems."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["alpha", "beta", "gamma"]
        result, _ = deduplicate_by_stem(terms)
        assert result == ["alpha", "beta", "gamma"]


# =============================================================================
# HCE-3.17: Test deduplicate_by_stem() handles ExtractedTerm (AC-3.4)
# =============================================================================


class TestDeduplicateByStemExtractedTerm:
    """HCE-3.17: Test deduplicate_by_stem() handles ExtractedTerm."""

    def test_handles_extracted_term_objects(self) -> None:
        """AC-3.4: Works with ExtractedTerm objects."""
        from src.nlp.ensemble_merger import ExtractedTerm
        from src.nlp.stemmer import deduplicate_by_stem

        terms = [
            ExtractedTerm(term="model", score=0.1, source="yake"),
            ExtractedTerm(term="models", score=0.2, source="textrank"),
        ]
        result, _ = deduplicate_by_stem(terms)
        assert len(result) == 1
        assert result[0].term == "model"

    def test_preserves_extracted_term_attributes(self) -> None:
        """AC-3.4: ExtractedTerm attributes preserved."""
        from src.nlp.ensemble_merger import ExtractedTerm
        from src.nlp.stemmer import deduplicate_by_stem

        terms = [
            ExtractedTerm(term="complexity", score=0.05, source="yake"),
            ExtractedTerm(term="complex", score=0.5, source="textrank"),
        ]
        result, _ = deduplicate_by_stem(terms)
        assert result[0].score == 0.05
        assert result[0].source == "yake"

    def test_mixed_list_rejected_or_handled(self) -> None:
        """AC-3.4: Mixed list of strings and ExtractedTerm works."""
        from src.nlp.stemmer import deduplicate_by_stem

        # Should handle uniform lists - test with strings
        terms = ["model", "models"]
        result, _ = deduplicate_by_stem(terms)
        assert len(result) == 1


# =============================================================================
# HCE-3.18: Test deduplicate_by_stem() returns removed count (AC-3.4)
# =============================================================================


class TestDeduplicateByStemRemovedCount:
    """HCE-3.18: Test deduplicate_by_stem() returns removed count."""

    def test_returns_tuple_with_count(self) -> None:
        """AC-3.4: Returns (result, removed_count) tuple."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["model", "models"]
        result = deduplicate_by_stem(terms)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_removed_count_correct_for_duplicates(self) -> None:
        """AC-3.4: Removed count accurate for duplicates."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["model", "models", "modeling"]
        _, removed_count = deduplicate_by_stem(terms)
        assert removed_count == 2

    def test_removed_count_zero_for_unique_stems(self) -> None:
        """AC-3.4: Removed count is 0 when no duplicates."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms = ["model", "pattern", "system"]
        _, removed_count = deduplicate_by_stem(terms)
        assert removed_count == 0

    def test_empty_list_returns_zero_count(self) -> None:
        """AC-3.4: Empty list returns 0 removed count."""
        from src.nlp.stemmer import deduplicate_by_stem

        terms: list[str] = []
        result, removed_count = deduplicate_by_stem(terms)
        assert result == []
        assert removed_count == 0
