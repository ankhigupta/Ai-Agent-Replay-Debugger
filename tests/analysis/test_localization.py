"""Tests for the localization vocabulary: LocalizationTarget, ValueType,
values_match, and the LocalizationAnalyzer base contract."""

from __future__ import annotations

import datetime
import inspect
import unittest

from agentdebug.analysis.localization import LocalizationAnalyzer, LocalizationTarget, ValueType
from agentdebug.analysis.localization.base import values_match
from agentdebug.analysis.localization.first_observed import FirstObservedOccurrenceAnalyzer
from agentdebug.core.enums import ExternalStatus
from agentdebug.core.failure_signal import FailureSignal


class LocalizationTargetConstructionTests(unittest.TestCase):
    def test_defaults(self) -> None:
        target = LocalizationTarget(value=42)
        self.assertEqual(target.value, 42)
        self.assertIsNone(target.field_path)
        self.assertIsNone(target.value_type)

    def test_from_failure_signal_uses_actual_only(self) -> None:
        signal = FailureSignal(
            source="pytest",
            verdict=ExternalStatus.FAILED,
            expected=True,
            actual=False,
            detail="refund_eligible should have been True but was False",
        )
        target = LocalizationTarget.from_failure_signal(signal)
        self.assertEqual(target.value, False)
        self.assertIsNone(target.field_path)

    def test_from_failure_signal_does_not_use_expected_or_detail(self) -> None:
        signal = FailureSignal(
            source="pytest", verdict=ExternalStatus.FAILED, expected="EXPECTED", actual="ACTUAL", detail="some text"
        )
        target = LocalizationTarget.from_failure_signal(signal)
        self.assertEqual(target.value, "ACTUAL")


class FieldPathExtractionTests(unittest.TestCase):
    def test_no_field_path_returns_whole_candidate(self) -> None:
        target = LocalizationTarget(value={"a": 1})
        found, value = target.extract({"a": 1, "b": 2})
        self.assertTrue(found)
        self.assertEqual(value, {"a": 1, "b": 2})

    def test_simple_field_path(self) -> None:
        target = LocalizationTarget(value=False, field_path="refund_eligible")
        found, value = target.extract({"refund_eligible": False})
        self.assertTrue(found)
        self.assertEqual(value, False)

    def test_nested_field_path(self) -> None:
        target = LocalizationTarget(value="approved", field_path="refund.status")
        found, value = target.extract({"refund": {"status": "approved"}})
        self.assertTrue(found)
        self.assertEqual(value, "approved")

    def test_missing_key_is_not_found(self) -> None:
        target = LocalizationTarget(value=1, field_path="missing")
        found, _ = target.extract({"present": 1})
        self.assertFalse(found)

    def test_non_dict_intermediate_is_not_found(self) -> None:
        target = LocalizationTarget(value=1, field_path="refund.status")
        found, _ = target.extract({"refund": "not-a-dict"})
        self.assertFalse(found)

    def test_non_dict_candidate_without_field_path_is_whole_match_candidate(self) -> None:
        target = LocalizationTarget(value=False)
        found, value = target.extract(False)
        self.assertTrue(found)
        self.assertEqual(value, False)


class StructuralEqualityTests(unittest.TestCase):
    def test_scalar_match(self) -> None:
        self.assertTrue(values_match(False, False, None))
        self.assertFalse(values_match(False, True, None))

    def test_dict_key_order_irrelevant(self) -> None:
        a = {"eligible": True, "amount": 500}
        b = {"amount": 500, "eligible": True}
        self.assertTrue(values_match(a, b, None))

    def test_list_order_is_significant(self) -> None:
        self.assertFalse(values_match([1, 2, 3], [3, 2, 1], None))
        self.assertTrue(values_match([1, 2, 3], [1, 2, 3], None))

    def test_nested_structures(self) -> None:
        a = {"orders": [{"id": "ORD-1", "eligible": True}]}
        b = {"orders": [{"eligible": True, "id": "ORD-1"}]}
        self.assertTrue(values_match(a, b, None))

    def test_bool_and_int_do_not_coincide(self) -> None:
        self.assertFalse(values_match(True, 1, None))
        self.assertFalse(values_match(False, 0, None))

    def test_int_and_float_do_not_coincide(self) -> None:
        self.assertFalse(values_match(1, 1.0, None))

    def test_different_dict_keys_do_not_match(self) -> None:
        self.assertFalse(values_match({"a": 1}, {"a": 1, "b": 2}, None))


class MonetaryNormalizationTests(unittest.TestCase):
    def test_dollar_string_matches_plain_number_when_typed(self) -> None:
        self.assertTrue(values_match("$100", 100, ValueType.MONETARY))

    def test_without_explicit_type_dollar_string_does_not_match_number(self) -> None:
        self.assertFalse(values_match("$100", 100, None))

    def test_thousands_separator_normalized(self) -> None:
        self.assertTrue(values_match("$1,200.50", 1200.50, ValueType.MONETARY))

    def test_unparseable_monetary_string_does_not_match(self) -> None:
        self.assertFalse(values_match("not-a-number", 100, ValueType.MONETARY))


class TemporalNormalizationTests(unittest.TestCase):
    def test_iso_string_matches_date_object_when_typed(self) -> None:
        self.assertTrue(values_match("2026-01-01", datetime.date(2026, 1, 1), ValueType.TEMPORAL))

    def test_without_explicit_type_date_string_does_not_match_arbitrary_string(self) -> None:
        self.assertFalse(values_match("2026-01-01", "2026-01-01-ish", None))

    def test_mismatched_dates_do_not_match(self) -> None:
        self.assertFalse(values_match("2026-01-01", "2026-01-02", ValueType.TEMPORAL))

    def test_non_date_string_does_not_match_when_typed_temporal(self) -> None:
        self.assertFalse(values_match("not-a-date", "also-not-a-date", ValueType.TEMPORAL))


class IdentifierNormalizationTests(unittest.TestCase):
    def test_int_identifier_matches_string_form_when_typed(self) -> None:
        self.assertTrue(values_match(123, "123", ValueType.IDENTIFIER))

    def test_case_sensitive(self) -> None:
        self.assertFalse(values_match("User-123", "user-123", ValueType.IDENTIFIER))

    def test_without_explicit_type_int_and_string_do_not_match(self) -> None:
        self.assertFalse(values_match(123, "123", None))


class LocalizationAnalyzerInterfaceTests(unittest.TestCase):
    def test_cannot_instantiate_abstract_base(self) -> None:
        with self.assertRaises(TypeError):
            LocalizationAnalyzer()  # type: ignore[abstract]

    def test_first_observed_occurrence_is_a_localization_analyzer(self) -> None:
        self.assertIsInstance(FirstObservedOccurrenceAnalyzer(), LocalizationAnalyzer)

    def test_localization_analyzer_is_not_a_trace_check(self) -> None:
        from agentdebug.analysis.check import TraceCheck

        self.assertFalse(issubclass(LocalizationAnalyzer, TraceCheck))

    def test_run_requires_a_failure_signal_argument(self) -> None:
        signature = inspect.signature(LocalizationAnalyzer.run)
        self.assertIn("failure_signal", signature.parameters)
        self.assertEqual(signature.parameters["failure_signal"].default, inspect.Parameter.empty)


class NoForbiddenDependenciesTests(unittest.TestCase):
    def test_no_langgraph_import_in_localization_source(self) -> None:
        import agentdebug.analysis.localization.base as base_mod
        import agentdebug.analysis.localization.first_observed as fo_mod

        for mod in (base_mod, fo_mod):
            source = inspect.getsource(mod).lower()
            self.assertNotIn("langgraph", source)

    def test_no_llm_or_network_client_in_localization_source(self) -> None:
        import agentdebug.analysis.localization.base as base_mod
        import agentdebug.analysis.localization.first_observed as fo_mod

        banned = ("openai", "anthropic", "requests", "httpx", "urllib", "socket")
        for mod in (base_mod, fo_mod):
            source = inspect.getsource(mod).lower()
            for term in banned:
                self.assertNotIn(term, source)


if __name__ == "__main__":
    unittest.main()
