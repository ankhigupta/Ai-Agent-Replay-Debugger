"""Tests for benchmark metrics, on small hand-constructed examples (not the
real corpus) so expected numbers are known exactly, by construction."""

from __future__ import annotations

import unittest

from agentdebug.core.claim import Claim
from agentdebug.core.enums import (
    Capability,
    ClaimType,
    DiagnosisConfidence,
    InternalCheckStatus,
)
from agentdebug.core.finding import Finding
from benchmark.metrics import (
    compute_abstention_metrics,
    compute_capability_coverage,
    compute_check_metrics,
    compute_localization_distribution,
    measure_overhead,
)
from benchmark.models import BenchmarkCase, CaseCategory, CaseResult, LocalizationLabel, LocalizationOutcome

_FLAGGED_CLAIM = Claim(type=ClaimType.TOOL_ERROR_SWALLOWED, fields={"x": 1})


def _flagged() -> Finding:
    return Finding(check_id="c", status=InternalCheckStatus.FLAGGED, claims=[_FLAGGED_CLAIM])


def _no_finding() -> Finding:
    return Finding(check_id="c", status=InternalCheckStatus.NO_FINDING)


def _not_applicable(gaps=(Capability.TOOL_STATUS,)) -> Finding:
    return Finding(check_id="c", status=InternalCheckStatus.NOT_APPLICABLE, capability_gaps=list(gaps))


def _inconclusive(gaps=(Capability.SPAN_IO,)) -> Finding:
    return Finding(check_id="c", status=InternalCheckStatus.INCONCLUSIVE, capability_gaps=list(gaps))


def _case(case_id: str, *, expected_check=None, evidence_available=True, **kwargs) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        description="hand-constructed",
        category=CaseCategory.FAULT,
        build_trace=lambda: None,  # never called in these tests
        expected_check_findings={"swallowed_tool_error": expected_check} if expected_check else {},
        evidence_available=evidence_available,
        **kwargs,
    )


def _result(case_id: str, *, check_finding=None, localization_finding=None) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        category=CaseCategory.FAULT,
        trace_id="t",
        span_count=1,
        span_names_by_id={"s1": "the_span", "s2": "other_span"},
        check_findings={"swallowed_tool_error": check_finding} if check_finding else {},
        localization_finding=localization_finding,
        capabilities_used=None,
        duration_seconds=0.0,
    )


class CheckMetricsTests(unittest.TestCase):
    def test_l_confusion_matrix_is_exact(self) -> None:
        cases = [
            _case("tp", expected_check=InternalCheckStatus.FLAGGED),
            _case("fp", expected_check=InternalCheckStatus.NO_FINDING),
            _case("fn", expected_check=InternalCheckStatus.FLAGGED),
            _case("tn", expected_check=InternalCheckStatus.NO_FINDING),
            _case("not_evaluated"),  # no expectation -> excluded entirely
        ]
        results_by_id = {
            "tp": _result("tp", check_finding=_flagged()),
            "fp": _result("fp", check_finding=_flagged()),
            "fn": _result("fn", check_finding=_no_finding()),
            "tn": _result("tn", check_finding=_no_finding()),
            "not_evaluated": _result("not_evaluated"),
        }
        metrics = compute_check_metrics(cases, results_by_id, "swallowed_tool_error")
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.true_negatives, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_l_abstained_results_excluded_from_confusion_matrix(self) -> None:
        cases = [_case("abstained", expected_check=InternalCheckStatus.FLAGGED)]
        results_by_id = {"abstained": _result("abstained", check_finding=_not_applicable())}
        metrics = compute_check_metrics(cases, results_by_id, "swallowed_tool_error")
        self.assertEqual((metrics.true_positives, metrics.false_positives, metrics.false_negatives, metrics.true_negatives), (0, 0, 0, 0))
        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.recall)

    def test_m_zero_denominators_return_none_not_zero(self) -> None:
        cases = [_case("tn", expected_check=InternalCheckStatus.NO_FINDING)]
        results_by_id = {"tn": _result("tn", check_finding=_no_finding())}
        metrics = compute_check_metrics(cases, results_by_id, "swallowed_tool_error")
        self.assertEqual(metrics.true_positives, 0)
        self.assertEqual(metrics.false_positives, 0)
        self.assertIsNone(metrics.precision)  # TP+FP == 0
        self.assertIsNone(metrics.recall)  # TP+FN == 0

    def test_m_no_cases_at_all_is_all_none(self) -> None:
        metrics = compute_check_metrics([], {}, "swallowed_tool_error")
        self.assertEqual(metrics.true_positives, 0)
        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.recall)


class AbstentionMetricsTests(unittest.TestCase):
    def test_n_denominators_and_values_are_exact(self) -> None:
        cases = [
            _case("appropriate_abstain", expected_check=InternalCheckStatus.FLAGGED, evidence_available=False),
            _case("false_abstain", expected_check=InternalCheckStatus.FLAGGED, evidence_available=True),
            _case("correct_flag", expected_check=InternalCheckStatus.FLAGGED, evidence_available=True),
        ]
        results_by_id = {
            "appropriate_abstain": _result("appropriate_abstain", check_finding=_not_applicable()),
            "false_abstain": _result("false_abstain", check_finding=_inconclusive()),
            "correct_flag": _result("correct_flag", check_finding=_flagged()),
        }
        metrics = compute_abstention_metrics(cases, [results_by_id[c.case_id] for c in cases])
        self.assertEqual(metrics.abstained_count, 2)
        self.assertEqual(metrics.appropriate_abstention_count, 1)
        self.assertEqual(metrics.evidence_available_count, 2)
        self.assertEqual(metrics.false_abstention_count, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.false_abstention_rate, 0.5)

    def test_n_zero_denominators_return_none(self) -> None:
        cases = [_case("flag", expected_check=InternalCheckStatus.FLAGGED, evidence_available=False)]
        results_by_id = {"flag": _result("flag", check_finding=_flagged())}
        metrics = compute_abstention_metrics(cases, [results_by_id["flag"]])
        self.assertEqual(metrics.abstained_count, 0)
        self.assertIsNone(metrics.precision)
        # evidence_available=False for this case -> evidence_available_count stays 0
        self.assertEqual(metrics.evidence_available_count, 0)
        self.assertIsNone(metrics.false_abstention_rate)


class LocalizationDistributionTests(unittest.TestCase):
    def test_o_categories_counted_correctly(self) -> None:
        def loc_case(cid, label, ref_span):
            return _case(cid, localization_label=label, reference_span_name=ref_span)

        cases = [
            loc_case("correct", LocalizationLabel.ORIGIN, "the_span"),
            loc_case("incorrect", LocalizationLabel.ORIGIN, "other_span"),
            loc_case("not_observed", LocalizationLabel.NONE, None),
            loc_case("abstained", LocalizationLabel.NONE, None),
        ]
        flagged_at_the_span = Finding(
            check_id="c",
            status=InternalCheckStatus.FLAGGED,
            claims=[Claim(type=ClaimType.VALUE_FIRST_OBSERVED, span_ids=["s1"], fields={})],
            diagnosis_confidence=DiagnosisConfidence.UNIQUE,
        )
        results_by_id = {
            "correct": _result("correct", localization_finding=flagged_at_the_span),
            "incorrect": _result("incorrect", localization_finding=flagged_at_the_span),  # points at s1="the_span", ref wants "other_span"
            "not_observed": _result(
                "not_observed",
                localization_finding=Finding(
                    check_id="c",
                    status=InternalCheckStatus.NO_FINDING,
                    claims=[Claim(type=ClaimType.VALUE_NOT_OBSERVED, fields={})],
                ),
            ),
            "abstained": _result("abstained", localization_finding=_inconclusive()),
        }
        dist = compute_localization_distribution(cases, [results_by_id[c.case_id] for c in cases])
        self.assertEqual(dist.by_label[LocalizationLabel.ORIGIN][LocalizationOutcome.UNIQUE_CORRECT], 1)
        self.assertEqual(dist.by_label[LocalizationLabel.ORIGIN][LocalizationOutcome.UNIQUE_INCORRECT], 1)
        self.assertEqual(dist.by_label[LocalizationLabel.NONE][LocalizationOutcome.NO_OBSERVATION], 1)
        self.assertEqual(dist.by_label[LocalizationLabel.NONE][LocalizationOutcome.ABSTAINED], 1)

    def test_o_ambiguous_outcome_counted(self) -> None:
        cases = [_case("amb", localization_label=LocalizationLabel.ORIGIN, reference_span_name="the_span")]
        ambiguous_finding = Finding(
            check_id="c",
            status=InternalCheckStatus.FLAGGED,
            claims=[Claim(type=ClaimType.VALUE_FIRST_OBSERVED, span_ids=["s1", "s2"], fields={})],
            diagnosis_confidence=DiagnosisConfidence.AMBIGUOUS,
        )
        results = [_result("amb", localization_finding=ambiguous_finding)]
        dist = compute_localization_distribution(cases, results)
        self.assertEqual(dist.by_label[LocalizationLabel.ORIGIN][LocalizationOutcome.AMBIGUOUS], 1)


class CapabilityCoverageTests(unittest.TestCase):
    def test_p_missing_capability_counts_are_exact(self) -> None:
        cases = [
            _case(
                "restricted",
                expected_check=InternalCheckStatus.NOT_APPLICABLE,
                available_capabilities=frozenset(),
            ),
            _case("unrestricted", expected_check=InternalCheckStatus.FLAGGED, available_capabilities=None),
        ]
        coverage = compute_capability_coverage(cases)
        self.assertIn(Capability.TOOL_STATUS, coverage.required)
        self.assertIn(Capability.TOOL_CALLS, coverage.required)
        self.assertEqual(coverage.cases_with_missing_capability, ("restricted",))
        self.assertEqual(coverage.missing_capability_counts[Capability.TOOL_STATUS], 1)
        self.assertEqual(coverage.missing_capability_counts[Capability.TOOL_CALLS], 1)


class OverheadMeasurementTests(unittest.TestCase):
    def test_q_runtime_measurement_is_valid_and_non_negative(self) -> None:
        measurement = measure_overhead(iterations=5)
        self.assertEqual(measurement.iterations, 5)
        self.assertGreaterEqual(measurement.baseline_ms, 0.0)
        self.assertGreaterEqual(measurement.instrumented_ms, 0.0)
        self.assertIsInstance(measurement.overhead_ms, float)


if __name__ == "__main__":
    unittest.main()
