"""Tests for the benchmark runner."""

from __future__ import annotations

import dataclasses
import json
import unittest

from agentdebug.core.enums import InternalCheckStatus
from benchmark.models import CaseCategory, LocalizationLabel
from benchmark.runner import run_case, run_corpus
from tests.benchmarking._corpus import CORPUS, requires_langgraph

CORPUS_BY_ID = {c.case_id: c for c in CORPUS}


def _result_for(case_id: str):
    return run_case(CORPUS_BY_ID[case_id])


class CleanCaseTests(unittest.TestCase):
    def test_d_clean_cases_produce_no_false_positives(self) -> None:
        for case in CORPUS:
            if case.category != CaseCategory.CLEAN:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                for check_id, finding in result.check_findings.items():
                    self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING, msg=check_id)


class KnownFaultDetectionTests(unittest.TestCase):
    def test_e_known_swallowed_errors_are_detected(self) -> None:
        for case in CORPUS:
            expected = case.expected_check_findings.get("swallowed_tool_error")
            if case.category != CaseCategory.FAULT or expected != InternalCheckStatus.FLAGGED:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                self.assertEqual(
                    result.check_findings["swallowed_tool_error"].status, InternalCheckStatus.FLAGGED
                )

    def test_f_known_repeated_calls_are_detected(self) -> None:
        for case in CORPUS:
            expected = case.expected_check_findings.get("repeated_identical_calls")
            if case.category != CaseCategory.FAULT or expected != InternalCheckStatus.FLAGGED:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                self.assertEqual(
                    result.check_findings["repeated_identical_calls"].status,
                    InternalCheckStatus.FLAGGED,
                )

    def test_g_known_incomplete_traces_are_detected(self) -> None:
        for case in CORPUS:
            expected = case.expected_check_findings.get("trace_incompleteness")
            if case.category != CaseCategory.FAULT or expected != InternalCheckStatus.FLAGGED:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                self.assertEqual(
                    result.check_findings["trace_incompleteness"].status, InternalCheckStatus.FLAGGED
                )

    def test_no_fault_controls_do_not_fire(self) -> None:
        for case in CORPUS:
            if case.category != CaseCategory.NO_FAULT_CONTROL or not case.expected_check_findings:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                for check_id, expected in case.expected_check_findings.items():
                    self.assertEqual(result.check_findings[check_id].status, expected)


class LocalizationBehaviorTests(unittest.TestCase):
    def test_h_localization_cases_match_expected_status_and_confidence(self) -> None:
        for case in CORPUS:
            if case.localization_label is None:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                finding = result.localization_finding
                self.assertIsNotNone(finding)
                if case.expected_localization_status is not None:
                    self.assertEqual(finding.status, case.expected_localization_status)
                if case.expected_localization_confidence is not None:
                    self.assertEqual(finding.diagnosis_confidence, case.expected_localization_confidence)


class AbstentionTests(unittest.TestCase):
    def test_i_unobservable_cases_abstain_appropriately(self) -> None:
        for case in CORPUS:
            if case.category != CaseCategory.UNOBSERVABLE:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                observations = list(result.check_findings.values())
                if result.localization_finding is not None:
                    observations.append(result.localization_finding)
                self.assertTrue(observations, "unobservable case produced no observations at all")
                for finding in observations:
                    self.assertIn(
                        finding.status,
                        (InternalCheckStatus.NOT_APPLICABLE, InternalCheckStatus.INCONCLUSIVE),
                    )
                    # Never silently reinterpreted as a confident NO_FINDING.
                    self.assertNotEqual(finding.status, InternalCheckStatus.NO_FINDING)


class SelfConsistentWrongTests(unittest.TestCase):
    def test_j_self_consistent_wrong_cases_do_not_fabricate_a_diagnosis(self) -> None:
        for case in CORPUS:
            if "self_consistent_wrong" not in case.tags:
                continue
            with self.subTest(case=case.case_id):
                result = run_case(case)
                for check_id, finding in result.check_findings.items():
                    self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING, msg=check_id)
                finding = result.localization_finding
                self.assertIsNotNone(finding)
                serialized = json.dumps(finding.to_dict()).lower()
                for banned in ("caused", "cause", "root_cause", "incorrect", "wrong", "responsible"):
                    self.assertNotIn(banned, serialized)


class HeldOutTests(unittest.TestCase):
    @requires_langgraph
    def test_k_held_out_cases_run_successfully(self) -> None:
        held_out = [c for c in CORPUS if c.category == CaseCategory.HELD_OUT]
        self.assertGreaterEqual(len(held_out), 2)
        for case in held_out:
            with self.subTest(case=case.case_id):
                result = run_case(case)  # must not raise
                self.assertEqual(result.category, CaseCategory.HELD_OUT)


class GroundTruthHasNoEffectTests(unittest.TestCase):
    """Behavioral proof (not just source inspection) that ground-truth
    fields never reach -- and therefore never influence -- an analyzer."""

    def test_mutating_ground_truth_does_not_change_the_observed_result(self) -> None:
        case = CORPUS_BY_ID["localization_span_output_origin"]
        mutated = dataclasses.replace(
            case,
            localization_label=LocalizationLabel.TERMINAL,  # deliberately wrong
            reference_span_name="some_other_span_entirely",  # deliberately wrong
            evidence_available=False,  # deliberately wrong
        )

        original_result = run_case(case)
        mutated_result = run_case(mutated)

        def content(result):
            # Compare structurally, resolving each run's own (freshly
            # generated, non-comparable-by-id) span ids to stable span
            # names first -- see CaseResult.span_names_by_id's docstring.
            finding = result.localization_finding
            d = finding.to_dict()
            d.pop("id")
            d.pop("triggering_signal_id")  # identity field of a freshly-built FailureSignal
            claim = dict(d["claims"][0])
            claim["span_ids"] = [result.span_names_by_id.get(sid) for sid in claim["span_ids"]]
            d["claims"] = [claim]
            return d

        self.assertEqual(content(original_result), content(mutated_result))


class RunCorpusTests(unittest.TestCase):
    def test_run_corpus_preserves_order(self) -> None:
        results = run_corpus(CORPUS)
        self.assertEqual([r.case_id for r in results], [c.case_id for c in CORPUS])


if __name__ == "__main__":
    unittest.main()
