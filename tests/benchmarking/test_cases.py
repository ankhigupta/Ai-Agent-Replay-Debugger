"""Tests for the benchmark corpus itself."""

from __future__ import annotations

import unittest
from collections import Counter

from benchmark.models import CaseCategory
from tests.benchmarking._corpus import CORPUS, requires_langgraph


class CorpusSizeTests(unittest.TestCase):
    @requires_langgraph
    def test_a_minimum_category_counts(self) -> None:
        by_category = Counter(c.category for c in CORPUS)
        self.assertGreaterEqual(by_category[CaseCategory.CLEAN], 3)
        self.assertGreaterEqual(by_category[CaseCategory.UNOBSERVABLE], 3)
        self.assertGreaterEqual(by_category[CaseCategory.HELD_OUT], 2)

    def test_a_minimum_check_specific_fault_counts(self) -> None:
        swallowed = [c for c in CORPUS if "swallowed_tool_error" in c.expected_check_findings]
        repeated = [c for c in CORPUS if "repeated_identical_calls" in c.expected_check_findings]
        incomplete = [c for c in CORPUS if "trace_incompleteness" in c.expected_check_findings]
        self.assertGreaterEqual(
            sum(1 for c in swallowed if c.category == CaseCategory.FAULT), 3
        )
        self.assertGreaterEqual(
            sum(1 for c in repeated if c.category == CaseCategory.FAULT), 3
        )
        self.assertGreaterEqual(
            sum(1 for c in incomplete if c.category == CaseCategory.FAULT), 2
        )

    def test_a_minimum_localization_cases(self) -> None:
        localization_cases = [c for c in CORPUS if c.localization_label is not None]
        self.assertGreaterEqual(len(localization_cases), 5)

    def test_a_minimum_self_consistent_wrong_cases(self) -> None:
        tagged = [c for c in CORPUS if "self_consistent_wrong" in c.tags]
        self.assertGreaterEqual(len(tagged), 2)

    def test_a_no_false_positive_controls_exist_per_check(self) -> None:
        controls = [c for c in CORPUS if c.category == CaseCategory.NO_FAULT_CONTROL]
        self.assertGreaterEqual(len(controls), 4)
        checked_ids = {
            check_id
            for c in controls
            for check_id, status in c.expected_check_findings.items()
            if status.value == "NO_FINDING"
        }
        self.assertIn("swallowed_tool_error", checked_ids)
        self.assertIn("repeated_identical_calls", checked_ids)
        self.assertIn("trace_incompleteness", checked_ids)

    def test_b_case_ids_are_unique(self) -> None:
        ids = [c.case_id for c in CORPUS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_cases_build_without_exceptions(self) -> None:
        for case in CORPUS:
            with self.subTest(case=case.case_id):
                trace = case.build_trace()
                self.assertGreater(len(trace.spans) + (0 if trace.spans else 0), -1)  # just: didn't raise
                if case.build_failure_signal is not None:
                    case.build_failure_signal()


class GroundTruthIsolationTests(unittest.TestCase):
    def test_c_runner_source_never_references_ground_truth_fields(self) -> None:
        # Inspect compiled attribute-access names (`co_names`), not raw
        # source text -- the module/function docstrings legitimately *name*
        # these fields in prose to explain that they're never read, which
        # would false-positive a plain substring search. `co_names` only
        # ever contains names actually used for attribute/name access in
        # the compiled bytecode, so it can't be fooled by prose.
        import benchmark.runner as runner_mod

        accessed_names = set(runner_mod.run_case.__code__.co_names) | set(
            runner_mod.run_corpus.__code__.co_names
        )
        for forbidden in ("localization_label", "reference_span_name", "evidence_available"):
            self.assertNotIn(
                forbidden,
                accessed_names,
                msg=f"runner.py must never read BenchmarkCase.{forbidden} (ground truth)",
            )


if __name__ == "__main__":
    unittest.main()
