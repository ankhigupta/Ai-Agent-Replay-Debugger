"""Tests for benchmark reporting, evidence export, and whole-run determinism."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentdebug.core.trace import Trace
from agentdebug.evidence import JsonlEvidenceSink
from benchmark.report import build_summary, render_json_report, render_text_report
from benchmark.runner import run_corpus
from tests.benchmarking._corpus import CORPUS


class ReportOrderingTests(unittest.TestCase):
    def test_r_result_ordering_is_deterministic(self) -> None:
        results_a = run_corpus(CORPUS)
        results_b = run_corpus(CORPUS)
        self.assertEqual([r.case_id for r in results_a], [c.case_id for c in CORPUS])
        self.assertEqual([r.case_id for r in results_a], [r.case_id for r in results_b])


class JsonReportTests(unittest.TestCase):
    def test_s_json_report_is_valid_json(self) -> None:
        results = run_corpus(CORPUS)
        summary = build_summary(CORPUS, results, overhead_iterations=5)
        report = render_json_report(summary)
        serialized = json.dumps(report)  # raises if not JSON-safe
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded["case_count"], len(CORPUS))

    def test_text_report_is_a_non_empty_string_containing_expected_sections(self) -> None:
        results = run_corpus(CORPUS)
        summary = build_summary(CORPUS, results, overhead_iterations=5)
        text = render_text_report(summary)
        for heading in ("Benchmark summary", "Trace checks", "Localization", "Abstention", "Capability coverage", "Runtime"):
            self.assertIn(heading, text)


class EvidenceExportTests(unittest.TestCase):
    def test_t_benchmark_evidence_can_be_reloaded_as_trace_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark_runs.jsonl"
            with JsonlEvidenceSink(path) as sink:
                for case in CORPUS:
                    sink.write_trace(case.build_trace())

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(CORPUS))
            for line in lines:
                restored = Trace.from_dict(json.loads(line))
                self.assertIsInstance(restored, Trace)
                self.assertGreaterEqual(len(restored.spans), 0)


class WholeRunDeterminismTests(unittest.TestCase):
    def test_u_running_benchmark_twice_produces_same_structural_metrics(self) -> None:
        results_a = run_corpus(CORPUS)
        results_b = run_corpus(CORPUS)
        summary_a = build_summary(CORPUS, results_a, overhead_iterations=5)
        summary_b = build_summary(CORPUS, results_b, overhead_iterations=5)

        report_a = render_json_report(summary_a)
        report_b = render_json_report(summary_b)

        # Runtime is inherently variable -- excluded from the equality check.
        report_a.pop("runtime")
        report_b.pop("runtime")

        self.assertEqual(report_a, report_b)


if __name__ == "__main__":
    unittest.main()
