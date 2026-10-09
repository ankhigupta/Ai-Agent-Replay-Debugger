"""Tests for how the benchmark corpus handles the optional `langgraph` extra.

The full corpus (FULL_CORPUS_SIZE cases) includes cases captured through the
real LangGraph integration. These tests pin the explicit-dependency design:
the default corpus is never silently shrunk; a missing LangGraph is an error
unless the core-only subset is requested by name.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from benchmark import cases
from benchmark.cases import (
    CORE_CORPUS_SIZE,
    FULL_CORPUS_SIZE,
    LANGGRAPH_TAG,
    BenchmarkDependencyError,
    build_corpus,
)
from benchmark.models import CaseCategory
from benchmark.runner import run_corpus
from tests.benchmarking._corpus import requires_langgraph

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LANGGRAPH_CASE_IDS = {
    "unobservable_langgraph_no_checkpointer",
    "held_out_fan_out_inconsistency",
}

_RUN_BENCHMARK_WITHOUT_LANGGRAPH = (
    "import sys, runpy\n"
    "sys.modules['langgraph'] = None\n"
    "sys.argv = ['benchmark'] + sys.argv[1:]\n"
    "runpy.run_module('benchmark', run_name='__main__')\n"
)


def _run_without_langgraph(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _RUN_BENCHMARK_WITHOUT_LANGGRAPH, *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


class CoreOnlySubsetTests(unittest.TestCase):
    """Run whether or not LangGraph is installed."""

    def test_core_only_subset_excludes_exactly_the_langgraph_cases(self) -> None:
        core = build_corpus(include_langgraph_cases=False)
        self.assertEqual(len(core), CORE_CORPUS_SIZE)
        self.assertFalse(_LANGGRAPH_CASE_IDS & {c.case_id for c in core})
        self.assertTrue(all(LANGGRAPH_TAG not in c.tags for c in core))

    def test_core_only_subset_builds_and_runs_without_langgraph(self) -> None:
        with mock.patch.object(cases, "_LANGGRAPH_AVAILABLE", False):
            core = build_corpus(include_langgraph_cases=False)
        for case in core:
            with self.subTest(case=case.case_id):
                case.build_trace()
        self.assertEqual(len(run_corpus(core)), CORE_CORPUS_SIZE)

    def test_core_only_subset_category_floor(self) -> None:
        # Floors that hold for the explicit core-only subset (the full-corpus
        # floors are asserted, unchanged, in test_cases.py).
        by_category = Counter(c.category for c in build_corpus(include_langgraph_cases=False))
        self.assertGreaterEqual(by_category[CaseCategory.CLEAN], 3)
        self.assertGreaterEqual(by_category[CaseCategory.UNOBSERVABLE], 2)
        self.assertGreaterEqual(by_category[CaseCategory.HELD_OUT], 1)

    def test_full_corpus_without_langgraph_raises_instead_of_shrinking(self) -> None:
        with mock.patch.object(cases, "_LANGGRAPH_AVAILABLE", False):
            with self.assertRaises(BenchmarkDependencyError) as ctx:
                build_corpus()
        self.assertIn("langgraph", str(ctx.exception))
        self.assertIn(".[langgraph]", str(ctx.exception))

    def test_langgraph_case_traces_fail_loudly_without_langgraph(self) -> None:
        by_id = {c.case_id: c for c in cases._CASES}
        with mock.patch.object(cases, "_LANGGRAPH_AVAILABLE", False):
            for case_id in _LANGGRAPH_CASE_IDS:
                with self.subTest(case=case_id):
                    with self.assertRaises(BenchmarkDependencyError):
                        by_id[case_id].build_trace()

    def test_cli_without_langgraph_errors_and_names_the_extra(self) -> None:
        result = _run_without_langgraph("--no-output")
        self.assertEqual(result.returncode, 2, msg=result.stderr)
        self.assertIn("langgraph", result.stderr)
        self.assertIn("--core-only", result.stderr)
        self.assertNotIn("Benchmark summary", result.stdout)

    def test_cli_core_only_without_langgraph_runs_and_is_labelled(self) -> None:
        result = _run_without_langgraph("--core-only", "--no-output")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(f"Cases: {CORE_CORPUS_SIZE}", result.stdout)
        self.assertIn(f"{CORE_CORPUS_SIZE} of {FULL_CORPUS_SIZE}", result.stdout)


@requires_langgraph
class FullCorpusTests(unittest.TestCase):
    """Need the `langgraph` extra."""

    def test_default_corpus_is_the_full_corpus(self) -> None:
        full = build_corpus()
        self.assertEqual(len(full), FULL_CORPUS_SIZE)
        self.assertEqual(
            {c.case_id for c in full if LANGGRAPH_TAG in c.tags}, _LANGGRAPH_CASE_IDS
        )

    def test_full_corpus_is_core_plus_langgraph_cases(self) -> None:
        full_ids = [c.case_id for c in build_corpus()]
        core_ids = [c.case_id for c in build_corpus(include_langgraph_cases=False)]
        self.assertEqual(
            [i for i in full_ids if i not in _LANGGRAPH_CASE_IDS], core_ids
        )

    def test_langgraph_cases_build_and_run(self) -> None:
        langgraph_cases = [c for c in build_corpus() if LANGGRAPH_TAG in c.tags]
        self.assertEqual(len(run_corpus(langgraph_cases)), len(_LANGGRAPH_CASE_IDS))

    def test_cli_default_runs_all_cases(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "benchmark", "--no-output"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ},
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(f"Cases: {FULL_CORPUS_SIZE}", result.stdout)


if __name__ == "__main__":
    unittest.main()
