"""Run the Phase 7 benchmark: `python -m benchmark [--output-dir PATH]`.

Executes the full deterministic corpus, prints the human-readable report,
and (unless `--no-output`) writes a machine-readable JSON report plus a
JSONL evidence file (via the existing Phase 6 `JsonlEvidenceSink` —
no new serialization format).

Default output directory is `.benchmark_output/` under the current working
directory, which is gitignored (see `.gitignore`) specifically so
a normal run never produces a file this repository tracks. Pass
`--output-dir` to write elsewhere.

Run from the repository root:

    python -m benchmark

The full corpus includes cases captured through the real LangGraph
integration, so the default run needs the optional extra
(`pip install ".[langgraph]"`) and exits with an error if LangGraph is
missing. `--core-only` explicitly runs only the cases that need no
LangGraph; the report is then a labelled subset, never presented as the
full benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentdebug.evidence import JsonlEvidenceSink
from benchmark.cases import CORE_CORPUS_SIZE, FULL_CORPUS_SIZE, BenchmarkDependencyError, build_corpus
from benchmark.report import build_summary, render_json_report, render_text_report
from benchmark.runner import run_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgentDebug Phase 7 benchmark.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".benchmark_output"), help="Where to write evidence.jsonl/report.json"
    )
    parser.add_argument(
        "--no-output", action="store_true", help="Print the text report only; write no files."
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help=f"Run only the {CORE_CORPUS_SIZE} cases that need no LangGraph, instead of the full "
        f"{FULL_CORPUS_SIZE}-case corpus (which requires the 'langgraph' extra).",
    )
    parser.add_argument("--overhead-iterations", type=int, default=200)
    args = parser.parse_args()

    try:
        corpus = build_corpus(include_langgraph_cases=not args.core_only)
    except BenchmarkDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if args.core_only:
        print(
            f"NOTE: --core-only subset ({len(corpus)} of {FULL_CORPUS_SIZE} cases); "
            "LangGraph cases are excluded, so this is not the full benchmark.\n"
        )
    results = run_corpus(corpus)
    summary = build_summary(corpus, results, overhead_iterations=args.overhead_iterations)

    print(render_text_report(summary))

    if args.no_output:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = args.output_dir / "evidence.jsonl"
    with JsonlEvidenceSink(evidence_path) as sink:
        for case in corpus:
            sink.write_trace(case.build_trace())

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(render_json_report(summary), indent=2), encoding="utf-8")

    print(f"\nWrote {len(corpus)} evidence record(s) to {evidence_path}")
    print(f"Wrote JSON report to {report_path}")


if __name__ == "__main__":
    main()
