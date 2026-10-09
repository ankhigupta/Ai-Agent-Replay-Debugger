"""Deterministic benchmark reports: human-readable text and machine JSON.

Both are rendered FROM already-computed `metrics.py` results — nothing in
this module calculates a metric itself, and nothing hard-codes an expected
value. This is a reporting format for benchmark evaluation metadata; it is
NOT a new Trace schema and does not touch `agentdebug.core.serialization`
(see `benchmark/__init__.py` and Phase 6's JSONL sink for where Trace
serialization actually lives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from benchmark.metrics import (
    AbstentionMetrics,
    CapabilityCoverage,
    CheckMetrics,
    LocalizationDistribution,
    OverheadMeasurement,
    compute_abstention_metrics,
    compute_all_check_metrics,
    compute_capability_coverage,
    compute_localization_distribution,
    measure_overhead,
)
from benchmark.models import BenchmarkCase, CaseResult


@dataclass(frozen=True)
class BenchmarkSummary:
    """Everything a report needs, bundled once so text/JSON rendering never
    recompute metrics differently from each other."""

    case_count: int
    check_metrics: dict[str, CheckMetrics]
    localization_distribution: LocalizationDistribution
    abstention: AbstentionMetrics
    capability_coverage: CapabilityCoverage
    overhead: OverheadMeasurement


def build_summary(
    cases: Sequence[BenchmarkCase], results: Sequence[CaseResult], *, overhead_iterations: int = 200
) -> BenchmarkSummary:
    return BenchmarkSummary(
        case_count=len(cases),
        check_metrics=compute_all_check_metrics(cases, results),
        localization_distribution=compute_localization_distribution(cases, results),
        abstention=compute_abstention_metrics(cases, results),
        capability_coverage=compute_capability_coverage(cases),
        overhead=measure_overhead(iterations=overhead_iterations),
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "undefined (zero denominator)"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_text_report(summary: BenchmarkSummary) -> str:
    lines: list[str] = []
    lines.append("Benchmark summary")
    lines.append("-----------------")
    lines.append(f"Cases: {summary.case_count}")
    lines.append("")

    lines.append("Trace checks")
    lines.append("-----------------")
    for check_id in sorted(summary.check_metrics):
        m = summary.check_metrics[check_id]
        lines.append(check_id)
        lines.append(f"  TP: {m.true_positives}  FP: {m.false_positives}  FN: {m.false_negatives}  TN: {m.true_negatives}")
        lines.append(f"  precision: {_fmt(m.precision)}")
        lines.append(f"  recall: {_fmt(m.recall)}")
    lines.append("")

    lines.append("Localization")
    lines.append("-----------------")
    for label, outcomes in summary.localization_distribution.by_label.items():
        total = sum(outcomes.values())
        if total == 0:
            lines.append(f"{label.value}: (no cases)")
            continue
        breakdown = ", ".join(f"{outcome.value}={count}" for outcome, count in outcomes.items() if count)
        lines.append(f"{label.value}: {total} case(s) -- {breakdown}")
    lines.append("")

    lines.append("Abstention")
    lines.append("-----------------")
    a = summary.abstention
    lines.append(
        f"  precision: {_fmt(a.precision)} "
        f"({a.appropriate_abstention_count}/{a.abstained_count} abstentions were appropriate)"
    )
    lines.append(
        f"  false-abstention rate: {_fmt(a.false_abstention_rate)} "
        f"({a.false_abstention_count}/{a.evidence_available_count} available-evidence results wrongly abstained)"
    )
    lines.append("")

    lines.append("Capability coverage")
    lines.append("-----------------")
    c = summary.capability_coverage
    lines.append(f"  required capabilities exercised by this corpus: {sorted(x.value for x in c.required)}")
    lines.append(f"  cases with a missing capability: {list(c.cases_with_missing_capability)}")
    lines.append(f"  missing-capability counts: {{{', '.join(f'{k.value}: {v}' for k, v in c.missing_capability_counts.items())}}}")
    lines.append("")

    lines.append("Runtime (benchmark-environment measurement only -- not a general performance claim)")
    lines.append("-----------------")
    o = summary.overhead
    lines.append(f"  iterations: {o.iterations}")
    lines.append(f"  baseline: {o.baseline_ms:.6f} ms/call")
    lines.append(f"  instrumented: {o.instrumented_ms:.6f} ms/call")
    lines.append(f"  overhead: {o.overhead_ms:.6f} ms/call")
    if o.baseline_ms < 0.01:
        lines.append(
            "  overhead_percentage: undefined-in-practice (baseline is near-zero, so a percentage "
            "is not a meaningful figure here -- see the raw ms values above)"
        )
    else:
        lines.append(f"  overhead_percentage: {_fmt(o.overhead_percentage)}%")

    return "\n".join(lines)


def render_json_report(summary: BenchmarkSummary) -> dict[str, Any]:
    """Plain JSON-safe dict. Deliberately hand-built (not a dataclass
    `asdict()`), since enum keys need `.value` and `None` denominators must
    stay explicit `null`, not be silently coerced."""
    return {
        "case_count": summary.case_count,
        "check_metrics": {
            check_id: {
                "true_positives": m.true_positives,
                "false_positives": m.false_positives,
                "false_negatives": m.false_negatives,
                "true_negatives": m.true_negatives,
                "precision": m.precision,
                "recall": m.recall,
            }
            for check_id, m in summary.check_metrics.items()
        },
        "localization_distribution": {
            label.value: {outcome.value: count for outcome, count in outcomes.items()}
            for label, outcomes in summary.localization_distribution.by_label.items()
        },
        "abstention": {
            "abstained_count": summary.abstention.abstained_count,
            "appropriate_abstention_count": summary.abstention.appropriate_abstention_count,
            "evidence_available_count": summary.abstention.evidence_available_count,
            "false_abstention_count": summary.abstention.false_abstention_count,
            "precision": summary.abstention.precision,
            "false_abstention_rate": summary.abstention.false_abstention_rate,
        },
        "capability_coverage": {
            "required": sorted(c.value for c in summary.capability_coverage.required),
            "cases_with_missing_capability": list(summary.capability_coverage.cases_with_missing_capability),
            "missing_capability_counts": {
                c.value: count for c, count in summary.capability_coverage.missing_capability_counts.items()
            },
        },
        "runtime": {
            "iterations": summary.overhead.iterations,
            "baseline_ms": summary.overhead.baseline_ms,
            "instrumented_ms": summary.overhead.instrumented_ms,
            "overhead_ms": summary.overhead.overhead_ms,
            "overhead_percentage": summary.overhead.overhead_percentage,
        },
    }
