"""The deterministic benchmark runner.

    for case in corpus:
        trace = case.build_trace()
        findings = {check_id: check.run(trace, available_capabilities=...) ...}
        if case.build_failure_signal: localization = analyzer.run(trace, failure_signal, target=..., available_capabilities=...)
        record CaseResult

GROUND-TRUTH ISOLATION (the load-bearing property of this whole package):
`run_case` below is the ONLY place that calls into `agentdebug` analyzers.
Read it top to bottom: every argument passed to `check.run(...)` and to
`FirstObservedOccurrenceAnalyzer().run(...)` comes from `trace`,
`failure_signal`, `case.localization_target`, or
`case.available_capabilities` — never from `case.localization_label`,
`case.reference_span_name`, or `case.evidence_available`, which are never
even read in this function. Nothing here inspects AgentDebug internals to
find a "planted" fault, nothing modifies a `Trace` after analysis to make
a result match, and nothing reinterprets a `NO_FINDING`/`INCONCLUSIVE`
result as success or failure after the fact — `metrics.py` compares the
Finding AgentDebug actually produced against ground truth, but the
analyzer call itself never sees that ground truth.
"""

from __future__ import annotations

import time
from typing import Iterable

from agentdebug.analysis.checks import ALL_CHECKS
from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer, LocalizationTarget
from benchmark.models import BenchmarkCase, CaseResult

_CHECKS_BY_ID = {check.definition.id: check for check in ALL_CHECKS}


def run_case(case: BenchmarkCase) -> CaseResult:
    """Execute one BenchmarkCase through real AgentDebug paths and return
    what actually happened. Does not read `case.localization_label`,
    `case.reference_span_name`, or `case.evidence_available` — see module
    docstring."""
    start = time.perf_counter()

    trace = case.build_trace()

    check_findings = {}
    for check_id in case.expected_check_findings:
        check = _CHECKS_BY_ID[check_id]
        check_findings[check_id] = check.run(trace, available_capabilities=case.available_capabilities)

    localization_finding = None
    if case.build_failure_signal is not None:
        failure_signal = case.build_failure_signal()
        target = case.localization_target
        if target is None:
            target = LocalizationTarget.from_failure_signal(failure_signal)
        localization_finding = FirstObservedOccurrenceAnalyzer().run(
            trace, failure_signal, target=target, available_capabilities=case.available_capabilities
        )

    duration = time.perf_counter() - start

    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        trace_id=trace.id,
        span_count=len(trace.spans),
        span_names_by_id={span.id: span.name for span in trace.spans},
        check_findings=check_findings,
        localization_finding=localization_finding,
        capabilities_used=case.available_capabilities,
        duration_seconds=duration,
    )


def run_corpus(cases: Iterable[BenchmarkCase]) -> list[CaseResult]:
    """Run every case, in the given order (deterministic — callers should
    pass the corpus in a fixed, stable order, which
    `benchmark.cases.build_corpus()` returns)."""
    return [run_case(case) for case in cases]
