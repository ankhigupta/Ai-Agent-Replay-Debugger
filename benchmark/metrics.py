"""Deterministic benchmark metrics.

Every metric here is computed from `BenchmarkCase` ground truth (never fed
to an analyzer — see `runner.py`) compared against the `Finding`s
AgentDebug actually produced. Four kinds of metric, kept explicitly
separate per the Phase 7 brief (section 10/11/12/13):

1. Per-check precision/recall (`compute_check_metrics`) — a confusion
   matrix per check, never one undifferentiated "accuracy" number across
   different checks.
2. Localization distribution (`compute_localization_distribution`) — a
   breakdown by `LocalizationLabel`, never a single accuracy number.
3. Abstention metrics (`compute_abstention_metrics`) — precision and
   false-abstention rate, both with explicit, documented denominators;
   `None` when a denominator is zero rather than a hidden/fabricated 0.0.
4. Capability coverage (`compute_capability_coverage`) — using each
   analyzer's own `definition.required_capabilities`, never assuming
   coverage merely because a Trace exists.

Plus `measure_overhead`, a small, honestly-labeled local timing comparison
(not a claim of general performance characteristics).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from agentdebug.analysis.checks import ALL_CHECKS
from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer
from agentdebug.core.enums import Capability, DiagnosisConfidence, InternalCheckStatus, SpanKind
from agentdebug.recording.recorder import Recorder
from benchmark.models import BenchmarkCase, CaseResult, LocalizationLabel, LocalizationOutcome

_CHECK_DEFINITIONS = {check.definition.id: check.definition for check in ALL_CHECKS}
_LOCALIZATION_DEFINITION = FirstObservedOccurrenceAnalyzer().definition

_ABSTAINING_STATUSES = (InternalCheckStatus.NOT_APPLICABLE, InternalCheckStatus.INCONCLUSIVE)


# ---------------------------------------------------------------------------
# 1. Per-check precision / recall
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckMetrics:
    """Confusion matrix for one check, over only the cases that define an
    expectation of `FLAGGED` or `NO_FINDING` for it. "Positive" = the check
    is expected to / did produce `FLAGGED`. Cases where the check abstained
    (`NOT_APPLICABLE`/`INCONCLUSIVE`) or wasn't evaluated at all for this
    case are excluded from this confusion matrix entirely — they are not
    meaningful true-negatives, and are covered instead by
    `compute_abstention_metrics`.
    """

    check_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: Optional[float]
    recall: Optional[float]


def compute_check_metrics(
    cases: Sequence[BenchmarkCase], results_by_case_id: dict, check_id: str
) -> CheckMetrics:
    tp = fp = fn = tn = 0
    for case in cases:
        expected = case.expected_check_findings.get(check_id)
        if expected not in (InternalCheckStatus.FLAGGED, InternalCheckStatus.NO_FINDING):
            continue  # not a binary expectation for this check on this case
        result = results_by_case_id[case.case_id]
        finding = result.check_findings.get(check_id)
        if finding is None or finding.status in _ABSTAINING_STATUSES:
            continue  # abstained -- not part of this confusion matrix
        observed_positive = finding.status == InternalCheckStatus.FLAGGED
        expected_positive = expected == InternalCheckStatus.FLAGGED
        if observed_positive and expected_positive:
            tp += 1
        elif observed_positive and not expected_positive:
            fp += 1
        elif not observed_positive and expected_positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return CheckMetrics(
        check_id=check_id,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
    )


def compute_all_check_metrics(
    cases: Sequence[BenchmarkCase], results: Sequence[CaseResult]
) -> dict[str, CheckMetrics]:
    results_by_case_id = {r.case_id: r for r in results}
    check_ids = sorted({check_id for case in cases for check_id in case.expected_check_findings})
    return {
        check_id: compute_check_metrics(cases, results_by_case_id, check_id) for check_id in check_ids
    }


# ---------------------------------------------------------------------------
# 2. Localization distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalizationDistribution:
    """Counts of `LocalizationOutcome` grouped by ground-truth
    `LocalizationLabel`. A benchmark label of ORIGIN mapping mostly to
    UNIQUE_CORRECT is a localization *relationship*, reported as such —
    never auto-collapsed into "AgentDebug was right/wrong"."""

    by_label: dict[LocalizationLabel, dict[LocalizationOutcome, int]]

    def total(self) -> int:
        return sum(count for outcomes in self.by_label.values() for count in outcomes.values())


def compute_localization_distribution(
    cases: Sequence[BenchmarkCase], results: Sequence[CaseResult]
) -> LocalizationDistribution:
    results_by_case_id = {r.case_id: r for r in results}
    by_label: dict[LocalizationLabel, dict[LocalizationOutcome, int]] = {
        label: {outcome: 0 for outcome in LocalizationOutcome} for label in LocalizationLabel
    }

    for case in cases:
        if case.localization_label is None:
            continue
        result = results_by_case_id[case.case_id]
        finding = result.localization_finding
        if finding is None:
            continue

        if finding.status in _ABSTAINING_STATUSES:
            outcome = LocalizationOutcome.ABSTAINED
        elif finding.status == InternalCheckStatus.NO_FINDING:
            outcome = LocalizationOutcome.NO_OBSERVATION
        elif finding.diagnosis_confidence == DiagnosisConfidence.AMBIGUOUS:
            outcome = LocalizationOutcome.AMBIGUOUS
        else:
            observed_names = {
                result.span_names_by_id.get(sid) for sid in finding.claims[0].span_ids
            }
            if case.reference_span_name is None:
                # Ground truth says "context", not a span: correct iff
                # AgentDebug pointed at no span at all (a context
                # observation also carries no span_ids).
                correct = not finding.claims[0].span_ids
            else:
                correct = case.reference_span_name in observed_names
            outcome = LocalizationOutcome.UNIQUE_CORRECT if correct else LocalizationOutcome.UNIQUE_INCORRECT

        by_label[case.localization_label][outcome] += 1

    return LocalizationDistribution(by_label=by_label)


# ---------------------------------------------------------------------------
# 3. Abstention metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstentionMetrics:
    """Computed over every individual check/localization result produced
    across the whole corpus (not per-case), wherever the originating
    case's `evidence_available` ground truth is defined (always is).

    `precision` = count(abstained AND evidence genuinely unavailable) /
        count(abstained). `None` if nothing abstained.
    `false_abstention_rate` = count(abstained AND evidence genuinely
        available) / count(evidence genuinely available). `None` if no
        result had available evidence.
    """

    abstained_count: int
    appropriate_abstention_count: int
    evidence_available_count: int
    false_abstention_count: int
    precision: Optional[float]
    false_abstention_rate: Optional[float]


def compute_abstention_metrics(
    cases: Sequence[BenchmarkCase], results: Sequence[CaseResult]
) -> AbstentionMetrics:
    results_by_case_id = {r.case_id: r for r in results}

    abstained = 0
    appropriate_abstention = 0
    evidence_available = 0
    false_abstention = 0

    for case in cases:
        result = results_by_case_id[case.case_id]
        observations = list(result.check_findings.values())
        if result.localization_finding is not None:
            observations.append(result.localization_finding)

        for finding in observations:
            did_abstain = finding.status in _ABSTAINING_STATUSES
            if case.evidence_available:
                evidence_available += 1
                if did_abstain:
                    false_abstention += 1
            if did_abstain:
                abstained += 1
                if not case.evidence_available:
                    appropriate_abstention += 1

    precision = appropriate_abstention / abstained if abstained > 0 else None
    false_abstention_rate = false_abstention / evidence_available if evidence_available > 0 else None

    return AbstentionMetrics(
        abstained_count=abstained,
        appropriate_abstention_count=appropriate_abstention,
        evidence_available_count=evidence_available,
        false_abstention_count=false_abstention,
        precision=precision,
        false_abstention_rate=false_abstention_rate,
    )


# ---------------------------------------------------------------------------
# 4. Capability coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityCoverage:
    required: frozenset
    cases_with_missing_capability: tuple[str, ...]
    missing_capability_counts: dict


def _required_capabilities_for_case(case: BenchmarkCase) -> frozenset:
    required: set = set()
    for check_id in case.expected_check_findings:
        required.update(_CHECK_DEFINITIONS[check_id].required_capabilities)
    if case.build_failure_signal is not None:
        required.update(_LOCALIZATION_DEFINITION.required_capabilities)
    return frozenset(required)


def compute_capability_coverage(cases: Sequence[BenchmarkCase]) -> CapabilityCoverage:
    all_required: set = set()
    missing_counts: Counter = Counter()
    affected_cases: list[str] = []

    for case in cases:
        required = _required_capabilities_for_case(case)
        all_required.update(required)
        if case.available_capabilities is None:
            continue  # no restriction simulated -- nothing reported missing
        missing = required - case.available_capabilities
        if missing:
            affected_cases.append(case.case_id)
            missing_counts.update(missing)

    return CapabilityCoverage(
        required=frozenset(all_required),
        cases_with_missing_capability=tuple(affected_cases),
        missing_capability_counts=dict(missing_counts),
    )


# ---------------------------------------------------------------------------
# Overhead
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverheadMeasurement:
    """Raw local timing measurements, benchmark-environment only. Not a
    general performance claim -- see `report.py`'s rendering, which labels
    this explicitly."""

    iterations: int
    baseline_ms: float
    instrumented_ms: float
    overhead_ms: float
    overhead_percentage: Optional[float]


def _baseline_work() -> str:
    order = {"order_id": "ORD-42", "refund_eligible": True}
    eligible = order["refund_eligible"]
    return "approved" if eligible else "denied"


def _instrumented_work() -> str:
    recorder = Recorder()
    recorder.start_trace()
    with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}) as span:
        order = {"order_id": "ORD-42", "refund_eligible": True}
        span.set_output(order)
    with recorder.span("decide_response", kind=SpanKind.CUSTOM) as span:
        eligible = order["refund_eligible"]
        decision = "approved" if eligible else "denied"
        span.set_output({"decision": decision})
    recorder.end_trace()
    return decision


def measure_overhead(iterations: int = 200) -> OverheadMeasurement:
    for _ in range(5):  # warm-up, excluded from the measurement
        _baseline_work()
        _instrumented_work()

    start = time.perf_counter()
    for _ in range(iterations):
        _baseline_work()
    baseline_total = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        _instrumented_work()
    instrumented_total = time.perf_counter() - start

    baseline_ms = (baseline_total / iterations) * 1000
    instrumented_ms = (instrumented_total / iterations) * 1000
    overhead_ms = instrumented_ms - baseline_ms
    overhead_percentage = (overhead_ms / baseline_ms * 100) if baseline_ms > 0 else None

    return OverheadMeasurement(
        iterations=iterations,
        baseline_ms=baseline_ms,
        instrumented_ms=instrumented_ms,
        overhead_ms=overhead_ms,
        overhead_percentage=overhead_percentage,
    )
