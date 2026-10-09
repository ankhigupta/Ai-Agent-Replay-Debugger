"""Benchmark data model: cases, results, and benchmark-only ground-truth labels.

This module is evaluation tooling, not SDK core. The enums defined here
(`CaseCategory`, `LocalizationLabel`, `LocalizationOutcome`) are deliberately
NOT added to `agentdebug.core.enums` — they describe what the BENCHMARK
author knows or is checking for, never something `agentdebug` itself
produces or is allowed to see. See the module docstring of `runner.py` for
exactly how that separation is enforced in code.

Epistemic note, load-bearing for this whole package: a `BenchmarkCase`'s
`localization_label` and `reference_span_name` are ground truth the
benchmark author plants when constructing a case (e.g. "span `update_state`
is where I, the benchmark author, decided the deviation happens"). Nothing
in `agentdebug` ever reads these fields. `agentdebug` only ever sees a
`Trace` and, where applicable, a `FailureSignal` built the way a real
external evaluator would build one — never the planted answer. Comparing
`agentdebug`'s observational output ("value X was first observed at span
Y") against this ground truth is the benchmark's job, done AFTER the SDK
has already produced its result; it is evaluation, not causal proof, and
it never feeds back into what the SDK is allowed to claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional

from agentdebug.analysis.localization import LocalizationTarget
from agentdebug.core.enums import Capability, DiagnosisConfidence, InternalCheckStatus
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.trace import Trace


class CaseCategory(str, Enum):
    """What kind of benchmark case this is. Five values, per spec -- do
    not add a sixth; use `BenchmarkCase.tags` for finer narrative
    distinctions (e.g. "self-consistent wrong") instead."""

    CLEAN = "CLEAN"
    FAULT = "FAULT"
    NO_FAULT_CONTROL = "NO_FAULT_CONTROL"
    UNOBSERVABLE = "UNOBSERVABLE"
    HELD_OUT = "HELD_OUT"


class LocalizationLabel(str, Enum):
    """Benchmark-only ground truth describing WHAT KIND of point a planted
    deviation's reference span represents in the case author's own
    understanding of the scenario. Never a ClaimType; never seen by
    `agentdebug`.

    ORIGIN: the point (span or pre-run context) where the case author
        planted the deviation's value.
    FIRST_OBSERVABLE_DEVIATION: the first point within the CAPTURED trace
        evidence where the deviation is visible, when that may differ from
        a conceptual origin outside the trace (e.g. context already wrong
        before any span ran) or when multiple spans echo the same value.
    SYMPTOM: a later, non-originating point that merely re-expresses or
        reflects an already-established deviation (e.g. a logging/
        notification step), not the point that produced it.
    TERMINAL: the final, end-of-workflow manifestation of the deviation
        (e.g. the response/decision step).
    NONE: no deviation exists to localize (used for the "target not
        observed" control).
    """

    ORIGIN = "ORIGIN"
    FIRST_OBSERVABLE_DEVIATION = "FIRST_OBSERVABLE_DEVIATION"
    SYMPTOM = "SYMPTOM"
    TERMINAL = "TERMINAL"
    NONE = "NONE"


class LocalizationOutcome(str, Enum):
    """Benchmark-only classification of what AgentDebug actually did on a
    localization case, independent of whether that matches ground truth.
    Reported as a relationship against `LocalizationLabel`, never
    collapsed into a single pass/fail."""

    UNIQUE_CORRECT = "UNIQUE_CORRECT"
    UNIQUE_INCORRECT = "UNIQUE_INCORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    NO_OBSERVATION = "NO_OBSERVATION"
    ABSTAINED = "ABSTAINED"


@dataclass(frozen=True, kw_only=True)
class BenchmarkCase:
    """One deterministic benchmark case.

    `build_trace`/`build_failure_signal` are zero-argument callables (not
    already-built objects) so each case is freshly, deterministically
    constructed every time it runs — no shared mutable state between runs.

    Ground-truth fields (`localization_label`, `reference_span_name`,
    `evidence_available`) exist ONLY for post-hoc evaluation in
    `metrics.py`. `runner.py` never passes them to an `agentdebug` call.

    `reference_span_name` (not an id): span ids are often generated at
    build time (e.g. `uuid4()` inside the Recorder), so ground truth
    refers to a span by its stable, author-chosen `name` instead; the
    benchmark resolves the actual id AFTER building the trace, purely for
    comparison.
    """

    case_id: str
    description: str
    category: CaseCategory
    build_trace: Callable[[], Trace]
    build_failure_signal: Optional[Callable[[], FailureSignal]] = None
    localization_target: Optional[LocalizationTarget] = None
    expected_check_findings: Mapping[str, InternalCheckStatus] = field(default_factory=dict)
    expected_localization_status: Optional[InternalCheckStatus] = None
    expected_localization_confidence: Optional[DiagnosisConfidence] = None
    localization_label: Optional[LocalizationLabel] = None
    reference_span_name: Optional[str] = None
    available_capabilities: Optional[frozenset[Capability]] = None
    evidence_available: bool = True
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CaseResult:
    """What actually happened when a `BenchmarkCase` was run.

    Deliberately does not embed the whole `Trace` object (cases are
    deterministic, so `case.build_trace()` reproduces it on demand for
    evidence export instead of duplicating it in every result). It DOES
    keep `span_names_by_id` — a small `{span.id: span.name}` map from the
    exact `Trace` instance this result's findings were produced against —
    because that is the only way `metrics.py` can resolve a Finding's
    `span_ids` back to the stable span *names* `BenchmarkCase.
    reference_span_name` ground truth is expressed in. Span ids are
    freshly generated (e.g. `uuid4()`) on every `build_trace()` call, so a
    second, separate rebuild would have different ids and could never be
    compared against this result's findings.
    """

    case_id: str
    category: CaseCategory
    trace_id: str
    span_count: int
    span_names_by_id: Mapping[str, str]
    check_findings: Mapping[str, Finding]
    localization_finding: Optional[Finding]
    capabilities_used: Optional[frozenset[Capability]]
    duration_seconds: float
