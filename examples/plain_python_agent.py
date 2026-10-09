"""Plain-Python end-to-end demonstration of AgentDebug.

Proves the existing Phase 1-4 primitives compose, around an ordinary
Python "agent" with no framework, no LLM, and no network calls:

    Plain Python Agent -> Recorder -> Trace + Spans -> FailureSignal
        -> Trace-driven checks + failure-driven localization -> Findings

The "agent" is a deterministic refund-policy workflow:

    user request -> retrieve refund policy (TOOL)
                  -> look up the order  (TOOL)
                  -> evaluate eligibility / update state (STATE)
                  -> produce a response (CUSTOM -- no real LLM is involved;
                     CUSTOM is used honestly instead of pretending an LLM
                     made this decision)

Five explicitly-selected scenarios (never randomized) exercise every
Phase 3 check and Phase 4 localization:

    CLEAN                  everything succeeds, no findings expected
    SWALLOWED_TOOL_ERROR    a tool call fails and is never retried
    REPEATED_TOOL_CALL      the same tool call repeats 3+ times
    LOCALIZATION_FAILURE    a value flips mid-run; an external evaluator
                             reports the mismatch via a FailureSignal
    INCOMPLETE_TRACE        one span has a dangling parent_id

Run as a module from the repository root (`examples` is not part of the
installed distribution, so it is only importable with the repository root
on `sys.path` -- `python -m` from that directory does this the same way
`python -m unittest` already does for this project's tests):

    python -m examples.plain_python_agent

IMPORTANT: this module only ever reports what pattern was observed, or
where a value was first observed. It never claims a span caused a
failure, that a value was wrong, or who/what is responsible.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from agentdebug.analysis.checks import ALL_CHECKS
from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer, LocalizationTarget
from agentdebug.core.context import TraceContext
from agentdebug.core.enums import ExternalStatus, SpanKind, SpanStatus
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace
from agentdebug.recording.recorder import Recorder


class Scenario(Enum):
    CLEAN = "CLEAN"
    SWALLOWED_TOOL_ERROR = "SWALLOWED_TOOL_ERROR"
    REPEATED_TOOL_CALL = "REPEATED_TOOL_CALL"
    LOCALIZATION_FAILURE = "LOCALIZATION_FAILURE"
    INCOMPLETE_TRACE = "INCOMPLETE_TRACE"


class ToolError(Exception):
    """Simulated failure from an external tool call."""


_ORDER_ID = "ORD-42"
_ORDER_INPUT = {"order_id": _ORDER_ID}


def _run_agent(recorder: Recorder, scenario: Scenario) -> None:
    """The deterministic refund-policy workflow, recorded via the real
    Phase 2 `Recorder` API. `scenario` only ever changes which explicit
    branch runs -- nothing here is randomized."""

    # Step 1: retrieve the refund policy (TOOL).
    with recorder.span("get_refund_policy", kind=SpanKind.TOOL, input={"policy": "STANDARD"}) as policy_span:
        policy_span.set_output({"max_days": 30, "eligible_categories": ["electronics", "books"]})

    # Step 2: look up the order (TOOL). This is where SWALLOWED_TOOL_ERROR
    # and REPEATED_TOOL_CALL are deliberately produced.
    order = {"order_id": _ORDER_ID, "category": "electronics", "refund_eligible": True}

    if scenario is Scenario.SWALLOWED_TOOL_ERROR:
        try:
            with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT):
                raise ToolError("order service unavailable")
        except ToolError:
            # The agent deliberately does not retry or fall back here --
            # this is what produces the swallowed-tool-error pattern.
            pass
    elif scenario is Scenario.REPEATED_TOOL_CALL:
        for _ in range(3):
            with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as order_span:
                order_span.set_output(order)
    else:
        with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as order_span:
            order_span.set_output(order)

    # Step 3: evaluate eligibility and update state (STATE). This is
    # where LOCALIZATION_FAILURE flips the value that an external
    # evaluator later reports as unexpected.
    eligible = order["refund_eligible"]
    if scenario is Scenario.LOCALIZATION_FAILURE:
        eligible = False
    with recorder.span("update_refund_state", kind=SpanKind.STATE) as state_span:
        state_span.set_output({"refund_eligible": eligible})

    # Step 4: produce the final response (CUSTOM -- no real LLM is
    # involved; CUSTOM is used honestly rather than pretending one was).
    decision = "approved" if eligible else "denied"
    with recorder.span(
        "decide_response", kind=SpanKind.CUSTOM, input={"refund_eligible": eligible}
    ) as decision_span:
        decision_span.set_output({"decision": decision, "message": f"Refund {decision}."})


def build_trace(scenario: Scenario) -> Trace:
    """Run the demo agent under `scenario` using the real Recorder, and
    return the resulting Trace. For INCOMPLETE_TRACE, one additional span
    with a dangling parent_id is added directly via `Trace.add_span` after
    recording -- the Recorder itself never produces one (by Phase 2
    design), so this models evidence that arrived some other way without
    modifying the Recorder."""
    recorder = Recorder()
    trace = recorder.start_trace(context=TraceContext(initial_state={"refund_eligible": True}))
    _run_agent(recorder, scenario)
    recorder.end_trace()

    if scenario is Scenario.INCOMPLETE_TRACE:
        last_end_time = trace.spans[-1].end_time
        orphan = Span(
            id="orphan-shipping-check",
            parent_id="never-recorded-parent",
            kind=SpanKind.TOOL,
            name="get_shipping_status",
            status=SpanStatus.OK,
            start_time=last_end_time + 1.0,
            end_time=last_end_time + 1.1,
            recorder_sequence_id=len(trace.spans),
            input={"order_id": _ORDER_ID},
            output={"status": "in_transit"},
        )
        trace.add_span(orphan)

    return trace


def build_localization_failure_signal() -> FailureSignal:
    """The external evaluator's report for LOCALIZATION_FAILURE: it
    expected refund_eligible to remain True, and observed False."""
    return FailureSignal(
        source="external_evaluator",
        verdict=ExternalStatus.FAILED,
        expected=True,
        actual=False,
        detail="refund_eligible was expected to remain True but the run produced False",
    )


def run_checks(trace: Trace) -> dict[str, Finding]:
    """Run every Phase 3 TraceCheck against `trace`. Reuses the existing
    checks as-is (`agentdebug.analysis.checks.ALL_CHECKS`) -- no check
    logic is duplicated here."""
    return {check.definition.id: check.run(trace) for check in ALL_CHECKS}


def run_localization(trace: Trace, failure_signal: FailureSignal, target: LocalizationTarget) -> Finding:
    """Run Phase 4 failure-driven localization. Only meaningful when an
    external FailureSignal actually exists -- trace checks (above) need no
    such signal, and this function is never called without one."""
    return FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)


def _print_finding(check_id: str, finding: Finding) -> None:
    print(f"  {check_id}: {finding.status.value}")
    for claim in finding.claims:
        print(f"    - {claim.type.value}: {claim.fields}")
    if finding.capability_gaps:
        print(f"    capability_gaps: {[c.value for c in finding.capability_gaps]}")


def main() -> None:
    for scenario in Scenario:
        print(f"\nScenario: {scenario.value}")
        trace = build_trace(scenario)
        print(f"Trace: {len(trace.spans)} spans recorded")

        print("Trace checks:")
        for check_id, finding in run_checks(trace).items():
            _print_finding(check_id, finding)

        if scenario is Scenario.LOCALIZATION_FAILURE:
            failure_signal = build_localization_failure_signal()
            target = LocalizationTarget(value=False, field_path="refund_eligible")
            print(
                f"FailureSignal: expected={failure_signal.expected!r} "
                f"actual={failure_signal.actual!r} verdict={failure_signal.verdict.value}"
            )
            finding = run_localization(trace, failure_signal, target)
            print("FirstObservedOccurrence:")
            print(f"  status: {finding.status.value}")
            if finding.claims:
                claim = finding.claims[0]
                print(f"  target: {claim.fields.get('target_value')!r}")
                print(f"  first observed at: {claim.fields.get('observed_at')} (span count: {len(claim.span_ids)})")
            confidence = finding.diagnosis_confidence
            print(f"  confidence: {confidence.value if confidence is not None else None}")


if __name__ == "__main__":
    main()
