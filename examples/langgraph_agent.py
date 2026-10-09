"""LangGraph integration demonstration for AgentDebug.

Proves the Phase 5B integration composes with the existing, unmodified
Phase 3 checks and Phase 4 localization:

    LangGraph graph -> LangGraphAdapter.capture() -> AgentDebug Trace
        -> ALL_CHECKS (Phase 3, unchanged)
        -> (with an external FailureSignal) FirstObservedOccurrenceAnalyzer
           (Phase 4, unchanged)

Same refund-policy scenario as `examples/plain_python_agent.py`, now
expressed as a LangGraph `StateGraph` instead of plain Python calls, so the
two examples are directly comparable: whichever framework produced the
Trace, the checks and localization on top of it are identical code.

IMPORTANT: LangGraph checkpoints are NOT treated as execution steps in
this integration. A `StateSnapshot` from `get_state_history()` is graph
state/history evidence; a `Span` is execution-activity evidence, built
from `stream(..., stream_mode="updates")`. See
`agentdebug/integrations/langgraph/adapter.py`'s module docstring for why,
and for how a checkpoint can be produced by more than one node at once
(the fan-out case) without being turned into one Span per node.

Run as a module from the repository root (same reasoning as
`examples/plain_python_agent.py`):

    python -m examples.langgraph_agent
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentdebug.analysis.checks import ALL_CHECKS
from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer, LocalizationTarget
from agentdebug.core.enums import ExternalStatus
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.trace import Trace
from agentdebug.integrations.langgraph import LangGraphAdapter


class Scenario(Enum):
    CLEAN = "CLEAN"
    SWALLOWED_TOOL_ERROR = "SWALLOWED_TOOL_ERROR"
    LOCALIZATION_FAILURE = "LOCALIZATION_FAILURE"


class _RefundState(TypedDict):
    refund_eligible: bool


class ToolError(Exception):
    """Simulated failure from an external tool call."""


def _build_graph(scenario: Scenario):
    """One small StateGraph: get_order (TOOL) -> update_state (STATE).

    `get_order` is explicitly declared TOOL via `add_node(metadata=...)` --
    a real LangGraph API the adapter reads back (see its module docstring)
    -- rather than the adapter guessing a semantic category from generic
    execution evidence that doesn't carry one.
    """

    def get_order(state: _RefundState) -> dict:
        if scenario is Scenario.SWALLOWED_TOOL_ERROR:
            raise ToolError("order service unavailable")
        return {"refund_eligible": state["refund_eligible"]}

    def update_state(state: _RefundState) -> dict:
        eligible = state["refund_eligible"]
        if scenario is Scenario.LOCALIZATION_FAILURE:
            eligible = False
        return {"refund_eligible": eligible}

    builder = StateGraph(_RefundState)
    builder.add_node("get_order", get_order, metadata={"span_kind": "TOOL"})
    builder.add_node("update_state", update_state, metadata={"span_kind": "STATE"})
    builder.add_edge(START, "get_order")
    builder.add_edge("get_order", "update_state")
    builder.add_edge("update_state", END)
    return builder.compile(checkpointer=MemorySaver())


def build_trace(scenario: Scenario) -> Trace:
    """Run the graph via the real LangGraphAdapter and return the
    resulting AgentDebug Trace. For SWALLOWED_TOOL_ERROR, the agent
    (here, this function) deliberately does not retry after the caught
    ToolError -- that is what produces the swallowed-error pattern."""
    adapter = LangGraphAdapter(_build_graph(scenario))
    try:
        return adapter.capture({"refund_eligible": True}, trace_id=scenario.value)
    except ToolError:
        return adapter.get_trace(scenario.value)


def build_localization_failure_signal() -> FailureSignal:
    return FailureSignal(
        source="external_evaluator",
        verdict=ExternalStatus.FAILED,
        expected=True,
        actual=False,
        detail="refund_eligible was expected to remain True but the run produced False",
    )


def run_checks(trace: Trace) -> dict[str, Finding]:
    return {check.definition.id: check.run(trace) for check in ALL_CHECKS}


def _print_finding(check_id: str, finding: Finding) -> None:
    print(f"  {check_id}: {finding.status.value}")
    for claim in finding.claims:
        print(f"    - {claim.type.value}: {claim.fields}")


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
            finding = FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)
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
