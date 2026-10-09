"""The Phase 7 deterministic benchmark corpus.

Every case is built from a zero-argument callable (`BenchmarkCase.build_trace`
/ `build_failure_signal`), reusing real AgentDebug integration paths — the
Phase 2 `Recorder`, the real `examples.plain_python_agent` scenarios where
they fit directly, and (where LangGraph is installed) the real Phase 5B
`LangGraphAdapter` — rather than hand-fabricating `Trace` objects wherever a
real path exists. No randomness, no wall-clock-dependent semantics, no
network/LLM calls anywhere in this file.

Ground-truth fields (`localization_label`, `reference_span_name`,
`evidence_available`) are benchmark-author knowledge attached to the
`BenchmarkCase`, never passed into any `agentdebug` call — see
`runner.py`'s module docstring for exactly how that boundary is enforced.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from agentdebug.analysis.localization import LocalizationTarget
from agentdebug.core.context import TraceContext
from agentdebug.core.enums import (
    Capability,
    DiagnosisConfidence,
    ExternalStatus,
    InternalCheckStatus,
    SpanKind,
    SpanStatus,
)
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace
from agentdebug.recording.recorder import Recorder
from benchmark.models import BenchmarkCase, CaseCategory, LocalizationLabel
from examples.plain_python_agent import Scenario as _PPAScenario
from examples.plain_python_agent import build_trace as _ppa_build_trace

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    from agentdebug.integrations.langgraph import LangGraphAdapter

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


class BenchmarkDependencyError(RuntimeError):
    """The full benchmark corpus needs an optional dependency that is missing."""


LANGGRAPH_TAG = "requires_langgraph"
"""Tag carried by every case whose trace is captured through the real
LangGraph integration. The set of tagged cases is exactly what
`build_corpus(include_langgraph_cases=False)` excludes."""

FULL_CORPUS_SIZE = 29
CORE_CORPUS_SIZE = 27


def _require_langgraph() -> None:
    if not _LANGGRAPH_AVAILABLE:
        raise BenchmarkDependencyError(
            "This benchmark case captures a real LangGraph execution and needs the "
            "optional 'langgraph' extra: pip install '.[langgraph]'"
        )


_ORDER_INPUT = {"order_id": "ORD-42"}


# ---------------------------------------------------------------------------
# A. CLEAN
# ---------------------------------------------------------------------------


def _trace_clean_order_lookup() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
        span.set_output({"order_id": "ORD-42", "category": "books"})
    recorder.end_trace()
    return trace


def _trace_clean_state_transition() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace(context=TraceContext(initial_state={"status": "pending"}))
    with recorder.span("update_shipping_state", kind=SpanKind.STATE) as span:
        span.set_output({"status": "shipped"})
    recorder.end_trace()
    return trace


# ---------------------------------------------------------------------------
# B. SWALLOWED TOOL ERROR (+ controls)
# ---------------------------------------------------------------------------


class _ToolError(Exception):
    pass


def _trace_swallowed_unrelated_activity() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    try:
        with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT):
            raise _ToolError("order service unavailable")
    except _ToolError:
        pass
    with recorder.span("get_shipping_options", kind=SpanKind.TOOL, input={"zip": "10001"}) as span:
        span.set_output({"options": ["standard", "express"]})
    recorder.end_trace()
    return trace


def _trace_swallowed_nested() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span("process_refund_request", kind=SpanKind.CUSTOM):
        try:
            with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT):
                raise _ToolError("order service unavailable")
        except _ToolError:
            pass
    recorder.end_trace()
    return trace


def _trace_swallowed_recovered_control() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    try:
        with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT):
            raise _ToolError("order service unavailable")
    except _ToolError:
        pass
    with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
        span.set_output({"order_id": "ORD-42", "category": "books"})
    recorder.end_trace()
    return trace


def _trace_tool_ok_control() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
        span.set_output({"order_id": "ORD-42", "category": "books"})
    recorder.end_trace()
    return trace


# ---------------------------------------------------------------------------
# C. REPEATED IDENTICAL CALLS (+ controls)
# ---------------------------------------------------------------------------


def _trace_repeated_four_calls() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    for _ in range(4):
        with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
            span.set_output({"order_id": "ORD-42"})
    recorder.end_trace()
    return trace


def _trace_repeated_with_surrounding_calls() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span("get_refund_policy", kind=SpanKind.TOOL, input={"policy": "STANDARD"}) as span:
        span.set_output({"max_days": 30})
    for _ in range(3):
        with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
            span.set_output({"order_id": "ORD-42"})
    with recorder.span("get_shipping_options", kind=SpanKind.TOOL, input={"zip": "10001"}) as span:
        span.set_output({"options": ["standard"]})
    recorder.end_trace()
    return trace


def _trace_repeated_two_calls_control() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    for _ in range(2):
        with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
            span.set_output({"order_id": "ORD-42"})
    recorder.end_trace()
    return trace


# ---------------------------------------------------------------------------
# D. TRACE INCOMPLETENESS (+ control)
# ---------------------------------------------------------------------------


def _trace_incomplete_multiple_dangling() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
        span.set_output({"order_id": "ORD-42"})
    recorder.end_trace()

    orphan_1 = Span(
        id="orphan-1",
        parent_id="never-recorded-a",
        kind=SpanKind.TOOL,
        name="get_shipping_status",
        status=SpanStatus.OK,
        start_time=100.0,
        end_time=100.1,
        recorder_sequence_id=len(trace.spans),
    )
    orphan_2 = Span(
        id="orphan-2",
        parent_id="never-recorded-b",
        kind=SpanKind.TOOL,
        name="get_invoice",
        status=SpanStatus.OK,
        start_time=101.0,
        end_time=101.1,
        recorder_sequence_id=len(trace.spans) + 1,
    )
    trace.add_span(orphan_1)
    trace.add_span(orphan_2)
    return trace


def _trace_valid_parent_control() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span("process_refund_request", kind=SpanKind.CUSTOM):
        with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
            span.set_output({"order_id": "ORD-42"})
    recorder.end_trace()
    return trace


# ---------------------------------------------------------------------------
# E. LOCALIZATION
# ---------------------------------------------------------------------------


def _trace_localization_context_origin() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace(context=TraceContext(initial_state={"refund_eligible": False}))
    with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
        span.set_output({"refund_eligible": False})
    with recorder.span("decide_response", kind=SpanKind.CUSTOM) as span:
        span.set_output({"decision": "denied"})
    recorder.end_trace()
    return trace


def _trace_localization_span_input_first() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span(
        "validate_request", kind=SpanKind.TOOL, input={"refund_eligible": False}
    ) as span:
        span.set_output({"validated": True})
    with recorder.span("decide_response", kind=SpanKind.CUSTOM) as span:
        span.set_output({"decision": "denied"})
    recorder.end_trace()
    return trace


def _trace_localization_chain() -> Trace:
    """Shared 3-span chain used for the multiple-occurrences, SYMPTOM, and
    TERMINAL localization cases: a value is set once (`update_state`), then
    separately echoed under different keys by two downstream spans."""
    recorder = Recorder()
    trace = recorder.start_trace()
    with recorder.span("update_state", kind=SpanKind.STATE) as span:
        span.set_output({"refund_eligible": False})
    with recorder.span("notify_customer", kind=SpanKind.CUSTOM) as span:
        span.set_output({"refund_eligible": False, "notification_status": "refund_denied"})
    with recorder.span("decide_response", kind=SpanKind.CUSTOM) as span:
        span.set_output({"decision": "denied"})
    recorder.end_trace()
    return trace


def _trace_localization_not_observed() -> Trace:
    return _ppa_build_trace(_PPAScenario.CLEAN)


# ---------------------------------------------------------------------------
# F. UNOBSERVABLE / ABSTENTION
# ---------------------------------------------------------------------------


def _trace_langgraph_no_checkpointer() -> Trace:
    _require_langgraph()
    def update_state(state: dict) -> dict:
        return {"refund_eligible": False}

    builder = StateGraph(dict)
    builder.add_node("update_state", update_state, metadata={"span_kind": "STATE"})
    builder.add_edge(START, "update_state")
    builder.add_edge("update_state", END)
    graph = builder.compile()  # deliberately no checkpointer
    return LangGraphAdapter(graph).capture({"refund_eligible": True})


# ---------------------------------------------------------------------------
# G. SELF-CONSISTENT WRONG
# ---------------------------------------------------------------------------


def _trace_self_consistent_wrong_refund() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace(context=TraceContext(initial_state={"refund_eligible": False}))
    with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
        span.set_output({"refund_eligible": False})
    with recorder.span("decide_response", kind=SpanKind.CUSTOM) as span:
        span.set_output({"decision": "denied"})
    recorder.end_trace()
    return trace


def _trace_self_consistent_wrong_category() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace(context=TraceContext(initial_state={"order_category": "excluded_category"}))
    with recorder.span("get_order", kind=SpanKind.TOOL, input=_ORDER_INPUT) as span:
        span.set_output({"order_category": "excluded_category"})
    with recorder.span("decide_response", kind=SpanKind.CUSTOM) as span:
        span.set_output({"decision": "denied"})
    recorder.end_trace()
    return trace


# ---------------------------------------------------------------------------
# H. HELD-OUT
# ---------------------------------------------------------------------------


def _trace_held_out_long_chain_quiet_fault() -> Trace:
    recorder = Recorder()
    trace = recorder.start_trace()
    names = (
        "get_refund_policy",
        "get_order",
        "get_customer_history",
        "get_shipping_options",
        "get_loyalty_status",
        "decide_response",
    )
    for i, name in enumerate(names):
        kind = SpanKind.CUSTOM if name == "decide_response" else SpanKind.TOOL
        with recorder.span(name, kind=kind, input={"step": i}) as span:
            # The last call is silently semantically wrong (eligible=True
            # for a category that should be ineligible) -- no error, no
            # repeat, no dangling reference: structurally invisible to all
            # three existing checks.
            span.set_output({"step": i, "ok": True})
    recorder.end_trace()
    return trace


class _FanOutState(TypedDict, total=False):
    branch_a_decision: str
    branch_b_decision: str
    decision: str


def _trace_held_out_fan_out_inconsistency() -> Trace:
    _require_langgraph()
    def node_a(state: _FanOutState) -> dict:
        return {"branch_a_decision": "approved"}

    def node_b(state: _FanOutState) -> dict:
        return {"branch_b_decision": "denied"}

    def merge(state: _FanOutState) -> dict:
        # Arbitrarily prefers branch A -- the two branches disagreed, and
        # nothing about this is a tool error, a repeated call, or a
        # dangling parent reference.
        return {"decision": state.get("branch_a_decision", "denied")}

    builder = StateGraph(_FanOutState)
    builder.add_node("node_a", node_a, metadata={"span_kind": "CUSTOM"})
    builder.add_node("node_b", node_b, metadata={"span_kind": "CUSTOM"})
    builder.add_node("merge", merge, metadata={"span_kind": "CUSTOM"})
    builder.add_edge(START, "node_a")
    builder.add_edge(START, "node_b")
    builder.add_edge("node_a", "merge")
    builder.add_edge("node_b", "merge")
    builder.add_edge("merge", END)
    graph = builder.compile(checkpointer=MemorySaver())
    return LangGraphAdapter(graph).capture({})


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


def _refund_signal(*, actual: object, expected: object = True) -> FailureSignal:
    return FailureSignal(source="external_evaluator", verdict=ExternalStatus.FAILED, expected=expected, actual=actual)


_CASES: list[BenchmarkCase] = [
    # A. CLEAN
    BenchmarkCase(
        case_id="clean_refund_workflow",
        description="Full refund workflow, nothing fails.",
        category=CaseCategory.CLEAN,
        build_trace=lambda: _ppa_build_trace(_PPAScenario.CLEAN),
        expected_check_findings={
            "swallowed_tool_error": InternalCheckStatus.NO_FINDING,
            "repeated_identical_calls": InternalCheckStatus.NO_FINDING,
            "trace_incompleteness": InternalCheckStatus.NO_FINDING,
        },
    ),
    BenchmarkCase(
        case_id="clean_order_lookup",
        description="A single successful tool call.",
        category=CaseCategory.CLEAN,
        build_trace=_trace_clean_order_lookup,
        expected_check_findings={
            "swallowed_tool_error": InternalCheckStatus.NO_FINDING,
            "repeated_identical_calls": InternalCheckStatus.NO_FINDING,
            "trace_incompleteness": InternalCheckStatus.NO_FINDING,
        },
    ),
    BenchmarkCase(
        case_id="clean_state_transition",
        description="A single successful state update.",
        category=CaseCategory.CLEAN,
        build_trace=_trace_clean_state_transition,
        expected_check_findings={
            "swallowed_tool_error": InternalCheckStatus.NO_FINDING,
            "repeated_identical_calls": InternalCheckStatus.NO_FINDING,
            "trace_incompleteness": InternalCheckStatus.NO_FINDING,
        },
    ),
    # B. SWALLOWED TOOL ERROR
    BenchmarkCase(
        case_id="swallowed_tool_error_no_retry",
        description="Tool error with no retry or fallback.",
        category=CaseCategory.FAULT,
        build_trace=lambda: _ppa_build_trace(_PPAScenario.SWALLOWED_TOOL_ERROR),
        expected_check_findings={"swallowed_tool_error": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="swallowed_tool_error_unrelated_activity",
        description="Tool error followed by an unrelated tool call (not a retry).",
        category=CaseCategory.FAULT,
        build_trace=_trace_swallowed_unrelated_activity,
        expected_check_findings={"swallowed_tool_error": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="swallowed_tool_error_nested",
        description="Tool error inside a nested (parent/child) activity.",
        category=CaseCategory.FAULT,
        build_trace=_trace_swallowed_nested,
        expected_check_findings={"swallowed_tool_error": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="control_tool_ok_not_error",
        description="Control: a successful tool call must not be flagged.",
        category=CaseCategory.NO_FAULT_CONTROL,
        build_trace=_trace_tool_ok_control,
        expected_check_findings={"swallowed_tool_error": InternalCheckStatus.NO_FINDING},
    ),
    BenchmarkCase(
        case_id="control_tool_error_with_matching_retry",
        description="Control: a tool error followed by a matching retry must not be flagged.",
        category=CaseCategory.NO_FAULT_CONTROL,
        build_trace=_trace_swallowed_recovered_control,
        expected_check_findings={"swallowed_tool_error": InternalCheckStatus.NO_FINDING},
    ),
    # C. REPEATED IDENTICAL CALLS
    BenchmarkCase(
        case_id="repeated_calls_exactly_three",
        description="Exactly three consecutive identical tool calls.",
        category=CaseCategory.FAULT,
        build_trace=lambda: _ppa_build_trace(_PPAScenario.REPEATED_TOOL_CALL),
        expected_check_findings={"repeated_identical_calls": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="repeated_calls_exactly_four",
        description="Four consecutive identical tool calls.",
        category=CaseCategory.FAULT,
        build_trace=_trace_repeated_four_calls,
        expected_check_findings={"repeated_identical_calls": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="repeated_calls_with_surrounding_activity",
        description="Three repeated calls surrounded by distinct, unrelated calls.",
        category=CaseCategory.FAULT,
        build_trace=_trace_repeated_with_surrounding_calls,
        expected_check_findings={"repeated_identical_calls": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="control_two_repeated_calls_not_three",
        description="Control: only two repeated calls must not be flagged.",
        category=CaseCategory.NO_FAULT_CONTROL,
        build_trace=_trace_repeated_two_calls_control,
        expected_check_findings={"repeated_identical_calls": InternalCheckStatus.NO_FINDING},
    ),
    # D. TRACE INCOMPLETENESS
    BenchmarkCase(
        case_id="trace_incompleteness_single_dangling",
        description="One span with a dangling parent_id.",
        category=CaseCategory.FAULT,
        build_trace=lambda: _ppa_build_trace(_PPAScenario.INCOMPLETE_TRACE),
        expected_check_findings={"trace_incompleteness": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="trace_incompleteness_multiple_dangling",
        description="Two spans with distinct dangling parent_ids.",
        category=CaseCategory.FAULT,
        build_trace=_trace_incomplete_multiple_dangling,
        expected_check_findings={"trace_incompleteness": InternalCheckStatus.FLAGGED},
    ),
    BenchmarkCase(
        case_id="control_valid_parent_relationships",
        description="Control: valid nested parent/child spans must not be flagged.",
        category=CaseCategory.NO_FAULT_CONTROL,
        build_trace=_trace_valid_parent_control,
        expected_check_findings={"trace_incompleteness": InternalCheckStatus.NO_FINDING},
    ),
    # E. LOCALIZATION
    BenchmarkCase(
        case_id="localization_context_origin",
        description="Target value already present in pre-run context.",
        category=CaseCategory.FAULT,
        build_trace=_trace_localization_context_origin,
        build_failure_signal=lambda: _refund_signal(actual=False),
        localization_target=LocalizationTarget(value=False, field_path="refund_eligible"),
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.ORIGIN,
        reference_span_name=None,  # ground truth: context, not a span
    ),
    BenchmarkCase(
        case_id="localization_span_input_first",
        description="Target value first observed in a span's input.",
        category=CaseCategory.FAULT,
        build_trace=_trace_localization_span_input_first,
        build_failure_signal=lambda: _refund_signal(actual=False),
        localization_target=LocalizationTarget(value=False, field_path="refund_eligible"),
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.FIRST_OBSERVABLE_DEVIATION,
        reference_span_name="validate_request",
    ),
    BenchmarkCase(
        case_id="localization_span_output_origin",
        description="Target value first observed in a span's output (the mutation site).",
        category=CaseCategory.FAULT,
        build_trace=lambda: _ppa_build_trace(_PPAScenario.LOCALIZATION_FAILURE),
        build_failure_signal=lambda: _refund_signal(actual=False),
        localization_target=LocalizationTarget(value=False, field_path="refund_eligible"),
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.ORIGIN,
        reference_span_name="update_refund_state",
    ),
    BenchmarkCase(
        case_id="localization_multiple_occurrences",
        description="Target value redundantly observed in several spans; earliest must win.",
        category=CaseCategory.FAULT,
        build_trace=_trace_localization_chain,
        build_failure_signal=lambda: _refund_signal(actual=False),
        localization_target=LocalizationTarget(value=False, field_path="refund_eligible"),
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.FIRST_OBSERVABLE_DEVIATION,
        reference_span_name="update_state",
    ),
    BenchmarkCase(
        case_id="localization_not_observed",
        description="Target value never appears anywhere in the trace.",
        category=CaseCategory.NO_FAULT_CONTROL,
        build_trace=_trace_localization_not_observed,
        build_failure_signal=lambda: _refund_signal(actual="value-that-never-appears"),
        localization_target=LocalizationTarget(value="value-that-never-appears"),
        expected_localization_status=InternalCheckStatus.NO_FINDING,
        expected_localization_confidence=DiagnosisConfidence.NONE,
        localization_label=LocalizationLabel.NONE,
        reference_span_name=None,
    ),
    BenchmarkCase(
        case_id="localization_symptom_echo",
        description="Target is a downstream re-expression of an already-established value.",
        category=CaseCategory.FAULT,
        build_trace=_trace_localization_chain,
        build_failure_signal=lambda: _refund_signal(actual="refund_denied", expected="refund_approved"),
        localization_target=LocalizationTarget(value="refund_denied", field_path="notification_status"),
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.SYMPTOM,
        reference_span_name="notify_customer",
    ),
    BenchmarkCase(
        case_id="localization_terminal_manifestation",
        description="Target is the final decision span's output.",
        category=CaseCategory.FAULT,
        build_trace=_trace_localization_chain,
        build_failure_signal=lambda: _refund_signal(actual="denied", expected="approved"),
        localization_target=LocalizationTarget(value="denied", field_path="decision"),
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.TERMINAL,
        reference_span_name="decide_response",
    ),
    # F. UNOBSERVABLE / ABSTENTION
    BenchmarkCase(
        case_id="unobservable_check_capability_absent",
        description="A real tool-error pattern exists, but required capabilities are unavailable.",
        category=CaseCategory.UNOBSERVABLE,
        build_trace=lambda: _ppa_build_trace(_PPAScenario.SWALLOWED_TOOL_ERROR),
        expected_check_findings={"swallowed_tool_error": InternalCheckStatus.NOT_APPLICABLE},
        available_capabilities=frozenset(),
        evidence_available=False,
    ),
    BenchmarkCase(
        case_id="unobservable_localization_partial_capability",
        description="Target exists only in span evidence, which is unavailable; context alone can't resolve it.",
        category=CaseCategory.UNOBSERVABLE,
        build_trace=_trace_localization_chain,
        build_failure_signal=lambda: _refund_signal(actual=False),
        localization_target=LocalizationTarget(value=False, field_path="refund_eligible"),
        expected_localization_status=InternalCheckStatus.INCONCLUSIVE,
        localization_label=LocalizationLabel.NONE,
        available_capabilities=frozenset({Capability.PRE_RUN_CONTEXT}),
        evidence_available=False,
    ),
    BenchmarkCase(
        case_id="unobservable_langgraph_no_checkpointer",
        description="Real LangGraph capability degradation: no checkpointer, so span evidence is unavailable.",
        category=CaseCategory.UNOBSERVABLE,
        build_trace=_trace_langgraph_no_checkpointer,
        build_failure_signal=lambda: _refund_signal(actual=False),
        localization_target=LocalizationTarget(value=False, field_path="refund_eligible"),
        expected_localization_status=InternalCheckStatus.INCONCLUSIVE,
        localization_label=LocalizationLabel.NONE,
        available_capabilities=frozenset({Capability.PRE_RUN_CONTEXT}),
        evidence_available=False,
        tags=(LANGGRAPH_TAG,),
    ),
    # G. SELF-CONSISTENT WRONG
    BenchmarkCase(
        case_id="self_consistent_wrong_refund",
        description="Internally consistent trace; external evaluator says the outcome is wrong.",
        category=CaseCategory.FAULT,
        build_trace=_trace_self_consistent_wrong_refund,
        build_failure_signal=lambda: _refund_signal(actual=False),
        localization_target=LocalizationTarget(value=False, field_path="refund_eligible"),
        expected_check_findings={
            "swallowed_tool_error": InternalCheckStatus.NO_FINDING,
            "repeated_identical_calls": InternalCheckStatus.NO_FINDING,
            "trace_incompleteness": InternalCheckStatus.NO_FINDING,
        },
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.ORIGIN,
        reference_span_name=None,
        tags=("self_consistent_wrong",),
    ),
    BenchmarkCase(
        case_id="self_consistent_wrong_category",
        description="Internally consistent category-based denial; external evaluator disagrees.",
        category=CaseCategory.FAULT,
        build_trace=_trace_self_consistent_wrong_category,
        build_failure_signal=lambda: _refund_signal(
            actual="excluded_category", expected="eligible_category"
        ),
        localization_target=LocalizationTarget(value="excluded_category", field_path="order_category"),
        expected_check_findings={
            "swallowed_tool_error": InternalCheckStatus.NO_FINDING,
            "repeated_identical_calls": InternalCheckStatus.NO_FINDING,
            "trace_incompleteness": InternalCheckStatus.NO_FINDING,
        },
        expected_localization_status=InternalCheckStatus.FLAGGED,
        expected_localization_confidence=DiagnosisConfidence.UNIQUE,
        localization_label=LocalizationLabel.ORIGIN,
        reference_span_name=None,
        tags=("self_consistent_wrong",),
    ),
    # H. HELD-OUT
    BenchmarkCase(
        case_id="held_out_long_chain_quiet_fault",
        description="A silently wrong value at the end of a long chain of distinct, successful calls.",
        category=CaseCategory.HELD_OUT,
        build_trace=_trace_held_out_long_chain_quiet_fault,
        expected_check_findings={
            "swallowed_tool_error": InternalCheckStatus.NO_FINDING,
            "repeated_identical_calls": InternalCheckStatus.NO_FINDING,
            "trace_incompleteness": InternalCheckStatus.NO_FINDING,
        },
    ),
    BenchmarkCase(
        case_id="held_out_fan_out_inconsistency",
        description="Two parallel branches disagree; no existing check is designed to detect this.",
        category=CaseCategory.HELD_OUT,
        build_trace=_trace_held_out_fan_out_inconsistency,
        expected_check_findings={
            "swallowed_tool_error": InternalCheckStatus.NO_FINDING,
            "repeated_identical_calls": InternalCheckStatus.NO_FINDING,
            "trace_incompleteness": InternalCheckStatus.NO_FINDING,
        },
        tags=(LANGGRAPH_TAG,),
    ),
]


def build_corpus(*, include_langgraph_cases: bool = True) -> tuple[BenchmarkCase, ...]:
    """Return the deterministic benchmark corpus.

    By default this is the FULL corpus (`FULL_CORPUS_SIZE` cases), which
    includes the cases captured through the real LangGraph integration and
    therefore requires the optional `langgraph` extra. If LangGraph is not
    installed this raises `BenchmarkDependencyError` rather than quietly
    returning a smaller corpus.

    Pass `include_langgraph_cases=False` to explicitly request the
    core-only subset (`CORE_CORPUS_SIZE` cases, no LangGraph needed).
    """
    if include_langgraph_cases:
        if not _LANGGRAPH_AVAILABLE:
            raise BenchmarkDependencyError(
                f"The full {FULL_CORPUS_SIZE}-case benchmark corpus needs the optional "
                "'langgraph' extra: pip install '.[langgraph]'. To run only the "
                f"{CORE_CORPUS_SIZE} cases that need no LangGraph, request the core-only "
                "subset explicitly (python -m benchmark --core-only)."
            )
        return tuple(_CASES)
    return tuple(c for c in _CASES if LANGGRAPH_TAG not in c.tags)
