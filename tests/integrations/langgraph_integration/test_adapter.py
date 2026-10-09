"""Tests for LangGraphAdapter against real compiled LangGraph graphs.

Requires `langgraph` to be importable -- run with the interpreter that has
it installed (same convention as `tests/test_langgraph_adapter.py`).
"""

from __future__ import annotations

import operator
import unittest
from typing import Annotated, TypedDict

from agentdebug.analysis.checks import ALL_CHECKS
from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer, LocalizationTarget
from agentdebug.core.enums import (
    Capability,
    DiagnosisConfidence,
    ExternalStatus,
    InternalCheckStatus,
    SpanKind,
    SpanStatus,
)
from agentdebug.core.failure_signal import FailureSignal

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"langgraph not importable, cannot test LangGraphAdapter: {exc}")

from agentdebug.integrations.langgraph import LangGraphAdapter


class _LinearState(TypedDict):
    refund_eligible: bool


def _build_linear_graph(*, checkpointer: bool = True, tool_fails: bool = False):
    def get_order(state: _LinearState) -> dict:
        if tool_fails:
            raise RuntimeError("order service unavailable")
        return {"refund_eligible": state["refund_eligible"]}

    def update_state(state: _LinearState) -> dict:
        return {"refund_eligible": False}

    builder = StateGraph(_LinearState)
    builder.add_node("get_order", get_order, metadata={"span_kind": "TOOL"})
    builder.add_node("update_state", update_state, metadata={"span_kind": "STATE"})
    builder.add_edge(START, "get_order")
    builder.add_edge("get_order", "update_state")
    builder.add_edge("update_state", END)
    return builder.compile(checkpointer=MemorySaver() if checkpointer else None)


class _FanOutState(TypedDict):
    log: Annotated[list[str], operator.add]


def _build_fan_out_graph():
    def node_a(state: _FanOutState) -> dict:
        return {"log": ["a"]}

    def node_b(state: _FanOutState) -> dict:
        return {"log": ["b"]}

    def node_join(state: _FanOutState) -> dict:
        return {"log": ["join"]}

    builder = StateGraph(_FanOutState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_join", node_join)
    builder.add_edge(START, "node_a")
    builder.add_edge(START, "node_b")
    builder.add_edge("node_a", "node_join")
    builder.add_edge("node_b", "node_join")
    builder.add_edge("node_join", END)
    return builder.compile(checkpointer=MemorySaver())


class LinearGraphTests(unittest.TestCase):
    """A. Simple linear graph."""

    def test_trace_exists_with_expected_spans_in_deterministic_order(self) -> None:
        graph = _build_linear_graph()
        adapter = LangGraphAdapter(graph)
        trace = adapter.capture({"refund_eligible": True})

        self.assertEqual(len(trace.spans), 2)
        names = [s.name for s in trace.spans]
        self.assertEqual(names, ["get_order", "update_state"])
        seqs = [s.recorder_sequence_id for s in trace.spans]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))

    def test_declared_node_kinds_are_used(self) -> None:
        graph = _build_linear_graph()
        trace = LangGraphAdapter(graph).capture({"refund_eligible": True})
        kinds = {s.name: s.kind for s in trace.spans}
        self.assertEqual(kinds["get_order"], SpanKind.TOOL)
        self.assertEqual(kinds["update_state"], SpanKind.STATE)

    def test_undeclared_node_defaults_to_custom(self) -> None:
        def plain_node(state: _LinearState) -> dict:
            return {"refund_eligible": state["refund_eligible"]}

        builder = StateGraph(_LinearState)
        builder.add_node("plain_node", plain_node)
        builder.add_edge(START, "plain_node")
        builder.add_edge("plain_node", END)
        graph = builder.compile(checkpointer=MemorySaver())

        trace = LangGraphAdapter(graph).capture({"refund_eligible": True})
        self.assertEqual(trace.spans[0].kind, SpanKind.CUSTOM)

    def test_flat_graph_spans_have_no_parent(self) -> None:
        graph = _build_linear_graph()
        trace = LangGraphAdapter(graph).capture({"refund_eligible": True})
        for span in trace.spans:
            self.assertIsNone(span.parent_id)

    def test_pre_run_context_captures_initial_input(self) -> None:
        graph = _build_linear_graph()
        trace = LangGraphAdapter(graph).capture({"refund_eligible": True})
        self.assertEqual(trace.context.initial_state, {"refund_eligible": True})


class StateUpdateGraphTests(unittest.TestCase):
    """B. State update graph."""

    def test_state_evidence_captured_in_span_output(self) -> None:
        graph = _build_linear_graph()
        trace = LangGraphAdapter(graph).capture({"refund_eligible": True})
        update_span = next(s for s in trace.spans if s.name == "update_state")
        self.assertEqual(update_span.output, {"refund_eligible": False})
        # Honestly derived from the prior checkpoint, not fabricated.
        self.assertEqual(update_span.input, {"refund_eligible": True})

    def test_localization_finds_the_changed_value(self) -> None:
        graph = _build_linear_graph()
        adapter = LangGraphAdapter(graph)
        trace = adapter.capture({"refund_eligible": True})

        signal = FailureSignal(source="external_evaluator", verdict=ExternalStatus.FAILED, expected=True, actual=False)
        target = LocalizationTarget(value=False, field_path="refund_eligible")
        finding = FirstObservedOccurrenceAnalyzer().run(trace, signal, target=target)

        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.diagnosis_confidence, DiagnosisConfidence.UNIQUE)
        update_span = next(s for s in trace.spans if s.name == "update_state")
        self.assertEqual(finding.claims[0].span_ids, [update_span.id])
        self.assertEqual(finding.triggering_signal_id, signal.id)
        # Observational only -- never a causal claim about update_state.
        self.assertNotIn("caused", str(finding.claims[0].fields).lower())


class ToolErrorScenarioTests(unittest.TestCase):
    """C. Tool/error scenario.

    `get_order` is declared TOOL via `add_node(metadata={"span_kind": "TOOL"})`
    (a real LangGraph API, confirmed in the adapter's module docstring);
    when it raises, the resulting evidence is a TOOL-kind, ERROR-status
    span, which `SwallowedToolErrorCheck` can see without any
    LangGraph-specific version of that check.
    """

    def test_failing_tool_node_produces_tool_error_span(self) -> None:
        graph = _build_linear_graph(tool_fails=True)
        adapter = LangGraphAdapter(graph)
        with self.assertRaises(RuntimeError):
            adapter.capture({"refund_eligible": True}, trace_id="tool-error-trace")

        trace = adapter.get_trace("tool-error-trace")
        error_span = next(s for s in trace.spans if s.name == "get_order")
        self.assertEqual(error_span.kind, SpanKind.TOOL)
        self.assertEqual(error_span.status, SpanStatus.ERROR)

    def test_swallowed_tool_error_check_flags_it_unchanged(self) -> None:
        graph = _build_linear_graph(tool_fails=True)
        adapter = LangGraphAdapter(graph)
        with self.assertRaises(RuntimeError):
            adapter.capture({"refund_eligible": True}, trace_id="tool-error-trace-2")
        trace = adapter.get_trace("tool-error-trace-2")

        findings = {check.definition.id: check.run(trace) for check in ALL_CHECKS}
        self.assertEqual(findings["swallowed_tool_error"].status, InternalCheckStatus.FLAGGED)


class FanOutGraphTests(unittest.TestCase):
    """D. Fan-out graph: node_a and node_b must not collapse into one span."""

    def test_both_branches_produce_independent_spans(self) -> None:
        graph = _build_fan_out_graph()
        trace = LangGraphAdapter(graph).capture({"log": []})

        names = {s.name for s in trace.spans}
        self.assertEqual(names, {"node_a", "node_b", "node_join"})

    def test_fan_out_branches_share_checkpoint_attribution_honestly(self) -> None:
        graph = _build_fan_out_graph()
        trace = LangGraphAdapter(graph).capture({"log": []})

        node_a = next(s for s in trace.spans if s.name == "node_a")
        node_b = next(s for s in trace.spans if s.name == "node_b")
        node_join = next(s for s in trace.spans if s.name == "node_join")

        self.assertEqual(
            node_a.metadata["langgraph_checkpoint_id"], node_b.metadata["langgraph_checkpoint_id"]
        )
        self.assertNotEqual(
            node_a.metadata["langgraph_checkpoint_id"], node_join.metadata["langgraph_checkpoint_id"]
        )


class ExceptionScenarioTests(unittest.TestCase):
    """E. A node raises -- partial trace retained, ERROR evidence exists,
    original exception re-raised unchanged."""

    def _build_failing_graph(self):
        class _S(TypedDict):
            x: int

        def node_a(state: _S) -> dict:
            return {"x": state["x"] + 1}

        def node_b(state: _S) -> dict:
            raise RuntimeError("boom in node_b")

        builder = StateGraph(_S)
        builder.add_node("node_a", node_a)
        builder.add_node("node_b", node_b)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", "node_b")
        builder.add_edge("node_b", END)
        return builder.compile(checkpointer=MemorySaver())

    def test_original_exception_is_reraised_unchanged(self) -> None:
        adapter = LangGraphAdapter(self._build_failing_graph())
        with self.assertRaises(RuntimeError) as ctx:
            adapter.capture({"x": 0}, trace_id="exc-trace")
        self.assertEqual(str(ctx.exception), "boom in node_b")

    def test_partial_trace_is_retained_and_retrievable(self) -> None:
        adapter = LangGraphAdapter(self._build_failing_graph())
        with self.assertRaises(RuntimeError):
            adapter.capture({"x": 0}, trace_id="exc-trace-2")

        trace = adapter.get_trace("exc-trace-2")
        names = [s.name for s in trace.spans]
        self.assertEqual(names, ["node_a", "node_b"])
        self.assertEqual(trace.spans[0].status, SpanStatus.OK)
        self.assertEqual(trace.spans[1].status, SpanStatus.ERROR)
        self.assertEqual(trace.spans[1].metadata["exception"]["type"], "RuntimeError")
        self.assertEqual(trace.spans[1].metadata["exception"]["message"], "boom in node_b")
        self.assertEqual(trace.spans[1].metadata["attributed_nodes"], ["node_b"])


class ExternalFailureSignalTests(unittest.TestCase):
    """F. External FailureSignal + localization."""

    def test_failure_signal_is_external_and_triggering_id_matches(self) -> None:
        graph = _build_linear_graph()
        trace = LangGraphAdapter(graph).capture({"refund_eligible": True})

        # The adapter never invents this -- it is constructed entirely by
        # the caller, independent of the trace/adapter.
        signal = FailureSignal(source="external_evaluator", verdict=ExternalStatus.FAILED, expected=True, actual=False)
        target = LocalizationTarget(value=False, field_path="refund_eligible")
        finding = FirstObservedOccurrenceAnalyzer().run(trace, signal, target=target)

        self.assertEqual(finding.triggering_signal_id, signal.id)


class TraceCheckCompatibilityTests(unittest.TestCase):
    """Section 16: existing Phase 3 checks run unmodified against a
    LangGraph-generated Trace."""

    def test_all_checks_run_without_modification(self) -> None:
        graph = _build_linear_graph()
        trace = LangGraphAdapter(graph).capture({"refund_eligible": True})
        for check in ALL_CHECKS:
            finding = check.run(trace)
            self.assertEqual(finding.check_id, check.definition.id)


class DeterminismTests(unittest.TestCase):
    """Section 18: repeated runs of the same deterministic scenario produce
    the same structural trace. UUIDs are not required to match."""

    def test_repeated_capture_is_structurally_identical(self) -> None:
        def structural_view(trace):
            return [
                (s.name, s.kind.value, s.input, s.output, s.parent_id)
                for s in sorted(trace.spans, key=lambda s: s.recorder_sequence_id)
            ]

        trace1 = LangGraphAdapter(_build_linear_graph()).capture({"refund_eligible": True})
        trace2 = LangGraphAdapter(_build_linear_graph()).capture({"refund_eligible": True})

        self.assertEqual(structural_view(trace1), structural_view(trace2))
        self.assertNotEqual(trace1.id, trace2.id)  # identity fields are allowed to differ


class CapabilityTests(unittest.TestCase):
    """Section 19: a capability that genuinely cannot be provided must not
    be falsely advertised, and downstream NOT_APPLICABLE/INCONCLUSIVE
    behavior must follow the existing Phase 3/4 semantics -- not a
    manufactured INCONCLUSIVE."""

    def test_capabilities_with_checkpointer(self) -> None:
        adapter = LangGraphAdapter(_build_linear_graph())
        self.assertEqual(
            adapter.capabilities,
            frozenset({Capability.PRE_RUN_CONTEXT, Capability.SPAN_IO, Capability.TOOL_STATUS, Capability.TOOL_CALLS}),
        )

    def test_capabilities_without_checkpointer_are_honestly_reduced(self) -> None:
        adapter = LangGraphAdapter(_build_linear_graph(checkpointer=False))
        self.assertEqual(adapter.capabilities, frozenset({Capability.PRE_RUN_CONTEXT}))

    def test_trace_check_is_not_applicable_when_capability_genuinely_absent(self) -> None:
        graph = _build_linear_graph(checkpointer=False)
        adapter = LangGraphAdapter(graph)
        trace = adapter.capture({"refund_eligible": True})

        from agentdebug.analysis.checks import SwallowedToolErrorCheck

        finding = SwallowedToolErrorCheck().run(trace, available_capabilities=adapter.capabilities)
        self.assertEqual(finding.status, InternalCheckStatus.NOT_APPLICABLE)
        self.assertEqual(
            set(finding.capability_gaps), {Capability.TOOL_STATUS, Capability.TOOL_CALLS}
        )

    def test_localization_is_inconclusive_not_manufactured_when_span_io_missing(self) -> None:
        # Without a checkpointer, span.input is never captured, so the
        # target (only ever visible via a node's *input*, never any
        # output) cannot be found in what this capture exposes -- but it
        # also isn't confidently "not observed", since SPAN_IO evidence is
        # genuinely missing, not merely absent of this value.
        def echo_input(state: _LinearState) -> dict:
            return {}

        builder = StateGraph(_LinearState)
        builder.add_node("echo_input", echo_input)
        builder.add_edge(START, "echo_input")
        builder.add_edge("echo_input", END)
        graph = builder.compile()  # no checkpointer

        adapter = LangGraphAdapter(graph)
        trace = adapter.capture({"refund_eligible": True})
        # Confirm the target truly isn't in context or in what little
        # evidence we do have -- the gap is genuine, not contrived.
        self.assertNotIn("some-target-not-in-context", str(trace.context.initial_state))

        signal = FailureSignal(source="external_evaluator", verdict=ExternalStatus.FAILED, actual="some-target-not-in-context")
        target = LocalizationTarget(value="some-target-not-in-context")
        finding = FirstObservedOccurrenceAnalyzer().run(
            trace, signal, target=target, available_capabilities=adapter.capabilities
        )
        self.assertEqual(finding.status, InternalCheckStatus.INCONCLUSIVE)
        self.assertEqual(finding.capability_gaps, [Capability.SPAN_IO])


if __name__ == "__main__":
    unittest.main()
