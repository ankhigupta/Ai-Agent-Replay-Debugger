"""End-to-end evidence tests.

    Recorder -> Trace -> FailureSignal -> FirstObservedOccurrence -> Finding
        -> JsonlEvidenceSink -> JSONL file -> Trace.from_dict -> same evidence

Plus a small LangGraph-compatibility check (section 20): the sink has no
idea where a Trace came from, so a LangGraph-produced Trace must round
-trip identically to a plain-Python one.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer, LocalizationTarget
from agentdebug.core.context import TraceContext
from agentdebug.core.enums import ExternalStatus, SpanKind
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.trace import Trace
from agentdebug.evidence.jsonl import JsonlEvidenceSink
from agentdebug.recording.recorder import Recorder

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    from agentdebug.integrations.langgraph import LangGraphAdapter

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False


class _EvidenceTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = Path(self._tmpdir.name) / "runs.jsonl"


class RecorderToSinkIntegrationTests(_EvidenceTestBase):
    def test_full_pipeline_round_trip(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace(context=TraceContext(initial_state={"refund_eligible": True}))
        with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}) as span:
            span.set_output({"refund_eligible": True})
        with recorder.span("update_state", kind=SpanKind.STATE) as span:
            span.set_output({"refund_eligible": False})
        recorder.end_trace()

        failure_signal = FailureSignal(
            source="external_evaluator", verdict=ExternalStatus.FAILED, expected=True, actual=False
        )
        target = LocalizationTarget(value=False, field_path="refund_eligible")
        finding = FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)

        trace.add_failure_signal(failure_signal)
        trace.add_finding(finding)

        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)

        line = self.path.read_text(encoding="utf-8").splitlines()[0]
        restored = Trace.from_dict(json.loads(line))

        self.assertEqual({s.name for s in restored.spans}, {"get_order", "update_state"})
        self.assertEqual(len(restored.failure_signals), 1)
        self.assertEqual(restored.failure_signals[0].id, failure_signal.id)
        self.assertEqual(len(restored.findings), 1)
        restored_finding = restored.findings[0]
        self.assertEqual(restored_finding.triggering_signal_id, failure_signal.id)
        self.assertEqual(restored_finding.claims[0].fields["target_value"], False)
        self.assertEqual(restored, trace)


@unittest.skipUnless(_LANGGRAPH_AVAILABLE, "langgraph not importable")
class LangGraphSinkCompatibilityTests(_EvidenceTestBase):
    def test_langgraph_trace_round_trips_like_any_other_trace(self) -> None:
        def node_a(state: dict) -> dict:
            return {"x": state["x"] + 1}

        builder = StateGraph(dict)
        builder.add_node("node_a", node_a)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", END)
        graph = builder.compile(checkpointer=MemorySaver())

        trace = LangGraphAdapter(graph).capture({"x": 0})

        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)

        line = self.path.read_text(encoding="utf-8").splitlines()[0]
        restored = Trace.from_dict(json.loads(line))
        self.assertEqual(restored, trace)


if __name__ == "__main__":
    unittest.main()
