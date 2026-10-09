"""Integration tests: Recorder -> real Trace -> TraceCheck -> Finding.

Demonstrates that Phase 2 recording and Phase 3 analysis actually connect,
using real `Recorder`-produced `Trace`/`Span` objects rather than
hand-built ones for every case.
"""

from __future__ import annotations

import unittest

from agentdebug.analysis.checks.repeated_identical_calls import RepeatedIdenticalCallsCheck
from agentdebug.analysis.checks.swallowed_tool_error import SwallowedToolErrorCheck
from agentdebug.analysis.checks.trace_incompleteness import TraceIncompletenessCheck
from agentdebug.core.enums import InternalCheckStatus, SpanKind, SpanStatus
from agentdebug.core.span import Span
from agentdebug.recording.recorder import Recorder


class RecorderIntegrationTests(unittest.TestCase):
    def test_recorder_captured_failed_tool_is_flagged_by_swallowed_tool_error(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with self.assertRaises(RuntimeError):
            with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}):
                raise RuntimeError("tool backend unavailable")

        finding = SwallowedToolErrorCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(len(finding.claims), 1)
        self.assertEqual(finding.claims[0].fields["tool_span_id"], trace.spans[0].id)
        self.assertEqual(trace.spans[0].status, SpanStatus.ERROR)

    def test_recorder_captured_repeated_tool_calls_is_flagged(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        for _ in range(3):
            with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}) as span:
                span.set_output({"status": "retrying"})

        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.claims[0].fields["occurrence_count"], 3)

    def test_dangling_parent_alongside_recorder_spans_is_flagged(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("agent", kind=SpanKind.CUSTOM):
            with recorder.span("get_order", kind=SpanKind.TOOL):
                pass

        # Deliberately construct one additional span with a dangling
        # parent_id -- Recorder itself never produces one (Phase 2), so
        # this models evidence that arrived some other way (e.g. a
        # partially-lost sub-trace) without modifying the Recorder.
        orphan = Span(
            id="orphan",
            parent_id="never-recorded",
            kind=SpanKind.TOOL,
            name="get_weather",
            status=SpanStatus.OK,
            start_time=100.0,
            end_time=100.1,
            recorder_sequence_id=999,
        )
        trace.add_span(orphan)

        finding = TraceIncompletenessCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(
            finding.claims[0].fields, {"span_id": "orphan", "missing_parent_id": "never-recorded"}
        )


if __name__ == "__main__":
    unittest.main()
