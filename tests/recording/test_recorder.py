"""Tests for Recorder: span/trace lifecycle, parenting, errors, concurrency."""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import unittest
import uuid
from unittest import mock

from agentdebug.core.enums import SpanKind, SpanStatus
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.serialization import is_json_serializable
from agentdebug.recording import context
from agentdebug.recording.recorder import Recorder, RecorderError


class RecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        context.clear()

    # -- trace lifecycle --------------------------------------------------

    def test_start_trace_returns_trace(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        self.assertIsNotNone(trace.id)

    def test_current_trace_available_after_start(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        self.assertIs(recorder.current_trace(), trace)

    def test_no_current_trace_before_start(self) -> None:
        recorder = Recorder()
        self.assertIsNone(recorder.current_trace())

    def test_end_trace_clears_current_and_returns_trace(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        ended = recorder.end_trace()
        self.assertIs(ended, trace)
        self.assertIsNone(recorder.current_trace())

    def test_end_trace_without_active_trace_raises(self) -> None:
        recorder = Recorder()
        with self.assertRaises(RecorderError):
            recorder.end_trace()

    def test_span_without_active_trace_raises(self) -> None:
        recorder = Recorder()
        with self.assertRaises(RecorderError):
            with recorder.span("s", kind=SpanKind.TOOL):
                pass

    # -- basic span creation ------------------------------------------------

    def test_span_basic_adds_one_completed_span(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("get_order", kind=SpanKind.TOOL):
            pass
        self.assertEqual(len(trace.spans), 1)

    def test_span_ids_are_unique(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("a", kind=SpanKind.TOOL):
            pass
        with recorder.span("b", kind=SpanKind.TOOL):
            pass
        ids = [s.id for s in trace.spans]
        self.assertEqual(len(ids), len(set(ids)))

    def test_kind_and_name_captured(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("get_order", kind=SpanKind.TOOL):
            pass
        s = trace.spans[0]
        self.assertEqual(s.name, "get_order")
        self.assertEqual(s.kind, SpanKind.TOOL)

    def test_input_and_output_captured(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}) as span:
            span.set_output({"refund_eligible": True})
        s = trace.spans[0]
        self.assertEqual(s.input, {"order_id": "ORD-42"})
        self.assertEqual(s.output, {"refund_eligible": True})

    def test_output_defaults_to_none_when_not_set(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("s", kind=SpanKind.TOOL):
            pass
        self.assertIsNone(trace.spans[0].output)

    def test_metadata_captured(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("s", kind=SpanKind.TOOL, metadata={"model": "gpt-4o"}):
            pass
        self.assertEqual(trace.spans[0].metadata["model"], "gpt-4o")

    def test_start_end_time_and_duration(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("s", kind=SpanKind.TOOL):
            pass
        s = trace.spans[0]
        self.assertGreaterEqual(s.end_time, s.start_time)
        self.assertAlmostEqual(s.duration, s.end_time - s.start_time)

    def test_span_added_only_after_completion(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("s", kind=SpanKind.TOOL):
            self.assertEqual(len(trace.spans), 0)
        self.assertEqual(len(trace.spans), 1)

    def test_completed_span_is_immutable(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("s", kind=SpanKind.TOOL):
            pass
        span = trace.spans[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            span.name = "renamed"  # type: ignore[misc]

    # -- sequence ids ---------------------------------------------------

    def test_sequence_id_starts_at_zero(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("s1", kind=SpanKind.TOOL):
            pass
        self.assertEqual(trace.spans[0].recorder_sequence_id, 0)

    def test_sequence_ids_increment_monotonically(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("s1", kind=SpanKind.TOOL):
            pass
        with recorder.span("s2", kind=SpanKind.TOOL):
            pass
        with recorder.span("s3", kind=SpanKind.TOOL):
            pass
        self.assertEqual([s.recorder_sequence_id for s in trace.spans], [0, 1, 2])

    # -- parent/child -----------------------------------------------------

    def test_nested_spans_receive_correct_parent_id(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("parent", kind=SpanKind.CUSTOM) as parent:
            with recorder.span("child", kind=SpanKind.TOOL) as child:
                pass
        parent_span = trace.get_span(parent.id)
        child_span = trace.get_span(child.id)
        self.assertIsNone(parent_span.parent_id)
        self.assertEqual(child_span.parent_id, parent_span.id)

    def test_parent_restored_after_child_finishes(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("parent", kind=SpanKind.CUSTOM) as parent:
            with recorder.span("child", kind=SpanKind.TOOL):
                pass
            with recorder.span("sibling", kind=SpanKind.TOOL) as sibling:
                pass
        sibling_span = trace.get_span(sibling.id)
        self.assertEqual(sibling_span.parent_id, parent.id)

    def test_active_span_context_empty_after_outermost_span(self) -> None:
        recorder = Recorder()
        recorder.start_trace()
        with recorder.span("s", kind=SpanKind.TOOL):
            pass
        self.assertIsNone(context.get_current_span_id())

    def test_normal_nesting_never_produces_dangling_parent_id(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("parent", kind=SpanKind.CUSTOM):
            with recorder.span("child", kind=SpanKind.TOOL):
                pass
        for span in trace.spans:
            if span.parent_id is not None:
                self.assertIsNotNone(trace.get_span(span.parent_id))

    # -- exceptions -------------------------------------------------------

    def test_exception_creates_error_span(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        captured_id = None
        with self.assertRaises(ValueError):
            with recorder.span("failing", kind=SpanKind.TOOL) as span:
                captured_id = span.id
                raise ValueError("bad tool response")
        failed_span = trace.get_span(captured_id)
        self.assertIsNotNone(failed_span)
        self.assertEqual(failed_span.status, SpanStatus.ERROR)

    def test_original_exception_is_reraised_unchanged(self) -> None:
        recorder = Recorder()
        recorder.start_trace()
        with self.assertRaises(ValueError) as ctx:
            with recorder.span("failing", kind=SpanKind.TOOL):
                raise ValueError("bad tool response")
        self.assertEqual(str(ctx.exception), "bad tool response")

    def test_exception_metadata_is_json_serializable_and_descriptive(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        captured_id = None
        with self.assertRaises(ValueError):
            with recorder.span("failing", kind=SpanKind.TOOL) as span:
                captured_id = span.id
                raise ValueError("bad tool response")
        failed_span = trace.get_span(captured_id)
        self.assertTrue(is_json_serializable(failed_span.metadata))
        self.assertEqual(failed_span.metadata["exception"]["type"], "ValueError")
        self.assertEqual(failed_span.metadata["exception"]["message"], "bad tool response")

    def test_failed_span_is_still_recorded_in_trace(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with self.assertRaises(RuntimeError):
            with recorder.span("failing", kind=SpanKind.TOOL):
                raise RuntimeError("boom")
        self.assertEqual(len(trace.spans), 1)

    def test_active_span_context_restored_after_exception(self) -> None:
        recorder = Recorder()
        recorder.start_trace()
        with self.assertRaises(ValueError):
            with recorder.span("failing", kind=SpanKind.TOOL):
                raise ValueError("boom")
        self.assertIsNone(context.get_current_span_id())

    def test_nested_exception_only_marks_inner_span_as_error(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        with recorder.span("outer", kind=SpanKind.CUSTOM) as outer:
            with self.assertRaises(ValueError):
                with recorder.span("inner", kind=SpanKind.TOOL) as inner:
                    raise ValueError("boom")
        outer_span = trace.get_span(outer.id)
        inner_span = trace.get_span(inner.id)
        self.assertEqual(outer_span.status, SpanStatus.OK)
        self.assertEqual(inner_span.status, SpanStatus.ERROR)

    # -- failure signals ----------------------------------------------------

    def test_add_failure_signal_attaches_to_active_trace(self) -> None:
        from agentdebug.core.enums import ExternalStatus
        from agentdebug.core.failure_signal import FailureSignal

        recorder = Recorder()
        trace = recorder.start_trace()
        recorder.add_failure_signal(FailureSignal(source="pytest", verdict=ExternalStatus.FAILED))
        self.assertEqual(len(trace.failure_signals), 1)

    def test_add_failure_signal_without_active_trace_raises(self) -> None:
        from agentdebug.core.enums import ExternalStatus
        from agentdebug.core.failure_signal import FailureSignal

        recorder = Recorder()
        with self.assertRaises(RecorderError):
            recorder.add_failure_signal(FailureSignal(source="pytest", verdict=ExternalStatus.FAILED))

    # -- mutation safety --------------------------------------------------

    def test_add_span_failure_does_not_leave_partial_state(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace()
        fixed_uuid = uuid.UUID(int=0)
        with mock.patch("agentdebug.recording.recorder.uuid.uuid4", return_value=fixed_uuid):
            with recorder.span("first", kind=SpanKind.TOOL):
                pass
            with self.assertRaises(TraceValidationError):
                with recorder.span("second", kind=SpanKind.TOOL):
                    pass
        self.assertEqual(len(trace.spans), 1)
        # Context must still be restored even though add_span raised.
        self.assertIsNone(context.get_current_span_id())


class RecorderThreadSafetyTests(unittest.TestCase):
    def test_concurrent_threads_receive_unique_sequence_ids(self) -> None:
        recorder = Recorder()
        all_seq_ids: list[int] = []
        collect_lock = threading.Lock()
        spans_per_thread = 20
        thread_count = 5

        def worker() -> None:
            trace = recorder.start_trace()
            for i in range(spans_per_thread):
                with recorder.span(f"s{i}", kind=SpanKind.TOOL):
                    pass
            with collect_lock:
                all_seq_ids.extend(s.recorder_sequence_id for s in trace.spans)

        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = spans_per_thread * thread_count
        self.assertEqual(len(all_seq_ids), expected_total)
        self.assertEqual(len(set(all_seq_ids)), expected_total)


class RecorderAsyncConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_tasks_no_parent_contamination(self) -> None:
        context.clear()
        recorder = Recorder()
        recorder.start_trace()

        async def task_a() -> tuple[str, str]:
            with recorder.span("task_a", kind=SpanKind.CUSTOM) as span_a:
                await asyncio.sleep(0)
                with recorder.span("task_a_child", kind=SpanKind.TOOL) as child:
                    await asyncio.sleep(0)
                return span_a.id, child.id

        async def task_b() -> tuple[str, str]:
            with recorder.span("task_b", kind=SpanKind.CUSTOM) as span_b:
                await asyncio.sleep(0)
                with recorder.span("task_b_child", kind=SpanKind.TOOL) as child:
                    await asyncio.sleep(0)
                return span_b.id, child.id

        with recorder.span("root", kind=SpanKind.CUSTOM) as root:
            root_id = root.id
            (a_id, a_child_id), (b_id, b_child_id) = await asyncio.gather(task_a(), task_b())

        trace = recorder.current_trace()
        self.assertEqual(trace.get_span(a_id).parent_id, root_id)
        self.assertEqual(trace.get_span(b_id).parent_id, root_id)
        self.assertEqual(trace.get_span(a_child_id).parent_id, a_id)
        self.assertEqual(trace.get_span(b_child_id).parent_id, b_id)
        self.assertIsNone(context.get_current_span_id())


if __name__ == "__main__":
    unittest.main()
