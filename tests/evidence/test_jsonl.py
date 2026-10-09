"""Tests for JsonlEvidenceSink. Letters refer to the Phase 6 spec's test list."""

from __future__ import annotations

import inspect
import json
import tempfile
import threading
import unittest
from pathlib import Path

from agentdebug.core.claim import Claim
from agentdebug.core.enums import ClaimType, ExternalStatus, InternalCheckStatus, SpanKind, SpanStatus
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.serialization import SCHEMA_VERSION, to_json
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace
from agentdebug.evidence.jsonl import JsonlEvidenceSink
from agentdebug.evidence.sink import EvidenceSinkError


def _make_trace(trace_id: str) -> Trace:
    span = Span(
        id=f"{trace_id}-span",
        kind=SpanKind.TOOL,
        name="get_order",
        status=SpanStatus.OK,
        start_time=0.0,
        end_time=0.1,
        recorder_sequence_id=0,
        input={"order_id": "ORD-42"},
        output={"ok": True},
    )
    return Trace(id=trace_id, spans=[span])


class _JsonlTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = Path(self._tmpdir.name) / "runs.jsonl"

    def _lines(self) -> list[str]:
        return self.path.read_text(encoding="utf-8").splitlines()


class BasicWriteTests(_JsonlTestBase):
    def test_a_empty_sink_can_be_created(self) -> None:
        sink = JsonlEvidenceSink(self.path)
        sink.close()
        self.assertTrue(self.path.exists())

    def test_b_one_trace_produces_one_line(self) -> None:
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(_make_trace("t1"))
        self.assertEqual(len(self._lines()), 1)

    def test_c_two_traces_append_not_overwrite(self) -> None:
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(_make_trace("t1"))
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(_make_trace("t2"))
        self.assertEqual(len(self._lines()), 2)

    def test_d_three_traces_preserve_order(self) -> None:
        with JsonlEvidenceSink(self.path) as sink:
            for tid in ("t1", "t2", "t3"):
                sink.write_trace(_make_trace(tid))
        ids = [json.loads(line)["id"] for line in self._lines()]
        self.assertEqual(ids, ["t1", "t2", "t3"])

    def test_e_every_line_is_independently_valid_json(self) -> None:
        with JsonlEvidenceSink(self.path) as sink:
            for tid in ("t1", "t2", "t3"):
                sink.write_trace(_make_trace(tid))
        for line in self._lines():
            json.loads(line)  # raises if invalid

    def test_f_schema_version_present_on_every_line(self) -> None:
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(_make_trace("t1"))
            sink.write_trace(_make_trace("t2"))
        for line in self._lines():
            self.assertEqual(json.loads(line)["schema_version"], SCHEMA_VERSION)

    def test_g_written_trace_reconstructs_via_from_dict(self) -> None:
        trace = _make_trace("t1")
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
        restored = Trace.from_dict(json.loads(self._lines()[0]))
        self.assertEqual(restored, trace)


class EncodingAndStructureTests(_JsonlTestBase):
    def test_h_unicode_survives(self) -> None:
        span = Span(
            id="s1",
            kind=SpanKind.LLM,
            name="respond",
            status=SpanStatus.OK,
            start_time=0.0,
            end_time=0.1,
            recorder_sequence_id=0,
            output={"message": "返金は承認されました \U0001f389 — café"},
        )
        trace = Trace(id="unicode-trace", spans=[span])
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
        restored = Trace.from_dict(json.loads(self._lines()[0]))
        self.assertEqual(restored.spans[0].output["message"], span.output["message"])

    def test_i_nested_structures_survive(self) -> None:
        nested = {"a": [1, 2, {"b": None, "c": [True, False, 3.5]}]}
        span = Span(
            id="s1",
            kind=SpanKind.TOOL,
            name="n",
            status=SpanStatus.OK,
            start_time=0.0,
            end_time=0.1,
            recorder_sequence_id=0,
            output=nested,
        )
        trace = Trace(id="nested-trace", spans=[span])
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
        restored = Trace.from_dict(json.loads(self._lines()[0]))
        self.assertEqual(restored.spans[0].output, nested)

    def test_j_failure_signals_survive(self) -> None:
        signal = FailureSignal(
            id="fs1", source="pytest", verdict=ExternalStatus.FAILED, expected=True, actual=False
        )
        trace = Trace(id="t1", failure_signals=[signal])
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
        restored = Trace.from_dict(json.loads(self._lines()[0]))
        self.assertEqual(restored.failure_signals, [signal])

    def test_k_l_findings_and_triggering_signal_id_survive(self) -> None:
        signal = FailureSignal(
            id="fs1", source="pytest", verdict=ExternalStatus.FAILED, expected=True, actual=False
        )
        claim = Claim(type=ClaimType.VALUE_FIRST_OBSERVED, fields={"target_value": False})
        finding = Finding(
            id="f1",
            check_id="first_observed_occurrence",
            status=InternalCheckStatus.FLAGGED,
            claims=[claim],
            triggering_signal_id=signal.id,
        )
        trace = Trace(id="t1", failure_signals=[signal], findings=[finding])
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
        restored = Trace.from_dict(json.loads(self._lines()[0]))
        self.assertEqual(restored.findings, [finding])
        self.assertEqual(restored.findings[0].triggering_signal_id, signal.id)


class CloseBehaviorTests(_JsonlTestBase):
    def test_m_write_after_close_raises(self) -> None:
        sink = JsonlEvidenceSink(self.path)
        sink.close()
        with self.assertRaises(EvidenceSinkError):
            sink.write_trace(_make_trace("t1"))

    def test_n_close_is_idempotent(self) -> None:
        sink = JsonlEvidenceSink(self.path)
        sink.close()
        sink.close()  # must not raise

    def test_o_serialization_error_is_not_swallowed(self) -> None:
        class _BrokenTrace:
            def to_dict(self):
                raise RuntimeError("boom")

        with JsonlEvidenceSink(self.path) as sink:
            with self.assertRaises(RuntimeError):
                sink.write_trace(_BrokenTrace())  # type: ignore[arg-type]
        # Nothing corrupt was written as a side effect of the failure.
        self.assertEqual(self.path.read_text(encoding="utf-8"), "")

    def test_flush_makes_write_visible_before_close(self) -> None:
        sink = JsonlEvidenceSink(self.path)
        sink.write_trace(_make_trace("t1"))
        self.assertEqual(len(self._lines()), 1)  # read independently, sink still open
        sink.close()


class FileHandlingTests(_JsonlTestBase):
    def test_p_existing_file_is_appended_to(self) -> None:
        self.path.write_text(json.dumps({"pre-existing": True}) + "\n", encoding="utf-8")
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(_make_trace("t1"))
        lines = self._lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), {"pre-existing": True})

    def test_q_missing_parent_directory_raises_by_default(self) -> None:
        missing = Path(self._tmpdir.name) / "does" / "not" / "exist" / "runs.jsonl"
        with self.assertRaises(FileNotFoundError):
            JsonlEvidenceSink(missing)

    def test_q_create_parents_true_creates_missing_directories(self) -> None:
        missing = Path(self._tmpdir.name) / "does" / "not" / "exist" / "runs.jsonl"
        sink = JsonlEvidenceSink(missing, create_parents=True)
        sink.write_trace(_make_trace("t1"))
        sink.close()
        self.assertTrue(missing.exists())


class DuplicateAndIntegrityTests(_JsonlTestBase):
    def test_r_same_trace_written_twice_produces_two_records(self) -> None:
        trace = _make_trace("t1")
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
            sink.write_trace(trace)
        lines = self._lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), json.loads(lines[1]))

    def test_s_no_duplicate_or_truncated_output_across_many_writes(self) -> None:
        with JsonlEvidenceSink(self.path) as sink:
            for i in range(20):
                sink.write_trace(_make_trace(f"t{i}"))
        lines = self._lines()
        self.assertEqual(len(lines), 20)
        ids = [json.loads(line)["id"] for line in lines]
        self.assertEqual(ids, [f"t{i}" for i in range(20)])

    def test_t_sink_does_not_mutate_the_trace_object(self) -> None:
        trace = _make_trace("t1")
        before = trace.to_dict()
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
        self.assertEqual(trace.to_dict(), before)

    def test_thread_safe_concurrent_writes_within_one_process(self) -> None:
        sink = JsonlEvidenceSink(self.path)
        traces = [_make_trace(f"t{i}") for i in range(50)]
        threads = [threading.Thread(target=sink.write_trace, args=(t,)) for t in traces]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        sink.close()

        lines = self._lines()
        self.assertEqual(len(lines), 50)
        for line in lines:
            json.loads(line)  # every line independently valid -- never interleaved/corrupted


class SourceOfTruthTests(_JsonlTestBase):
    def test_w_written_line_exactly_matches_existing_to_json(self) -> None:
        trace = _make_trace("t1")
        with JsonlEvidenceSink(self.path) as sink:
            sink.write_trace(trace)
        self.assertEqual(self._lines()[0], to_json(trace))

    def test_u_v_no_database_or_langgraph_imports_in_evidence_source(self) -> None:
        import agentdebug.evidence.jsonl as jsonl_mod
        import agentdebug.evidence.sink as sink_mod

        banned = ("langgraph", "sqlite3", "psycopg2", "pymongo", "sqlalchemy")
        for mod in (sink_mod, jsonl_mod):
            source = inspect.getsource(mod).lower()
            for term in banned:
                self.assertNotIn(term, source)


if __name__ == "__main__":
    unittest.main()
