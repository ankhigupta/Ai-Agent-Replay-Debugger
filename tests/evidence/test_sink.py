"""Tests for the EvidenceSink abstract contract."""

from __future__ import annotations

import unittest

from agentdebug.core.exceptions import AgentDebugError
from agentdebug.core.trace import Trace
from agentdebug.evidence.sink import EvidenceSink, EvidenceSinkError


class _RecordingSink(EvidenceSink):
    """Minimal concrete EvidenceSink, for testing the base contract in
    isolation from JsonlEvidenceSink's storage details."""

    def __init__(self) -> None:
        self.written: list[Trace] = []
        self.closed = False

    def write_trace(self, trace: Trace) -> None:
        if self.closed:
            raise EvidenceSinkError("closed")
        self.written.append(trace)

    def close(self) -> None:
        self.closed = True


class EvidenceSinkContractTests(unittest.TestCase):
    def test_cannot_instantiate_abstract_base(self) -> None:
        with self.assertRaises(TypeError):
            EvidenceSink()  # type: ignore[abstract]

    def test_context_manager_calls_close_on_normal_exit(self) -> None:
        sink = _RecordingSink()
        with sink:
            self.assertFalse(sink.closed)
        self.assertTrue(sink.closed)

    def test_context_manager_calls_close_even_on_exception(self) -> None:
        sink = _RecordingSink()
        with self.assertRaises(ValueError):
            with sink:
                raise ValueError("boom")
        self.assertTrue(sink.closed)

    def test_evidence_sink_error_is_an_agentdebug_error(self) -> None:
        self.assertTrue(issubclass(EvidenceSinkError, AgentDebugError))

    def test_concrete_sink_implements_write_and_close(self) -> None:
        sink = _RecordingSink()
        trace = Trace(id="t1")
        sink.write_trace(trace)
        self.assertEqual(sink.written, [trace])
        sink.close()
        with self.assertRaises(EvidenceSinkError):
            sink.write_trace(trace)


if __name__ == "__main__":
    unittest.main()
