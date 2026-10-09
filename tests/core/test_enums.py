"""Tests for the closed AgentDebug core enums."""

from __future__ import annotations

import unittest

from agentdebug.core.enums import (
    Capability,
    ClaimType,
    DiagnosisConfidence,
    ExternalStatus,
    InternalCheckStatus,
    InternalRunStatus,
    SpanKind,
    SpanStatus,
)
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.span import Span


class SpanKindTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        expected = {
            "LLM": "LLM",
            "TOOL": "TOOL",
            "RETRIEVAL": "RETRIEVAL",
            "STATE": "STATE",
            "ERROR": "ERROR",
            "CUSTOM": "CUSTOM",
        }
        for name, value in expected.items():
            self.assertEqual(getattr(SpanKind, name).value, value)

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpanKind("NOT_A_KIND")


class SpanStatusTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        self.assertEqual(SpanStatus.OK.value, "OK")
        self.assertEqual(SpanStatus.ERROR.value, "ERROR")
        self.assertEqual(SpanStatus.UNKNOWN.value, "UNKNOWN")

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpanStatus("BOGUS")


class ExternalStatusTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        self.assertEqual(ExternalStatus.FAILED.value, "FAILED")
        self.assertEqual(ExternalStatus.PASSED.value, "PASSED")
        self.assertEqual(ExternalStatus.UNKNOWN.value, "UNKNOWN")

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExternalStatus("BOGUS")


class InternalCheckStatusTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        self.assertEqual(InternalCheckStatus.FLAGGED.value, "FLAGGED")
        self.assertEqual(InternalCheckStatus.NO_FINDING.value, "NO_FINDING")
        self.assertEqual(InternalCheckStatus.NOT_APPLICABLE.value, "NOT_APPLICABLE")
        self.assertEqual(InternalCheckStatus.INCONCLUSIVE.value, "INCONCLUSIVE")

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InternalCheckStatus("BOGUS")

    def test_distinct_member_set_from_internal_run_status(self) -> None:
        self.assertNotEqual({m.value for m in InternalCheckStatus}, {m.value for m in InternalRunStatus})


class InternalRunStatusTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        self.assertEqual(InternalRunStatus.FLAGGED.value, "FLAGGED")
        self.assertEqual(InternalRunStatus.INCONCLUSIVE.value, "INCONCLUSIVE")
        self.assertEqual(InternalRunStatus.CLEAN_UNDER_CHECKS.value, "CLEAN_UNDER_CHECKS")

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InternalRunStatus("BOGUS")


class ClaimTypeTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        expected = {
            "TOOL_ERROR_SWALLOWED",
            "REPEATED_IDENTICAL_CALLS",
            "TRACE_INCOMPLETE",
            "VALUE_FIRST_OBSERVED",
            "VALUE_NOT_OBSERVED",
        }
        self.assertEqual({m.value for m in ClaimType}, expected)

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ClaimType("BOGUS")


class CapabilityTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        expected = {"TOOL_STATUS", "TOOL_CALLS", "SPAN_IO", "PRE_RUN_CONTEXT", "RETRIEVAL_DOCS"}
        self.assertEqual({m.value for m in Capability}, expected)

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Capability("BOGUS")


class DiagnosisConfidenceTests(unittest.TestCase):
    def test_members_and_values(self) -> None:
        self.assertEqual(DiagnosisConfidence.UNIQUE.value, "UNIQUE")
        self.assertEqual(DiagnosisConfidence.AMBIGUOUS.value, "AMBIGUOUS")
        self.assertEqual(DiagnosisConfidence.NONE.value, "NONE")

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DiagnosisConfidence("BOGUS")


class EnumDeserializationTests(unittest.TestCase):
    """Invalid enum values encountered during deserialization must raise
    TraceValidationError, not a bare ValueError. Span.from_dict is used
    here as a representative path through serialization.enum_from_value."""

    def test_invalid_kind_in_span_from_dict_raises_trace_validation_error(self) -> None:
        data = {
            "id": "s1",
            "parent_id": None,
            "kind": "NOT_A_KIND",
            "name": "n",
            "status": "OK",
            "start_time": 0.0,
            "end_time": 0.0,
            "recorder_sequence_id": 0,
            "input": None,
            "output": None,
            "metadata": {},
        }
        with self.assertRaises(TraceValidationError):
            Span.from_dict(data)


if __name__ == "__main__":
    unittest.main()
