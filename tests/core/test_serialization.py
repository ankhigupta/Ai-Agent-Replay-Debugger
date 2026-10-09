"""Tests for Trace/model serialization: round trips, schema version, strictness."""

from __future__ import annotations

import json
import unittest

from agentdebug.core.claim import Claim
from agentdebug.core.context import TraceContext
from agentdebug.core.enums import (
    ClaimType,
    DiagnosisConfidence,
    ExternalStatus,
    InternalCheckStatus,
    SpanKind,
    SpanStatus,
)
from agentdebug.core.exceptions import SchemaVersionMismatchError, TraceValidationError
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.serialization import SCHEMA_VERSION, from_json, to_json
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace


def build_sample_trace() -> Trace:
    context = TraceContext(
        initial_state={"refund_eligible": None},
        user_input="Can I get a refund for order ORD-42?",
        system_prompt="You are a refund assistant.",
        config={},
    )
    span1 = Span(
        id="span-1",
        parent_id=None,
        kind=SpanKind.TOOL,
        name="get_order",
        status=SpanStatus.OK,
        start_time=1000.0,
        end_time=1000.2,
        recorder_sequence_id=0,
        input={"order_id": "ORD-42"},
        output={"order_id": "ORD-42", "refund_eligible": True},
        metadata={},
    )
    span2 = Span(
        id="span-2",
        parent_id=None,
        kind=SpanKind.STATE,
        name="state_update",
        status=SpanStatus.OK,
        start_time=1000.3,
        end_time=1000.3,
        recorder_sequence_id=1,
        input=None,
        output={"refund_eligible": False},
        metadata={},
    )
    span3 = Span(
        id="span-3",
        parent_id=None,
        kind=SpanKind.LLM,
        name="final_answer",
        status=SpanStatus.OK,
        start_time=1000.4,
        end_time=1000.9,
        recorder_sequence_id=2,
        input={"prompt": "Refund decision for ORD-42?"},
        output={"text": "Refund denied."},
        metadata={"model": "gpt-4o"},
    )
    signal = FailureSignal(
        id="fs-1",
        source="pytest",
        verdict=ExternalStatus.FAILED,
        expected="approve_refund",
        actual="Refund denied.",
        detail="test_refund_approval failed",
    )
    claim = Claim(
        type=ClaimType.VALUE_FIRST_OBSERVED,
        span_ids=["span-2"],
        fields={"value": False, "field_path": "refund_eligible"},
    )
    finding = Finding(
        id="finding-1",
        check_id="first_observed_occurrence",
        status=InternalCheckStatus.FLAGGED,
        claims=[claim],
        diagnosis_confidence=DiagnosisConfidence.UNIQUE,
        triggering_signal_id="fs-1",
    )
    return Trace(
        id="trace-001",
        context=context,
        spans=[span1, span2, span3],
        failure_signals=[signal],
        findings=[finding],
    )


class SerializationRoundTripTests(unittest.TestCase):
    def test_to_dict_from_dict_round_trip(self) -> None:
        trace = build_sample_trace()
        restored = Trace.from_dict(trace.to_dict())
        self.assertEqual(trace, restored)

    def test_to_json_from_json_round_trip(self) -> None:
        trace = build_sample_trace()
        restored = from_json(to_json(trace))
        self.assertEqual(trace, restored)

    def test_enums_serialize_as_plain_strings(self) -> None:
        d = build_sample_trace().to_dict()
        self.assertEqual(d["spans"][0]["kind"], "TOOL")
        self.assertEqual(d["findings"][0]["status"], "FLAGGED")
        self.assertNotIn("SpanKind", json.dumps(d))
        self.assertNotIn("InternalCheckStatus", json.dumps(d))

    def test_no_causal_field_in_claim_output(self) -> None:
        d = build_sample_trace().to_dict()
        self.assertNotIn("caused_failure", d["findings"][0]["claims"][0])

    def test_schema_version_present_and_correct(self) -> None:
        self.assertEqual(build_sample_trace().to_dict()["schema_version"], SCHEMA_VERSION)


class SchemaVersionTests(unittest.TestCase):
    def test_mismatched_schema_version_rejected(self) -> None:
        data = build_sample_trace().to_dict()
        data["schema_version"] = "0.9.0"
        with self.assertRaises(SchemaVersionMismatchError):
            Trace.from_dict(data)

    def test_missing_schema_version_rejected(self) -> None:
        data = build_sample_trace().to_dict()
        del data["schema_version"]
        with self.assertRaises(TraceValidationError):
            Trace.from_dict(data)

    def test_matching_schema_version_accepted(self) -> None:
        data = build_sample_trace().to_dict()
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        Trace.from_dict(data)  # must not raise


class UnknownFieldTests(unittest.TestCase):
    def test_unknown_top_level_field_rejected(self) -> None:
        data = build_sample_trace().to_dict()
        data["unexpected"] = True
        with self.assertRaises(TraceValidationError):
            Trace.from_dict(data)

    def test_unknown_nested_span_field_rejected(self) -> None:
        data = build_sample_trace().to_dict()
        data["spans"][0]["unexpected"] = True
        with self.assertRaises(TraceValidationError):
            Trace.from_dict(data)

    def test_unknown_nested_finding_field_rejected(self) -> None:
        data = build_sample_trace().to_dict()
        data["findings"][0]["unexpected"] = True
        with self.assertRaises(TraceValidationError):
            Trace.from_dict(data)


class InvalidEnumTests(unittest.TestCase):
    def test_invalid_span_kind_rejected(self) -> None:
        data = build_sample_trace().to_dict()
        data["spans"][0]["kind"] = "NOT_A_KIND"
        with self.assertRaises(TraceValidationError):
            Trace.from_dict(data)

    def test_invalid_finding_status_rejected(self) -> None:
        data = build_sample_trace().to_dict()
        data["findings"][0]["status"] = "NOT_A_STATUS"
        with self.assertRaises(TraceValidationError):
            Trace.from_dict(data)


class MalformedDataTests(unittest.TestCase):
    def test_non_dict_top_level_rejected(self) -> None:
        with self.assertRaises(TraceValidationError):
            Trace.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_invalid_json_string_rejected(self) -> None:
        with self.assertRaises(TraceValidationError):
            from_json("{not valid json")

    def test_from_json_revalidates_structural_invariants(self) -> None:
        """Trace validation must re-run after deserialization, not just at
        construction from already-valid Python objects: a duplicate span id
        injected directly into serialized JSON must still be rejected."""
        data = build_sample_trace().to_dict()
        data["spans"].append(dict(data["spans"][0]))  # duplicate span id
        with self.assertRaises(TraceValidationError):
            from_json(json.dumps(data))


if __name__ == "__main__":
    unittest.main()
