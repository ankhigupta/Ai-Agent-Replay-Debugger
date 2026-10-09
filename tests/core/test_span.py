"""Tests for Span."""

from __future__ import annotations

import dataclasses
import unittest
from typing import Any

from agentdebug.core.enums import SpanKind, SpanStatus
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.span import Span


def make_span(**overrides: Any) -> Span:
    defaults: dict[str, Any] = dict(
        id="s1",
        parent_id=None,
        kind=SpanKind.TOOL,
        name="get_order",
        status=SpanStatus.OK,
        start_time=0.0,
        end_time=1.0,
        recorder_sequence_id=0,
        input={"a": 1},
        output={"b": 2},
        metadata={},
    )
    defaults.update(overrides)
    return Span(**defaults)


class SpanTests(unittest.TestCase):
    def test_valid_construction(self) -> None:
        span = make_span()
        self.assertEqual(span.id, "s1")
        self.assertEqual(span.kind, SpanKind.TOOL)

    def test_empty_id_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(id="")

    def test_empty_name_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(name="")

    def test_negative_start_time_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(start_time=-1.0)

    def test_end_before_start_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(start_time=5.0, end_time=1.0)

    def test_negative_recorder_sequence_id_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(recorder_sequence_id=-1)

    def test_invalid_input_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(input=object())

    def test_invalid_output_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(output=object())

    def test_invalid_metadata_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            make_span(metadata={"bad": object()})

    def test_duration(self) -> None:
        span = make_span(start_time=1.5, end_time=4.0)
        self.assertAlmostEqual(span.duration, 2.5)

    def test_structural_equality(self) -> None:
        self.assertEqual(make_span(), make_span())

    def test_dangling_parent_id_allowed(self) -> None:
        span = make_span(parent_id="does-not-exist-anywhere")
        self.assertEqual(span.parent_id, "does-not-exist-anywhere")

    def test_frozen(self) -> None:
        span = make_span()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            span.name = "renamed"  # type: ignore[misc]

    def test_round_trip_dict(self) -> None:
        span = make_span()
        restored = Span.from_dict(span.to_dict())
        self.assertEqual(span, restored)

    def test_from_dict_rejects_unknown_field(self) -> None:
        data = make_span().to_dict()
        data["bogus"] = 1
        with self.assertRaises(TraceValidationError):
            Span.from_dict(data)

    def test_from_dict_missing_required_field_fails(self) -> None:
        data = make_span().to_dict()
        del data["name"]
        with self.assertRaises(TraceValidationError):
            Span.from_dict(data)


if __name__ == "__main__":
    unittest.main()
