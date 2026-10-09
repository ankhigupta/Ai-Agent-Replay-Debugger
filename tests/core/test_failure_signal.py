"""Tests for FailureSignal."""

from __future__ import annotations

import dataclasses
import unittest
import uuid

from agentdebug.core.enums import ExternalStatus
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.failure_signal import FailureSignal


class FailureSignalTests(unittest.TestCase):
    def test_id_auto_generated_and_is_valid_uuid4(self) -> None:
        signal = FailureSignal(source="pytest", verdict=ExternalStatus.FAILED)
        parsed = uuid.UUID(signal.id, version=4)
        self.assertEqual(str(parsed), signal.id)

    def test_valid_construction(self) -> None:
        signal = FailureSignal(
            source="pytest",
            verdict=ExternalStatus.FAILED,
            expected="approve_refund",
            actual="Refund denied.",
            detail="test_refund_approval failed",
        )
        self.assertEqual(signal.verdict, ExternalStatus.FAILED)

    def test_invalid_source_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            FailureSignal(source="", verdict=ExternalStatus.FAILED)

    def test_invalid_verdict_rejected_by_enum(self) -> None:
        with self.assertRaises(ValueError):
            ExternalStatus("NOT_A_VERDICT")

    def test_invalid_expected_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            FailureSignal(source="pytest", verdict=ExternalStatus.FAILED, expected=object())

    def test_invalid_actual_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            FailureSignal(source="pytest", verdict=ExternalStatus.FAILED, actual=object())

    def test_invalid_detail_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            FailureSignal(source="pytest", verdict=ExternalStatus.FAILED, detail=123)  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        signal = FailureSignal(source="pytest", verdict=ExternalStatus.PASSED)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            signal.source = "other"  # type: ignore[misc]

    def test_round_trip_dict(self) -> None:
        signal = FailureSignal(
            source="pytest", verdict=ExternalStatus.FAILED, expected="x", actual="y", detail="z"
        )
        restored = FailureSignal.from_dict(signal.to_dict())
        self.assertEqual(signal, restored)

    def test_from_dict_rejects_unknown_field(self) -> None:
        data = FailureSignal(source="pytest", verdict=ExternalStatus.PASSED).to_dict()
        data["bogus"] = 1
        with self.assertRaises(TraceValidationError):
            FailureSignal.from_dict(data)


if __name__ == "__main__":
    unittest.main()
