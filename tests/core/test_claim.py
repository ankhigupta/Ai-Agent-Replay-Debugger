"""Tests for Claim."""

from __future__ import annotations

import unittest

from agentdebug.core.claim import Claim
from agentdebug.core.enums import ClaimType
from agentdebug.core.exceptions import TraceValidationError


class ClaimTests(unittest.TestCase):
    def test_valid_construction(self) -> None:
        claim = Claim(
            type=ClaimType.VALUE_FIRST_OBSERVED,
            span_ids=["span-2"],
            fields={"value": False, "field_path": "refund_eligible"},
        )
        self.assertEqual(claim.type, ClaimType.VALUE_FIRST_OBSERVED)

    def test_invalid_claim_type_rejected_by_enum(self) -> None:
        with self.assertRaises(ValueError):
            ClaimType("NOT_A_TYPE")

    def test_invalid_span_id_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            Claim(type=ClaimType.TRACE_INCOMPLETE, span_ids=[""])

    def test_non_string_span_id_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            Claim(type=ClaimType.TRACE_INCOMPLETE, span_ids=[123])  # type: ignore[list-item]

    def test_invalid_fields_value_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            Claim(type=ClaimType.TRACE_INCOMPLETE, fields={"bad": object()})

    def test_nested_dict_list_primitives_accepted(self) -> None:
        claim = Claim(
            type=ClaimType.REPEATED_IDENTICAL_CALLS,
            fields={"calls": [{"name": "get_order", "args": {"id": "ORD-42", "retries": 2}}]},
        )
        self.assertEqual(claim.fields["calls"][0]["args"]["retries"], 2)

    def test_empty_span_ids_accepted(self) -> None:
        claim = Claim(type=ClaimType.TRACE_INCOMPLETE)
        self.assertEqual(claim.span_ids, [])

    def test_round_trip_dict(self) -> None:
        claim = Claim(
            type=ClaimType.VALUE_FIRST_OBSERVED,
            span_ids=["span-2"],
            fields={"value": False, "field_path": "refund_eligible"},
        )
        restored = Claim.from_dict(claim.to_dict())
        self.assertEqual(claim, restored)

    def test_from_dict_rejects_unknown_field(self) -> None:
        data = Claim(type=ClaimType.TRACE_INCOMPLETE).to_dict()
        data["bogus"] = 1
        with self.assertRaises(TraceValidationError):
            Claim.from_dict(data)


if __name__ == "__main__":
    unittest.main()
