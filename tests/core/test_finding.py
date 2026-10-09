"""Tests for Finding, covering the status-conditioned shape table."""

from __future__ import annotations

import dataclasses
import unittest

from agentdebug.core.claim import Claim
from agentdebug.core.enums import (
    Capability,
    ClaimType,
    DiagnosisConfidence,
    InternalCheckStatus,
)
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.finding import Finding

_CLAIM = Claim(type=ClaimType.VALUE_FIRST_OBSERVED, span_ids=["s1"], fields={"value": True})


class FindingFlaggedTests(unittest.TestCase):
    def test_requires_claims(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(check_id="c", status=InternalCheckStatus.FLAGGED, claims=[])

    def test_forbids_capability_gaps(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(
                check_id="c",
                status=InternalCheckStatus.FLAGGED,
                claims=[_CLAIM],
                capability_gaps=[Capability.SPAN_IO],
            )

    def test_valid(self) -> None:
        finding = Finding(check_id="c", status=InternalCheckStatus.FLAGGED, claims=[_CLAIM])
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)


class FindingNoFindingTests(unittest.TestCase):
    def test_forbids_non_value_not_observed_claims(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(check_id="c", status=InternalCheckStatus.NO_FINDING, claims=[_CLAIM])

    def test_allows_value_not_observed_claim(self) -> None:
        """Narrow exception (added for Phase 4 localization): a
        VALUE_NOT_OBSERVED claim records that a search was carried out and
        came up empty -- a factual, non-causal claim, not a defect
        finding -- so it may coexist with NO_FINDING."""
        claim = Claim(type=ClaimType.VALUE_NOT_OBSERVED, fields={"target_value": 1})
        finding = Finding(check_id="c", status=InternalCheckStatus.NO_FINDING, claims=[claim])
        self.assertEqual(finding.claims, [claim])

    def test_forbids_mixed_claims_with_non_value_not_observed_type(self) -> None:
        not_observed = Claim(type=ClaimType.VALUE_NOT_OBSERVED, fields={"target_value": 1})
        with self.assertRaises(TraceValidationError):
            Finding(check_id="c", status=InternalCheckStatus.NO_FINDING, claims=[not_observed, _CLAIM])

    def test_forbids_capability_gaps(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(
                check_id="c", status=InternalCheckStatus.NO_FINDING, capability_gaps=[Capability.SPAN_IO]
            )

    def test_valid(self) -> None:
        finding = Finding(check_id="c", status=InternalCheckStatus.NO_FINDING)
        self.assertEqual(finding.claims, [])


class FindingNotApplicableTests(unittest.TestCase):
    def test_forbids_claims(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(
                check_id="c",
                status=InternalCheckStatus.NOT_APPLICABLE,
                claims=[_CLAIM],
                capability_gaps=[Capability.SPAN_IO],
            )

    def test_requires_capability_gaps(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(check_id="c", status=InternalCheckStatus.NOT_APPLICABLE, capability_gaps=[])

    def test_valid(self) -> None:
        finding = Finding(
            check_id="c",
            status=InternalCheckStatus.NOT_APPLICABLE,
            capability_gaps=[Capability.TOOL_STATUS],
        )
        self.assertEqual(finding.capability_gaps, [Capability.TOOL_STATUS])


class FindingInconclusiveTests(unittest.TestCase):
    def test_requires_capability_gaps(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(check_id="c", status=InternalCheckStatus.INCONCLUSIVE, capability_gaps=[])

    def test_claims_may_be_empty(self) -> None:
        finding = Finding(
            check_id="c",
            status=InternalCheckStatus.INCONCLUSIVE,
            capability_gaps=[Capability.TOOL_STATUS],
        )
        self.assertEqual(finding.claims, [])

    def test_claims_may_be_non_empty(self) -> None:
        finding = Finding(
            check_id="c",
            status=InternalCheckStatus.INCONCLUSIVE,
            claims=[_CLAIM],
            capability_gaps=[Capability.TOOL_STATUS],
        )
        self.assertEqual(finding.claims, [_CLAIM])


class FindingMiscTests(unittest.TestCase):
    def test_invalid_check_id_fails(self) -> None:
        with self.assertRaises(TraceValidationError):
            Finding(check_id="", status=InternalCheckStatus.NO_FINDING)

    def test_id_auto_generated(self) -> None:
        finding = Finding(check_id="c", status=InternalCheckStatus.NO_FINDING)
        self.assertTrue(finding.id)

    def test_diagnosis_confidence_accepts_none_or_valid(self) -> None:
        finding = Finding(
            check_id="c",
            status=InternalCheckStatus.FLAGGED,
            claims=[_CLAIM],
            diagnosis_confidence=DiagnosisConfidence.UNIQUE,
        )
        self.assertEqual(finding.diagnosis_confidence, DiagnosisConfidence.UNIQUE)

    def test_triggering_signal_id_accepts_none_or_non_empty_string(self) -> None:
        finding = Finding(
            check_id="c", status=InternalCheckStatus.FLAGGED, claims=[_CLAIM], triggering_signal_id="fs-1"
        )
        self.assertEqual(finding.triggering_signal_id, "fs-1")
        with self.assertRaises(TraceValidationError):
            Finding(
                check_id="c", status=InternalCheckStatus.FLAGGED, claims=[_CLAIM], triggering_signal_id=""
            )

    def test_frozen(self) -> None:
        finding = Finding(check_id="c", status=InternalCheckStatus.NO_FINDING)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            finding.check_id = "other"  # type: ignore[misc]

    def test_round_trip_dict(self) -> None:
        finding = Finding(
            check_id="c",
            status=InternalCheckStatus.FLAGGED,
            claims=[_CLAIM],
            diagnosis_confidence=DiagnosisConfidence.UNIQUE,
            triggering_signal_id="fs-1",
        )
        restored = Finding.from_dict(finding.to_dict())
        self.assertEqual(finding, restored)

    def test_from_dict_rejects_unknown_field(self) -> None:
        data = Finding(check_id="c", status=InternalCheckStatus.NO_FINDING).to_dict()
        data["bogus"] = 1
        with self.assertRaises(TraceValidationError):
            Finding.from_dict(data)


if __name__ == "__main__":
    unittest.main()
