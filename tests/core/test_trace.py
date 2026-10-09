"""Tests for Trace: invariants, mutation API, lookup, and computed status."""

from __future__ import annotations

import unittest
from typing import Optional

from agentdebug.core.claim import Claim
from agentdebug.core.enums import (
    Capability,
    ClaimType,
    ExternalStatus,
    InternalCheckStatus,
    InternalRunStatus,
    SpanKind,
    SpanStatus,
)
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace


def make_span(id_: str, seq: int, parent_id: Optional[str] = None) -> Span:
    return Span(
        id=id_,
        parent_id=parent_id,
        kind=SpanKind.TOOL,
        name="n",
        status=SpanStatus.OK,
        start_time=0.0,
        end_time=1.0,
        recorder_sequence_id=seq,
    )


class TraceConstructionInvariantTests(unittest.TestCase):
    def test_duplicate_span_id_rejected(self) -> None:
        with self.assertRaises(TraceValidationError):
            Trace(id="t1", spans=[make_span("s1", 0), make_span("s1", 1)])

    def test_duplicate_recorder_sequence_id_rejected(self) -> None:
        with self.assertRaises(TraceValidationError):
            Trace(id="t1", spans=[make_span("s1", 0), make_span("s2", 0)])

    def test_duplicate_finding_id_rejected(self) -> None:
        finding = Finding(id="f1", check_id="c", status=InternalCheckStatus.NO_FINDING)
        with self.assertRaises(TraceValidationError):
            Trace(id="t1", findings=[finding, finding])

    def test_duplicate_failure_signal_id_rejected(self) -> None:
        signal = FailureSignal(id="fs1", source="pytest", verdict=ExternalStatus.PASSED)
        with self.assertRaises(TraceValidationError):
            Trace(id="t1", failure_signals=[signal, signal])

    def test_dangling_span_parent_id_allowed(self) -> None:
        trace = Trace(id="t1", spans=[make_span("s1", 0, parent_id="missing-span")])
        self.assertEqual(trace.spans[0].parent_id, "missing-span")

    def test_invalid_claim_span_id_rejected(self) -> None:
        claim = Claim(type=ClaimType.VALUE_FIRST_OBSERVED, span_ids=["does-not-exist"])
        finding = Finding(check_id="c", status=InternalCheckStatus.FLAGGED, claims=[claim])
        with self.assertRaises(TraceValidationError):
            Trace(id="t1", findings=[finding])

    def test_invalid_triggering_signal_id_rejected(self) -> None:
        finding = Finding(
            check_id="c", status=InternalCheckStatus.NO_FINDING, triggering_signal_id="no-such-signal"
        )
        with self.assertRaises(TraceValidationError):
            Trace(id="t1", findings=[finding])


class TraceLookupTests(unittest.TestCase):
    def test_get_span_found(self) -> None:
        span = make_span("s1", 0)
        trace = Trace(id="t1", spans=[span])
        self.assertEqual(trace.get_span("s1"), span)

    def test_get_span_not_found(self) -> None:
        trace = Trace(id="t1")
        self.assertIsNone(trace.get_span("nope"))


class TraceExternalStatusTests(unittest.TestCase):
    def test_no_signals_is_unknown(self) -> None:
        self.assertEqual(Trace(id="t1").external_status, ExternalStatus.UNKNOWN)

    def test_all_passed(self) -> None:
        trace = Trace(
            id="t1",
            failure_signals=[
                FailureSignal(source="a", verdict=ExternalStatus.PASSED),
                FailureSignal(source="b", verdict=ExternalStatus.PASSED),
            ],
        )
        self.assertEqual(trace.external_status, ExternalStatus.PASSED)

    def test_any_failed_wins(self) -> None:
        trace = Trace(
            id="t1",
            failure_signals=[
                FailureSignal(source="a", verdict=ExternalStatus.FAILED),
                FailureSignal(source="b", verdict=ExternalStatus.PASSED),
            ],
        )
        self.assertEqual(trace.external_status, ExternalStatus.FAILED)

    def test_single_unknown(self) -> None:
        trace = Trace(id="t1", failure_signals=[FailureSignal(source="a", verdict=ExternalStatus.UNKNOWN)])
        self.assertEqual(trace.external_status, ExternalStatus.UNKNOWN)

    def test_passed_plus_unknown_is_unknown(self) -> None:
        trace = Trace(
            id="t1",
            failure_signals=[
                FailureSignal(source="a", verdict=ExternalStatus.PASSED),
                FailureSignal(source="b", verdict=ExternalStatus.UNKNOWN),
            ],
        )
        self.assertEqual(trace.external_status, ExternalStatus.UNKNOWN)


class TraceInternalStatusTests(unittest.TestCase):
    def test_no_findings_is_inconclusive(self) -> None:
        self.assertEqual(Trace(id="t1").internal_status, InternalRunStatus.INCONCLUSIVE)

    def test_all_not_applicable_is_inconclusive(self) -> None:
        trace = Trace(
            id="t1",
            findings=[
                Finding(
                    check_id="c",
                    status=InternalCheckStatus.NOT_APPLICABLE,
                    capability_gaps=[Capability.SPAN_IO],
                )
            ],
        )
        self.assertEqual(trace.internal_status, InternalRunStatus.INCONCLUSIVE)

    def test_flagged_wins_over_everything(self) -> None:
        claim = Claim(type=ClaimType.VALUE_FIRST_OBSERVED)
        trace = Trace(
            id="t1",
            findings=[
                Finding(check_id="a", status=InternalCheckStatus.FLAGGED, claims=[claim]),
                Finding(
                    check_id="b",
                    status=InternalCheckStatus.INCONCLUSIVE,
                    capability_gaps=[Capability.SPAN_IO],
                ),
            ],
        )
        self.assertEqual(trace.internal_status, InternalRunStatus.FLAGGED)

    def test_inconclusive_without_flagged(self) -> None:
        trace = Trace(
            id="t1",
            findings=[
                Finding(check_id="a", status=InternalCheckStatus.NO_FINDING),
                Finding(
                    check_id="b",
                    status=InternalCheckStatus.INCONCLUSIVE,
                    capability_gaps=[Capability.SPAN_IO],
                ),
            ],
        )
        self.assertEqual(trace.internal_status, InternalRunStatus.INCONCLUSIVE)

    def test_clean_under_checks(self) -> None:
        trace = Trace(
            id="t1",
            findings=[
                Finding(check_id="a", status=InternalCheckStatus.NO_FINDING),
                Finding(
                    check_id="b",
                    status=InternalCheckStatus.NOT_APPLICABLE,
                    capability_gaps=[Capability.SPAN_IO],
                ),
            ],
        )
        self.assertEqual(trace.internal_status, InternalRunStatus.CLEAN_UNDER_CHECKS)


class TraceCounterTests(unittest.TestCase):
    def test_counters(self) -> None:
        trace = Trace(
            id="t1",
            findings=[
                Finding(
                    check_id="a",
                    status=InternalCheckStatus.FLAGGED,
                    claims=[Claim(type=ClaimType.VALUE_FIRST_OBSERVED)],
                ),
                Finding(
                    check_id="b",
                    status=InternalCheckStatus.NOT_APPLICABLE,
                    capability_gaps=[Capability.SPAN_IO],
                ),
                Finding(
                    check_id="c",
                    status=InternalCheckStatus.INCONCLUSIVE,
                    capability_gaps=[Capability.SPAN_IO],
                ),
                Finding(check_id="d", status=InternalCheckStatus.NO_FINDING),
            ],
        )
        self.assertEqual(trace.checks_run, 3)
        self.assertEqual(trace.checks_not_applicable, 1)
        self.assertEqual(trace.checks_inconclusive, 1)


class TraceMutationApiTests(unittest.TestCase):
    def test_add_span(self) -> None:
        trace = Trace(id="t1")
        trace.add_span(make_span("s1", 0))
        self.assertEqual(len(trace.spans), 1)

    def test_add_span_duplicate_id_rejected_without_mutation(self) -> None:
        trace = Trace(id="t1")
        trace.add_span(make_span("s1", 0))
        with self.assertRaises(TraceValidationError):
            trace.add_span(make_span("s1", 1))
        self.assertEqual(len(trace.spans), 1)

    def test_add_span_duplicate_sequence_id_rejected_without_mutation(self) -> None:
        trace = Trace(id="t1")
        trace.add_span(make_span("s1", 0))
        with self.assertRaises(TraceValidationError):
            trace.add_span(make_span("s2", 0))
        self.assertEqual(len(trace.spans), 1)

    def test_add_failure_signal(self) -> None:
        trace = Trace(id="t1")
        trace.add_failure_signal(FailureSignal(id="fs1", source="a", verdict=ExternalStatus.PASSED))
        self.assertEqual(len(trace.failure_signals), 1)

    def test_add_failure_signal_duplicate_rejected_without_mutation(self) -> None:
        trace = Trace(id="t1")
        trace.add_failure_signal(FailureSignal(id="fs1", source="a", verdict=ExternalStatus.PASSED))
        with self.assertRaises(TraceValidationError):
            trace.add_failure_signal(FailureSignal(id="fs1", source="b", verdict=ExternalStatus.FAILED))
        self.assertEqual(len(trace.failure_signals), 1)

    def test_add_finding(self) -> None:
        trace = Trace(id="t1")
        trace.add_span(make_span("s1", 0))
        claim = Claim(type=ClaimType.VALUE_FIRST_OBSERVED, span_ids=["s1"])
        trace.add_finding(Finding(check_id="c", status=InternalCheckStatus.FLAGGED, claims=[claim]))
        self.assertEqual(len(trace.findings), 1)

    def test_add_finding_duplicate_id_rejected_without_mutation(self) -> None:
        trace = Trace(id="t1")
        finding = Finding(id="f1", check_id="c", status=InternalCheckStatus.NO_FINDING)
        trace.add_finding(finding)
        with self.assertRaises(TraceValidationError):
            trace.add_finding(finding)
        self.assertEqual(len(trace.findings), 1)

    def test_add_finding_unknown_span_reference_rejected_without_mutation(self) -> None:
        trace = Trace(id="t1")
        claim = Claim(type=ClaimType.VALUE_FIRST_OBSERVED, span_ids=["ghost"])
        finding = Finding(check_id="c", status=InternalCheckStatus.FLAGGED, claims=[claim])
        with self.assertRaises(TraceValidationError):
            trace.add_finding(finding)
        self.assertEqual(len(trace.findings), 0)

    def test_add_finding_unknown_triggering_signal_rejected_without_mutation(self) -> None:
        trace = Trace(id="t1")
        finding = Finding(check_id="c", status=InternalCheckStatus.NO_FINDING, triggering_signal_id="ghost")
        with self.assertRaises(TraceValidationError):
            trace.add_finding(finding)
        self.assertEqual(len(trace.findings), 0)


if __name__ == "__main__":
    unittest.main()
