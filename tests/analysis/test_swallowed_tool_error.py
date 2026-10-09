"""Tests for SwallowedToolErrorCheck."""

from __future__ import annotations

import json
import unittest

from agentdebug.analysis.checks.swallowed_tool_error import SwallowedToolErrorCheck
from agentdebug.core.enums import Capability, InternalCheckStatus, SpanStatus
from tests.analysis._helpers import finding_content, make_span, make_trace


class ErrorWithNoRecoveryTests(unittest.TestCase):
    def test_error_tool_with_no_later_matching_sibling_is_flagged(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        trace = make_trace(spans=[failed])
        finding = SwallowedToolErrorCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(len(finding.claims), 1)
        self.assertEqual(finding.claims[0].fields["tool_span_id"], failed.id)

    def test_error_tool_followed_by_unrelated_tool_still_flagged(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        unrelated = make_span(
            "get_weather", status=SpanStatus.OK, parent_id="root", start_time=2.0, recorder_sequence_id=2
        )
        trace = make_trace(spans=[failed, unrelated])
        finding = SwallowedToolErrorCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)


class RecoveryTests(unittest.TestCase):
    def test_error_tool_followed_by_matching_retry_is_no_finding(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        retry = make_span(
            "get_order", status=SpanStatus.OK, parent_id="root", start_time=2.0, recorder_sequence_id=2
        )
        trace = make_trace(spans=[failed, retry])
        finding = SwallowedToolErrorCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)

    def test_matching_descendant_does_not_count_as_sibling(self) -> None:
        failed = make_span(
            "get_order",
            status=SpanStatus.ERROR,
            parent_id="root",
            start_time=1.0,
            recorder_sequence_id=1,
            id="failed",
        )
        # Same name, but nested *under* the failed span, not a sibling of it.
        descendant = make_span(
            "get_order", status=SpanStatus.OK, parent_id="failed", start_time=2.0, recorder_sequence_id=2
        )
        trace = make_trace(spans=[failed, descendant])
        finding = SwallowedToolErrorCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)

    def test_failed_tool_with_dangling_parent_is_still_analyzable(self) -> None:
        """Sibling matching uses the literal parent_id value, never
        dereferencing it — so a failed tool whose parent doesn't exist
        anywhere in the trace is still analyzed normally, and a sibling
        sharing that same (dangling) parent_id value is still found."""
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="ghost-parent", start_time=1.0, recorder_sequence_id=1
        )
        trace_alone = make_trace(spans=[failed])
        self.assertEqual(SwallowedToolErrorCheck().run(trace_alone).status, InternalCheckStatus.FLAGGED)

        retry = make_span(
            "get_order", status=SpanStatus.OK, parent_id="ghost-parent", start_time=2.0, recorder_sequence_id=2
        )
        trace_with_retry = make_trace(spans=[failed, retry])
        self.assertEqual(
            SwallowedToolErrorCheck().run(trace_with_retry).status, InternalCheckStatus.NO_FINDING
        )


class MultipleFailuresOrderingTests(unittest.TestCase):
    def test_multiple_failed_tools_produce_deterministic_claim_ordering(self) -> None:
        failed_later = make_span(
            "b", status=SpanStatus.ERROR, parent_id="root", start_time=5.0, recorder_sequence_id=5, id="failed-later"
        )
        failed_earlier = make_span(
            "a", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1, id="failed-earlier"
        )
        # Insert in reverse order: the check must not rely on list order.
        trace = make_trace(spans=[failed_later, failed_earlier])
        finding = SwallowedToolErrorCheck().run(trace)
        self.assertEqual(
            [c.fields["tool_span_id"] for c in finding.claims], ["failed-earlier", "failed-later"]
        )


class CapabilityGateTests(unittest.TestCase):
    def test_missing_required_capability_is_not_applicable(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        trace = make_trace(spans=[failed])
        finding = SwallowedToolErrorCheck().run(trace, available_capabilities=frozenset())
        self.assertEqual(finding.status, InternalCheckStatus.NOT_APPLICABLE)
        self.assertEqual(
            set(finding.capability_gaps), {Capability.TOOL_STATUS, Capability.TOOL_CALLS}
        )

    def test_partial_capability_availability_reports_only_missing_ones(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        trace = make_trace(spans=[failed])
        finding = SwallowedToolErrorCheck().run(
            trace, available_capabilities=frozenset({Capability.TOOL_STATUS})
        )
        self.assertEqual(finding.status, InternalCheckStatus.NOT_APPLICABLE)
        self.assertEqual(finding.capability_gaps, [Capability.TOOL_CALLS])

    def test_no_fabricated_inconclusive_when_capabilities_present(self) -> None:
        """V1's operation identity is `span.name`, always present on any
        valid Span (a Phase 1 construction invariant) -- so once the
        required capabilities are declared available, this check has no
        genuine per-Trace reason to report INCONCLUSIVE. It only ever
        resolves to FLAGGED or NO_FINDING from the evidence itself. This
        documents that deliberately, per the instruction not to invent
        unnecessary INCONCLUSIVE cases when the required data is present.
        """
        empty_trace = make_trace(spans=[])
        finding = SwallowedToolErrorCheck().run(
            empty_trace, available_capabilities=frozenset({Capability.TOOL_STATUS, Capability.TOOL_CALLS})
        )
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)


class ClaimDisciplineTests(unittest.TestCase):
    def test_claims_contain_only_factual_fields(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        trace = make_trace(spans=[failed])
        finding = SwallowedToolErrorCheck().run(trace)
        claim = finding.claims[0]
        self.assertEqual(
            set(claim.fields.keys()),
            {"tool_span_id", "operation", "error_status", "retry_or_fallback_observed"},
        )
        self.assertEqual(claim.fields["error_status"], "ERROR")
        self.assertEqual(claim.fields["retry_or_fallback_observed"], False)

    def test_no_causal_or_intent_language(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        trace = make_trace(spans=[failed])
        finding = SwallowedToolErrorCheck().run(trace)
        serialized = json.dumps(finding.to_dict()).lower()
        for banned in (
            "cause",
            "caused",
            "root_cause",
            "intentional",
            "ignored",
            "should have",
            "incorrect",
            "bug",
        ):
            self.assertNotIn(banned, serialized)


class DeterminismTests(unittest.TestCase):
    def test_same_trace_run_twice_same_analytical_content(self) -> None:
        failed = make_span(
            "get_order", status=SpanStatus.ERROR, parent_id="root", start_time=1.0, recorder_sequence_id=1
        )
        trace = make_trace(spans=[failed])
        check = SwallowedToolErrorCheck()
        self.assertEqual(finding_content(check.run(trace)), finding_content(check.run(trace)))


if __name__ == "__main__":
    unittest.main()
