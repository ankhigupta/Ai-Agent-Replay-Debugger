"""Tests for TraceIncompletenessCheck."""

from __future__ import annotations

import json
import unittest

from agentdebug.analysis.checks.trace_incompleteness import TraceIncompletenessCheck
from agentdebug.core.enums import Capability, InternalCheckStatus
from tests.analysis._helpers import finding_content, make_span, make_trace


class ValidReferencesTests(unittest.TestCase):
    def test_valid_parent_references_is_no_finding(self) -> None:
        parent = make_span("agent", start_time=0.0, recorder_sequence_id=0, id="parent")
        child = make_span("get_order", parent_id="parent", start_time=1.0, recorder_sequence_id=1)
        trace = make_trace(spans=[parent, child])
        finding = TraceIncompletenessCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)

    def test_no_spans_is_no_finding(self) -> None:
        trace = make_trace(spans=[])
        finding = TraceIncompletenessCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)


class DanglingParentTests(unittest.TestCase):
    def test_dangling_parent_reference_is_flagged(self) -> None:
        span = make_span(
            "get_order", parent_id="missing-span", start_time=0.0, recorder_sequence_id=0, id="span-2"
        )
        trace = make_trace(spans=[span])
        finding = TraceIncompletenessCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(len(finding.claims), 1)
        self.assertEqual(finding.claims[0].fields, {"span_id": "span-2", "missing_parent_id": "missing-span"})
        self.assertEqual(finding.claims[0].span_ids, ["span-2"])

    def test_multiple_dangling_parents_deterministic_claims(self) -> None:
        later = make_span(
            "b", parent_id="ghost-b", start_time=5.0, recorder_sequence_id=5, id="later"
        )
        earlier = make_span(
            "a", parent_id="ghost-a", start_time=1.0, recorder_sequence_id=1, id="earlier"
        )
        # Insert in reverse order: the check must not rely on list order.
        trace = make_trace(spans=[later, earlier])
        finding = TraceIncompletenessCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual([c.fields["span_id"] for c in finding.claims], ["earlier", "later"])


class CapabilityGateTests(unittest.TestCase):
    def test_missing_required_capability_is_not_applicable(self) -> None:
        span = make_span("get_order", parent_id="missing", start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        finding = TraceIncompletenessCheck().run(trace, available_capabilities=frozenset())
        self.assertEqual(finding.status, InternalCheckStatus.NOT_APPLICABLE)
        self.assertEqual(finding.capability_gaps, [Capability.SPAN_IO])


class ClaimDisciplineTests(unittest.TestCase):
    def test_no_causal_or_validity_language(self) -> None:
        span = make_span("get_order", parent_id="missing-span", start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        finding = TraceIncompletenessCheck().run(trace)
        serialized = json.dumps(finding.to_dict()).lower()
        for banned in ("invalid", "unreliable", "caused", "cause", "broken", "corrupt"):
            self.assertNotIn(banned, serialized)


class DeterminismTests(unittest.TestCase):
    def test_same_trace_run_twice_same_analytical_content(self) -> None:
        span = make_span("get_order", parent_id="missing-span", start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        check = TraceIncompletenessCheck()
        self.assertEqual(finding_content(check.run(trace)), finding_content(check.run(trace)))


if __name__ == "__main__":
    unittest.main()
