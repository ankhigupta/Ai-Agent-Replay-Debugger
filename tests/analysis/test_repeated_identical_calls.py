"""Tests for RepeatedIdenticalCallsCheck."""

from __future__ import annotations

import unittest

from agentdebug.analysis.checks.repeated_identical_calls import RepeatedIdenticalCallsCheck
from agentdebug.core.enums import Capability, InternalCheckStatus
from tests.analysis._helpers import finding_content, make_span, make_trace


def _call(name: str, input_: object, start: float, seq: int) -> object:
    return make_span(name, input=input_, start_time=start, recorder_sequence_id=seq)


class RunLengthTests(unittest.TestCase):
    def test_exactly_two_identical_calls_is_no_finding(self) -> None:
        spans = [_call("get_order", {"id": "ORD-42"}, i, i) for i in range(2)]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)

    def test_exactly_three_identical_consecutive_calls_is_flagged(self) -> None:
        spans = [_call("get_order", {"id": "ORD-42"}, i, i) for i in range(3)]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(len(finding.claims), 1)
        self.assertEqual(finding.claims[0].fields["occurrence_count"], 3)
        self.assertEqual(finding.claims[0].span_ids, [s.id for s in spans])

    def test_four_or_more_identical_calls_is_flagged_single_claim(self) -> None:
        spans = [_call("get_order", {"id": "ORD-42"}, i, i) for i in range(5)]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(len(finding.claims), 1)
        self.assertEqual(finding.claims[0].fields["occurrence_count"], 5)

    def test_non_consecutive_identical_calls_is_no_finding(self) -> None:
        # A A B A: the repeated "A" calls are split by a "B", so the
        # longest run of identical consecutive calls is length 2.
        spans = [
            _call("get_order", {"id": "ORD-42"}, 0, 0),
            _call("get_order", {"id": "ORD-42"}, 1, 1),
            _call("get_weather", {"city": "NYC"}, 2, 2),
            _call("get_order", {"id": "ORD-42"}, 3, 3),
        ]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)


class IdentityTests(unittest.TestCase):
    def test_different_inputs_is_no_finding(self) -> None:
        spans = [_call("get_order", {"id": f"ORD-{i}"}, i, i) for i in range(3)]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)

    def test_different_tool_names_is_no_finding(self) -> None:
        spans = [_call(f"tool_{i}", {"id": "ORD-42"}, i, i) for i in range(3)]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)

    def test_dict_key_order_is_canonicalized(self) -> None:
        spans = [
            _call("get_order", {"id": "ORD-42", "retries": 0}, 0, 0),
            _call("get_order", {"retries": 0, "id": "ORD-42"}, 1, 1),
            _call("get_order", {"id": "ORD-42", "retries": 0}, 2, 2),
        ]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)

    def test_list_order_is_meaningful_not_canonicalized(self) -> None:
        spans = [
            _call("get_order", {"ids": ["a", "b"]}, 0, 0),
            _call("get_order", {"ids": ["b", "a"]}, 1, 1),
            _call("get_order", {"ids": ["a", "b"]}, 2, 2),
        ]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        # No run of 3 *consecutive identical* calls: list order differs
        # between calls, so each is a distinct canonical identity.
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)


class CapabilityGateTests(unittest.TestCase):
    def test_tool_calls_missing_is_not_applicable(self) -> None:
        spans = [_call("get_order", {"id": "ORD-42"}, i, i) for i in range(3)]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace, available_capabilities=frozenset())
        self.assertEqual(finding.status, InternalCheckStatus.NOT_APPLICABLE)
        self.assertEqual(finding.capability_gaps, [Capability.TOOL_CALLS])

    def test_no_finding_is_never_reported_as_inconclusive(self) -> None:
        trace = make_trace(spans=[])
        finding = RepeatedIdenticalCallsCheck().run(
            trace, available_capabilities=frozenset({Capability.TOOL_CALLS})
        )
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)


class ClaimDisciplineTests(unittest.TestCase):
    def test_claims_contain_only_factual_fields(self) -> None:
        spans = [_call("get_order", {"id": "ORD-42"}, i, i) for i in range(3)]
        trace = make_trace(spans=spans)
        finding = RepeatedIdenticalCallsCheck().run(trace)
        claim = finding.claims[0]
        self.assertEqual(set(claim.fields.keys()), {"operation", "input", "occurrence_count"})
        self.assertEqual(claim.fields["operation"], "get_order")
        self.assertEqual(claim.fields["input"], {"id": "ORD-42"})


class DeterminismTests(unittest.TestCase):
    def test_same_trace_run_twice_same_analytical_content(self) -> None:
        spans = [_call("get_order", {"id": "ORD-42"}, i, i) for i in range(3)]
        trace = make_trace(spans=spans)
        check = RepeatedIdenticalCallsCheck()
        self.assertEqual(finding_content(check.run(trace)), finding_content(check.run(trace)))


if __name__ == "__main__":
    unittest.main()
