"""Tests for CheckDefinition and the TraceCheck base contract."""

from __future__ import annotations

import dataclasses
import unittest

from agentdebug.analysis.check import CheckDefinition, TraceCheck, missing_capabilities
from agentdebug.analysis.checks import (
    RepeatedIdenticalCallsCheck,
    SwallowedToolErrorCheck,
    TraceIncompletenessCheck,
)
from agentdebug.core.enums import Capability, SpanStatus
from tests.analysis._helpers import finding_content, make_span, make_trace

_ALL_CHECK_CLASSES = (SwallowedToolErrorCheck, RepeatedIdenticalCallsCheck, TraceIncompletenessCheck)


class CheckDefinitionTests(unittest.TestCase):
    def test_construction(self) -> None:
        definition = CheckDefinition(
            id="example_check",
            basis="Detects an example pattern.",
            required_capabilities=(Capability.SPAN_IO,),
            known_false_positive_modes=("Example limitation.",),
        )
        self.assertEqual(definition.id, "example_check")
        self.assertEqual(definition.basis, "Detects an example pattern.")
        self.assertEqual(definition.required_capabilities, (Capability.SPAN_IO,))
        self.assertEqual(definition.known_false_positive_modes, ("Example limitation.",))

    def test_required_capabilities_default_empty(self) -> None:
        self.assertEqual(CheckDefinition(id="c", basis="b").required_capabilities, ())

    def test_known_false_positive_modes_default_empty(self) -> None:
        self.assertEqual(CheckDefinition(id="c", basis="b").known_false_positive_modes, ())

    def test_frozen(self) -> None:
        definition = CheckDefinition(id="c", basis="b")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            definition.id = "other"  # type: ignore[misc]


class MissingCapabilitiesTests(unittest.TestCase):
    def test_none_available_means_nothing_missing(self) -> None:
        definition = CheckDefinition(id="c", basis="b", required_capabilities=(Capability.TOOL_CALLS,))
        self.assertEqual(missing_capabilities(definition, None), [])

    def test_missing_detected_in_definition_order(self) -> None:
        definition = CheckDefinition(
            id="c", basis="b", required_capabilities=(Capability.TOOL_STATUS, Capability.TOOL_CALLS)
        )
        self.assertEqual(
            missing_capabilities(definition, frozenset()),
            [Capability.TOOL_STATUS, Capability.TOOL_CALLS],
        )

    def test_partial_availability(self) -> None:
        definition = CheckDefinition(
            id="c", basis="b", required_capabilities=(Capability.TOOL_STATUS, Capability.TOOL_CALLS)
        )
        self.assertEqual(
            missing_capabilities(definition, frozenset({Capability.TOOL_STATUS})),
            [Capability.TOOL_CALLS],
        )

    def test_all_available_means_nothing_missing(self) -> None:
        definition = CheckDefinition(id="c", basis="b", required_capabilities=(Capability.TOOL_CALLS,))
        self.assertEqual(missing_capabilities(definition, frozenset({Capability.TOOL_CALLS})), [])


class TraceCheckInterfaceTests(unittest.TestCase):
    def test_cannot_instantiate_abstract_base(self) -> None:
        with self.assertRaises(TypeError):
            TraceCheck()  # type: ignore[abstract]

    def test_concrete_checks_expose_definition(self) -> None:
        for check_cls in _ALL_CHECK_CLASSES:
            self.assertIsInstance(check_cls().definition, CheckDefinition)

    def test_check_ids_are_distinct_and_match_spec(self) -> None:
        ids = {cls().definition.id for cls in _ALL_CHECK_CLASSES}
        self.assertEqual(
            ids, {"swallowed_tool_error", "repeated_identical_calls", "trace_incompleteness"}
        )

    def test_finding_check_id_matches_definition_id(self) -> None:
        trace = make_trace()
        for check_cls in _ALL_CHECK_CLASSES:
            check = check_cls()
            finding = check.run(trace)
            self.assertEqual(finding.check_id, check.definition.id)


class DeterminismTests(unittest.TestCase):
    def test_same_trace_run_twice_same_analytical_content(self) -> None:
        trace = make_trace(
            spans=[make_span("s1", parent_id="missing", start_time=0.0, recorder_sequence_id=0)]
        )
        check_a = TraceIncompletenessCheck()
        check_b = TraceIncompletenessCheck()
        self.assertEqual(finding_content(check_a.run(trace)), finding_content(check_b.run(trace)))

    def test_checks_are_stateless_across_unrelated_traces(self) -> None:
        trace_with_error = make_trace(
            spans=[make_span("get_order", status=SpanStatus.ERROR, start_time=0.0, recorder_sequence_id=0)]
        )
        trace_without_error = make_trace(
            spans=[make_span("get_order", status=SpanStatus.OK, start_time=0.0, recorder_sequence_id=0)]
        )
        check = SwallowedToolErrorCheck()
        first = check.run(trace_with_error)
        second = check.run(trace_without_error)
        third = check.run(trace_with_error)
        self.assertEqual(finding_content(first), finding_content(third))
        self.assertNotEqual(finding_content(first)["status"], finding_content(second)["status"])


if __name__ == "__main__":
    unittest.main()
