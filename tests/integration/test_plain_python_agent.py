"""End-to-end tests: plain-Python agent -> Recorder -> Trace -> checks
+ localization -> Findings.

Exercises the real demo in `examples/plain_python_agent.py` rather than
re-implementing agent/trace construction here, so these tests prove the
same code path a developer running the demo actually sees.
"""

from __future__ import annotations

import json
import unittest

from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer, LocalizationTarget
from agentdebug.core.enums import InternalCheckStatus
from examples.plain_python_agent import (
    Scenario,
    build_localization_failure_signal,
    build_trace,
    run_checks,
)

_BANNED_LANGUAGE = (
    "caused",
    "cause",
    "root cause",
    "root_cause",
    "responsible",
    "incorrect",
    "wrong",
    "hallucinat",
    "intended",
    "should have",
    "led to",
    "resulted in",
    "because",
    "reason",
)


class CleanScenarioTests(unittest.TestCase):
    def test_1_trace_created_with_spans_and_no_unexpected_findings(self) -> None:
        trace = build_trace(Scenario.CLEAN)
        self.assertGreater(len(trace.spans), 0)
        findings = run_checks(trace)
        for check_id, finding in findings.items():
            self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING, msg=check_id)

    def test_7_clean_trace_produces_no_false_positives(self) -> None:
        trace = build_trace(Scenario.CLEAN)
        findings = run_checks(trace)
        flagged = [check_id for check_id, finding in findings.items() if finding.status == InternalCheckStatus.FLAGGED]
        self.assertEqual(flagged, [])


class SwallowedToolErrorScenarioTests(unittest.TestCase):
    def test_2_swallowed_tool_error_is_flagged_with_relevant_claim(self) -> None:
        trace = build_trace(Scenario.SWALLOWED_TOOL_ERROR)
        findings = run_checks(trace)
        finding = findings["swallowed_tool_error"]
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(len(finding.claims), 1)
        claimed_span_id = finding.claims[0].fields["tool_span_id"]
        tool_span = trace.get_span(claimed_span_id)
        self.assertIsNotNone(tool_span)
        self.assertEqual(tool_span.name, "get_order")


class RepeatedToolCallScenarioTests(unittest.TestCase):
    def test_3_repeated_tool_calls_is_flagged(self) -> None:
        trace = build_trace(Scenario.REPEATED_TOOL_CALL)
        findings = run_checks(trace)
        finding = findings["repeated_identical_calls"]
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertGreaterEqual(finding.claims[0].fields["occurrence_count"], 3)


class LocalizationFailureScenarioTests(unittest.TestCase):
    def test_4_localization_failure_is_flagged_with_correct_span(self) -> None:
        trace = build_trace(Scenario.LOCALIZATION_FAILURE)
        failure_signal = build_localization_failure_signal()
        target = LocalizationTarget(value=False, field_path="refund_eligible")

        finding = FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)

        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.claims[0].type.value, "VALUE_FIRST_OBSERVED")
        self.assertEqual(finding.triggering_signal_id, failure_signal.id)

        state_span = next(s for s in trace.spans if s.name == "update_refund_state")
        self.assertEqual(finding.claims[0].span_ids, [state_span.id])

    def test_5_target_not_observed_is_no_finding(self) -> None:
        trace = build_trace(Scenario.CLEAN)
        failure_signal = build_localization_failure_signal()
        target = LocalizationTarget(value="this-value-never-appears-anywhere")

        finding = FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)

        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)
        self.assertEqual(len(finding.claims), 1)
        self.assertEqual(finding.claims[0].type.value, "VALUE_NOT_OBSERVED")


class IncompleteTraceScenarioTests(unittest.TestCase):
    def test_6_incomplete_trace_is_flagged(self) -> None:
        trace = build_trace(Scenario.INCOMPLETE_TRACE)
        findings = run_checks(trace)
        finding = findings["trace_incompleteness"]
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.claims[0].fields["missing_parent_id"], "never-recorded-parent")
        # The other checks should not be incidentally triggered by the
        # orphan span added for this scenario.
        self.assertEqual(findings["swallowed_tool_error"].status, InternalCheckStatus.NO_FINDING)
        self.assertEqual(findings["repeated_identical_calls"].status, InternalCheckStatus.NO_FINDING)


class NoCausalLanguageTests(unittest.TestCase):
    """Every Finding the demo can produce, across every scenario and the
    localization path, must contain only observational language -- never
    causal, intent, or correctness claims. This inspects actual Claim
    fields, not one hard-coded sentence."""

    def test_no_causal_or_intent_language_anywhere_in_demo_output(self) -> None:
        for scenario in Scenario:
            trace = build_trace(scenario)
            findings = list(run_checks(trace).values())

            if scenario is Scenario.LOCALIZATION_FAILURE:
                failure_signal = build_localization_failure_signal()
                target = LocalizationTarget(value=False, field_path="refund_eligible")
                findings.append(FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target))

            for finding in findings:
                serialized = json.dumps(finding.to_dict()).lower()
                for banned in _BANNED_LANGUAGE:
                    self.assertNotIn(
                        banned,
                        serialized,
                        msg=f"scenario={scenario.value} check={finding.check_id} contains {banned!r}",
                    )


if __name__ == "__main__":
    unittest.main()
