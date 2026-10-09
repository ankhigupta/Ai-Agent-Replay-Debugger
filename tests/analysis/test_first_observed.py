"""Tests for FirstObservedOccurrenceAnalyzer (lettered per the Phase 4 spec)."""

from __future__ import annotations

import datetime
import json
import unittest

from agentdebug.analysis.localization import FirstObservedOccurrenceAnalyzer, LocalizationTarget, ValueType
from agentdebug.core.context import TraceContext
from agentdebug.core.enums import Capability, DiagnosisConfidence, ExternalStatus, InternalCheckStatus, SpanKind
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.recording.recorder import Recorder
from tests.analysis._helpers import finding_content, make_span, make_trace

ANALYZER = FirstObservedOccurrenceAnalyzer


def _signal(actual: object = None) -> FailureSignal:
    return FailureSignal(source="pytest", verdict=ExternalStatus.FAILED, expected=True, actual=actual)


class ContextAndSpanLocationTests(unittest.TestCase):
    def test_a_value_found_in_initial_context(self) -> None:
        trace = make_trace(context=TraceContext(initial_state={"refund_eligible": False}))
        target = LocalizationTarget(value=False, field_path="refund_eligible")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.claims[0].fields["observed_at"], "context.initial_state")
        self.assertEqual(finding.diagnosis_confidence, DiagnosisConfidence.UNIQUE)

    def test_b_value_first_found_in_span_input(self) -> None:
        span = make_span("get_order", input={"order_id": "ORD-42"}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value="ORD-42", field_path="order_id")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.claims[0].fields["observed_at"], "span.input")
        self.assertEqual(finding.claims[0].span_ids, [span.id])

    def test_c_value_first_found_in_span_output(self) -> None:
        span = make_span("get_order", output={"refund_eligible": True}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value=True, field_path="refund_eligible")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.claims[0].fields["observed_at"], "span.output")

    def test_context_takes_precedence_over_spans(self) -> None:
        span = make_span("get_order", output={"v": 1}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(context=TraceContext(initial_state={"v": 1}), spans=[span])
        target = LocalizationTarget(value=1, field_path="v")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.claims[0].fields["observed_at"], "context.initial_state")
        self.assertEqual(finding.claims[0].span_ids, [])


class OccurrenceOrderingTests(unittest.TestCase):
    def test_d_multiple_occurrences_deterministic_first(self) -> None:
        span1 = make_span("a", output={"v": 1}, start_time=0.0, recorder_sequence_id=0, id="first")
        span2 = make_span("b", output={"v": 1}, start_time=1.0, recorder_sequence_id=1, id="second")
        span3 = make_span("c", output={"v": 1}, start_time=2.0, recorder_sequence_id=2, id="third")
        trace = make_trace(spans=[span3, span1, span2])  # insertion order scrambled
        target = LocalizationTarget(value=1, field_path="v")
        signal = _signal()
        analyzer = ANALYZER()
        finding1 = analyzer.run(trace, signal, target=target)
        finding2 = analyzer.run(trace, signal, target=target)
        self.assertEqual(finding1.claims[0].span_ids, ["first"])
        self.assertEqual(finding_content(finding1), finding_content(finding2))

    def test_e_same_start_time_recorder_sequence_id_breaks_tie(self) -> None:
        lower_seq = make_span("b", output={"v": 1}, start_time=5.0, recorder_sequence_id=1, id="first-by-seq")
        higher_seq = make_span("a", output={"v": 1}, start_time=5.0, recorder_sequence_id=2, id="second-by-seq")
        trace = make_trace(spans=[higher_seq, lower_seq])  # insertion order reversed
        target = LocalizationTarget(value=1, field_path="v")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.diagnosis_confidence, DiagnosisConfidence.UNIQUE)
        self.assertEqual(finding.claims[0].span_ids, ["first-by-seq"])


class NotObservedTests(unittest.TestCase):
    def test_f_value_not_observed(self) -> None:
        span = make_span("get_order", output={"refund_eligible": True}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value="something-else-entirely")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)
        self.assertEqual(len(finding.claims), 1)
        self.assertEqual(finding.claims[0].type.value, "VALUE_NOT_OBSERVED")
        self.assertEqual(finding.diagnosis_confidence, DiagnosisConfidence.NONE)


class StructuralMatchingTests(unittest.TestCase):
    def test_g_nested_dictionary_equality(self) -> None:
        nested = {"refund": {"eligible": False, "amount": 100}}
        span = make_span("s", output=nested, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value={"refund": {"amount": 100, "eligible": False}})
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)

    def test_h_dict_insertion_order_does_not_affect_matching(self) -> None:
        span = make_span("s", output={"eligible": True, "amount": 500}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value={"amount": 500, "eligible": True})
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)

    def test_i_list_order_remains_significant(self) -> None:
        span = make_span("s", output=[3, 2, 1], start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value=[1, 2, 3])
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)

    def test_j_type_incompatible_values_do_not_match(self) -> None:
        span = make_span("s", output={"v": 0}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value=False, field_path="v")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)


class TypeNormalizationTests(unittest.TestCase):
    def test_k_monetary_normalization(self) -> None:
        span = make_span("charge", output={"amount": "$1,200.00"}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value=1200.00, field_path="amount", value_type=ValueType.MONETARY)
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)

    def test_l_temporal_normalization(self) -> None:
        span = make_span(
            "schedule", output={"due": "2026-01-01"}, start_time=0.0, recorder_sequence_id=0
        )
        trace = make_trace(spans=[span])
        target = LocalizationTarget(
            value=datetime.date(2026, 1, 1), field_path="due", value_type=ValueType.TEMPORAL
        )
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)

    def test_m_identifier_normalization(self) -> None:
        span = make_span("lookup", output={"user_id": 123}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value="123", field_path="user_id", value_type=ValueType.IDENTIFIER)
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)


class MultiHopProvenanceTests(unittest.TestCase):
    def _build_chain_trace(self):
        span1 = make_span("step1", output={"amount": 100}, start_time=0.0, recorder_sequence_id=0, id="s1")
        span2 = make_span(
            "step2", input={"amount": 100}, output={"amount": 120}, start_time=1.0, recorder_sequence_id=1, id="s2"
        )
        span3 = make_span("step3", input={"amount": 120}, start_time=2.0, recorder_sequence_id=2, id="s3")
        return make_trace(spans=[span1, span2, span3])

    def test_n_no_arithmetic_inference(self) -> None:
        trace = self._build_chain_trace()
        # 100 + 20 = 120 is NOT an observed value anywhere as "220"; the
        # analyzer must not compute or infer it.
        target = LocalizationTarget(value=220, field_path="amount")
        finding = ANALYZER().run(trace, _signal(), target=target)
        self.assertEqual(finding.status, InternalCheckStatus.NO_FINDING)

    def test_o_multi_hop_observation_only_from_actual_values(self) -> None:
        trace = self._build_chain_trace()
        analyzer = ANALYZER()

        first_value = analyzer.run(trace, _signal(), target=LocalizationTarget(value=100, field_path="amount"))
        self.assertEqual(first_value.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(first_value.claims[0].span_ids, ["s1"])

        second_value = analyzer.run(trace, _signal(), target=LocalizationTarget(value=120, field_path="amount"))
        self.assertEqual(second_value.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(second_value.claims[0].span_ids, ["s2"])
        # The claim about 120 says only where 120 was observed -- nothing
        # about it being derived from the earlier 100.
        self.assertNotIn("derived", json.dumps(second_value.claims[0].fields).lower())
        self.assertNotIn("from", second_value.claims[0].fields.keys())


class CapabilityGateTests(unittest.TestCase):
    def test_p_missing_both_capabilities_is_not_applicable(self) -> None:
        span = make_span("s", output={"v": 1}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value=1, field_path="v")
        finding = ANALYZER().run(trace, _signal(), target=target, available_capabilities=frozenset())
        self.assertEqual(finding.status, InternalCheckStatus.NOT_APPLICABLE)
        self.assertEqual(set(finding.capability_gaps), {Capability.PRE_RUN_CONTEXT, Capability.SPAN_IO})

    def test_q_partial_capability_with_no_match_is_inconclusive(self) -> None:
        # SPAN_IO available, PRE_RUN_CONTEXT not -- the target happens to
        # live only in context, which this run cannot see, so "not found
        # in what we could see" must not be reported as a confident
        # NO_FINDING.
        trace = make_trace(context=TraceContext(initial_state={"v": 1}), spans=[])
        target = LocalizationTarget(value=1, field_path="v")
        finding = ANALYZER().run(
            trace, _signal(), target=target, available_capabilities=frozenset({Capability.SPAN_IO})
        )
        self.assertEqual(finding.status, InternalCheckStatus.INCONCLUSIVE)
        self.assertEqual(finding.capability_gaps, [Capability.PRE_RUN_CONTEXT])

    def test_partial_capability_with_match_in_available_evidence_is_still_flagged(self) -> None:
        span = make_span("s", output={"v": 1}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value=1, field_path="v")
        finding = ANALYZER().run(
            trace, _signal(), target=target, available_capabilities=frozenset({Capability.SPAN_IO})
        )
        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.capability_gaps, [])


class TriggeringSignalTests(unittest.TestCase):
    def test_r_triggering_signal_id_populated_on_every_status(self) -> None:
        span = make_span("s", output={"v": 1}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        analyzer = ANALYZER()

        flagged_signal = _signal()
        flagged = analyzer.run(trace, flagged_signal, target=LocalizationTarget(value=1, field_path="v"))
        self.assertEqual(flagged.triggering_signal_id, flagged_signal.id)

        no_finding_signal = _signal()
        no_finding = analyzer.run(
            trace, no_finding_signal, target=LocalizationTarget(value="nowhere")
        )
        self.assertEqual(no_finding.triggering_signal_id, no_finding_signal.id)

        not_applicable_signal = _signal()
        not_applicable = analyzer.run(
            trace, not_applicable_signal, target=LocalizationTarget(value=1), available_capabilities=frozenset()
        )
        self.assertEqual(not_applicable.triggering_signal_id, not_applicable_signal.id)

    def test_s_multiple_failure_signals_do_not_cross_link(self) -> None:
        span = make_span("s", output={"v": 1}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        analyzer = ANALYZER()
        signal_a = _signal()
        signal_b = _signal()
        finding_a = analyzer.run(trace, signal_a, target=LocalizationTarget(value=1, field_path="v"))
        finding_b = analyzer.run(trace, signal_b, target=LocalizationTarget(value=1, field_path="v"))
        self.assertEqual(finding_a.triggering_signal_id, signal_a.id)
        self.assertEqual(finding_b.triggering_signal_id, signal_b.id)
        self.assertNotEqual(finding_a.triggering_signal_id, finding_b.triggering_signal_id)


class ApiContractTests(unittest.TestCase):
    def test_t_analyzer_requires_a_failure_signal(self) -> None:
        trace = make_trace()
        with self.assertRaises(TypeError):
            ANALYZER().run(trace)  # type: ignore[call-arg]


class ClaimLanguageSafetyTests(unittest.TestCase):
    def test_w_no_causal_or_intent_language(self) -> None:
        span = make_span("s", output={"v": 1}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        finding = ANALYZER().run(trace, _signal(), target=LocalizationTarget(value=1, field_path="v"))
        serialized = json.dumps(finding.to_dict()).lower()
        for banned in (
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
        ):
            self.assertNotIn(banned, serialized)


class DeterminismTests(unittest.TestCase):
    def test_x_same_trace_produces_identical_finding_across_repeated_runs(self) -> None:
        span = make_span("s", output={"v": 1}, start_time=0.0, recorder_sequence_id=0)
        trace = make_trace(spans=[span])
        target = LocalizationTarget(value=1, field_path="v")
        signal = _signal()
        analyzer = ANALYZER()
        first = analyzer.run(trace, signal, target=target)
        second = ANALYZER().run(trace, signal, target=target)
        self.assertEqual(finding_content(first), finding_content(second))


class RecorderIntegrationTests(unittest.TestCase):
    """Recorder -> real Trace -> FailureSignal -> FirstObservedOccurrenceAnalyzer -> Finding."""

    def test_end_to_end_localization_from_recorder_trace(self) -> None:
        recorder = Recorder()
        trace = recorder.start_trace(context=TraceContext(initial_state={"refund_eligible": True}))
        with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}) as span:
            span.set_output({"refund_eligible": True})
        with recorder.span("state_update", kind=SpanKind.STATE) as span:
            span.set_output({"refund_eligible": False})

        failure_signal = FailureSignal(
            source="pytest", verdict=ExternalStatus.FAILED, expected=True, actual=False,
            detail="expected refund_eligible=True but the run ended with False",
        )
        target = LocalizationTarget(value=False, field_path="refund_eligible")

        finding = FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)

        self.assertEqual(finding.status, InternalCheckStatus.FLAGGED)
        self.assertEqual(finding.diagnosis_confidence, DiagnosisConfidence.UNIQUE)
        self.assertEqual(finding.triggering_signal_id, failure_signal.id)
        # Localized to the state_update span's output -- NOT a claim that
        # this span (or anything else) caused the failure.
        state_span_id = trace.spans[1].id
        self.assertEqual(finding.claims[0].span_ids, [state_span_id])
        serialized = json.dumps(finding.to_dict()).lower()
        self.assertNotIn("caused", serialized)
        self.assertNotIn("root_cause", serialized)


if __name__ == "__main__":
    unittest.main()
