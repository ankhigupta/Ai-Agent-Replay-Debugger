"""FirstObservedOccurrenceAnalyzer: failure-driven deterministic localization.

Given a FailureSignal (and, optionally, an explicit LocalizationTarget),
searches a Trace's already-captured evidence — pre-run context, then each
span's input and output, in deterministic (start_time, recorder_sequence_id)
order — for the first point at which the target value was observed.

This is LOCALIZATION, not causal diagnosis. A Finding from this analyzer
says only "value X was first observed here" (FLAGGED / VALUE_FIRST_OBSERVED),
"value X was not observed in the available evidence" (NO_FINDING /
VALUE_NOT_OBSERVED — NOT a claim that X is absent from reality, only that
this analyzer's search over this trace's evidence didn't find it), or "the
available evidence does not resolve this" (INCONCLUSIVE, when required
capabilities are only partially available) / "this trace's integration
cannot expose the evidence this analyzer needs at all" (NOT_APPLICABLE, when
none of it is available). It never says why a value appeared, whether it
was correct, or that any span caused anything.

Evidence buckets and why input/output of one span aren't individually
ordered: Phase 1's `Span` model carries exactly one `start_time`/`end_time`
boundary per span — there is no finer-grained timestamp distinguishing
"when the input became known" from "when the output was produced" within
that one span. Rather than impose an arbitrary convention (e.g. "input
always before output"), both are treated as one unordered bucket: if the
target matches in only one of them, that's a UNIQUE first observation; if
it matches in both, that's reported as AMBIGUOUS rather than guessed. The
three `TraceContext` fields searched as pre-run evidence (`config`,
`initial_state`, `user_input`) are treated the same way, for the same
reason — Phase 1's `TraceContext` carries no sub-field ordering either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentdebug.analysis.check import CheckDefinition, missing_capabilities
from agentdebug.analysis.localization.base import LocalizationAnalyzer, LocalizationTarget
from agentdebug.analysis.ordering import ordered_spans
from agentdebug.core.claim import Claim
from agentdebug.core.enums import Capability, ClaimType, DiagnosisConfidence, InternalCheckStatus
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.serialization import is_json_serializable
from agentdebug.core.trace import Trace


def _json_safe(value: object) -> object:
    """Claim.fields must be JSON-compatible (a Phase 1 invariant), but a
    LocalizationTarget's `value` may legitimately be a non-JSON type such
    as `datetime.date` (e.g. for TEMPORAL-typed matching). Such values are
    reported in claims via `str()` instead of being dropped or raising."""
    return value if is_json_serializable(value) else str(value)

# Fixed, deterministic enumeration order for the three TraceContext fields
# searched as pre-run evidence. They form one unordered bucket for
# ambiguity-detection purposes (see module docstring) but are always
# enumerated in this order for reproducible claim content.
_CONTEXT_FIELDS = ("config", "initial_state", "user_input")


@dataclass(frozen=True)
class _Observation:
    category: str  # "context.<field>" | "span.input" | "span.output"
    span_id: Optional[str]
    value: object


def _context_bucket(trace: Trace) -> list[_Observation]:
    values = {
        "config": trace.context.config,
        "initial_state": trace.context.initial_state,
        "user_input": trace.context.user_input,
    }
    return [
        _Observation(category=f"context.{field}", span_id=None, value=values[field])
        for field in _CONTEXT_FIELDS
    ]


def _span_buckets(trace: Trace) -> list[list[_Observation]]:
    return [
        [
            _Observation(category="span.input", span_id=span.id, value=span.input),
            _Observation(category="span.output", span_id=span.id, value=span.output),
        ]
        for span in ordered_spans(trace.spans)
    ]


class FirstObservedOccurrenceAnalyzer(LocalizationAnalyzer):
    """Finds the first point in a Trace's evidence where a failure
    target's value was observed. See module docstring for status/claim
    semantics and the bucket-based ambiguity model.
    """

    definition = CheckDefinition(
        id="first_observed_occurrence",
        basis=(
            "Searches Trace.context and all span input/output, in "
            "deterministic (start_time, recorder_sequence_id) order, for "
            "the first observation of a given failure target's value."
        ),
        required_capabilities=(Capability.PRE_RUN_CONTEXT, Capability.SPAN_IO),
        known_false_positive_modes=(
            "The target value may have existed before this trace's "
            "recorded evidence begins (e.g. in an upstream system never "
            "captured here), making the reported 'first' observation only "
            "first within what this trace actually captured.",
            "A span's input and output share one start_time/end_time "
            "boundary with no finer sub-ordering evidence; if the target "
            "matches both, this analyzer reports ambiguity rather than "
            "guessing which one is 'first'.",
        ),
    )

    def run(
        self,
        trace: Trace,
        failure_signal: FailureSignal,
        *,
        target: Optional[LocalizationTarget] = None,
        available_capabilities: Optional[frozenset[Capability]] = None,
    ) -> Finding:
        resolved_target = (
            target if target is not None else LocalizationTarget.from_failure_signal(failure_signal)
        )

        missing = missing_capabilities(self.definition, available_capabilities)
        has_context = Capability.PRE_RUN_CONTEXT not in missing
        has_spans = Capability.SPAN_IO not in missing

        # Neither evidence source this analyzer depends on is available at
        # all -> the integration genuinely cannot support this analysis.
        if not has_context and not has_spans:
            return Finding(
                check_id=self.definition.id,
                status=InternalCheckStatus.NOT_APPLICABLE,
                capability_gaps=list(missing),
                triggering_signal_id=failure_signal.id,
            )

        buckets: list[list[_Observation]] = []
        if has_context:
            buckets.append(_context_bucket(trace))
        if has_spans:
            buckets.extend(_span_buckets(trace))

        for bucket in buckets:
            matched = [obs for obs in bucket if resolved_target.matches(obs.value)]
            if not matched:
                continue
            if len(matched) == 1:
                return self._flagged(resolved_target, [matched[0]], DiagnosisConfidence.UNIQUE, failure_signal)
            return self._flagged(resolved_target, matched, DiagnosisConfidence.AMBIGUOUS, failure_signal)

        # Not found anywhere we could search. If one evidence source was
        # genuinely unavailable (a real blind spot), we can't rule out
        # that the target was observed there -> INCONCLUSIVE, not a
        # confident "not observed". Only when both sources were fully
        # searched does "not found" become a confident NO_FINDING.
        if missing:
            return Finding(
                check_id=self.definition.id,
                status=InternalCheckStatus.INCONCLUSIVE,
                capability_gaps=list(missing),
                triggering_signal_id=failure_signal.id,
            )

        not_observed_claim = Claim(
            type=ClaimType.VALUE_NOT_OBSERVED,
            fields={
                "target_value": _json_safe(resolved_target.value),
                "field_path": resolved_target.field_path,
            },
        )
        return Finding(
            check_id=self.definition.id,
            status=InternalCheckStatus.NO_FINDING,
            claims=[not_observed_claim],
            diagnosis_confidence=DiagnosisConfidence.NONE,
            triggering_signal_id=failure_signal.id,
        )

    def _flagged(
        self,
        target: LocalizationTarget,
        observations: list[_Observation],
        confidence: DiagnosisConfidence,
        failure_signal: FailureSignal,
    ) -> Finding:
        span_ids = sorted({obs.span_id for obs in observations if obs.span_id is not None})
        categories = sorted(obs.category for obs in observations)
        claim = Claim(
            type=ClaimType.VALUE_FIRST_OBSERVED,
            span_ids=span_ids,
            fields={
                "target_value": _json_safe(target.value),
                "field_path": target.field_path,
                "observed_at": categories if len(categories) > 1 else categories[0],
                "candidate_count": len(observations),
            },
        )
        return Finding(
            check_id=self.definition.id,
            status=InternalCheckStatus.FLAGGED,
            claims=[claim],
            diagnosis_confidence=confidence,
            triggering_signal_id=failure_signal.id,
        )
