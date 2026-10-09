"""Trace: the top-level, framework-independent record of one agent run.

Unlike TraceContext/Span/FailureSignal/Claim/Finding, Trace is mutable:
spans, failure signals, and findings are appended over the life of a run
(later, by a Recorder — not implemented in Phase 1) via the `add_*`
methods, which are the supported mutation path. The underlying lists stay
publicly readable, but callers should not append to them directly, since
only the `add_*` methods validate the cross-field invariants below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentdebug.core.context import TraceContext
from agentdebug.core.enums import ExternalStatus, InternalCheckStatus, InternalRunStatus
from agentdebug.core.exceptions import SchemaVersionMismatchError, TraceValidationError
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.serialization import SCHEMA_VERSION, check_known_keys, require_dict
from agentdebug.core.span import Span

_ALLOWED_KEYS = frozenset(
    {"id", "schema_version", "context", "spans", "failure_signals", "findings"}
)


@dataclass
class Trace:
    """The top-level record of one agent run: context + spans + evidence.

    Construction validates the same cross-field invariants as the `add_*`
    methods (uniqueness of ids, claim/signal referential integrity), so a
    `Trace` built directly from a full set of spans/findings/signals is
    checked exactly like one built incrementally. The one deliberate
    exception is `Span.parent_id`: it is never checked against the
    trace's own span ids, here or in `add_span`, because a dangling parent
    reference is a legitimate state a Trace can be in and must remain
    representable for a later Trace Incompleteness check to detect.
    """

    id: str
    schema_version: str = SCHEMA_VERSION
    context: TraceContext = field(default_factory=TraceContext)
    spans: list[Span] = field(default_factory=list)
    failure_signals: list[FailureSignal] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise TraceValidationError("Trace.id must be a non-empty string")
        self._validate_invariants()

    # -- invariant checking ---------------------------------------------

    def _validate_invariants(self) -> None:
        span_ids = [s.id for s in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise TraceValidationError("Trace contains duplicate Span.id values")

        sequence_ids = [s.recorder_sequence_id for s in self.spans]
        if len(sequence_ids) != len(set(sequence_ids)):
            raise TraceValidationError(
                "Trace contains duplicate Span.recorder_sequence_id values"
            )

        finding_ids = [f.id for f in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise TraceValidationError("Trace contains duplicate Finding.id values")

        signal_ids = [s.id for s in self.failure_signals]
        if len(signal_ids) != len(set(signal_ids)):
            raise TraceValidationError("Trace contains duplicate FailureSignal.id values")

        known_span_ids = set(span_ids)
        known_signal_ids = set(signal_ids)
        for finding in self.findings:
            for claim in finding.claims:
                for span_id in claim.span_ids:
                    if span_id not in known_span_ids:
                        raise TraceValidationError(
                            f"Finding {finding.id!r} claim references unknown span_id {span_id!r}"
                        )
            if (
                finding.triggering_signal_id is not None
                and finding.triggering_signal_id not in known_signal_ids
            ):
                raise TraceValidationError(
                    f"Finding {finding.id!r} triggering_signal_id "
                    f"{finding.triggering_signal_id!r} does not reference an existing FailureSignal"
                )
        # Span.parent_id is intentionally NOT checked here — see class docstring.

    # -- mutation API -----------------------------------------------------

    def add_span(self, span: Span) -> None:
        """Append `span`. Rejects a duplicate `id` or `recorder_sequence_id`
        without mutating the trace."""
        if any(s.id == span.id for s in self.spans):
            raise TraceValidationError(f"Duplicate Span.id: {span.id!r}")
        if any(s.recorder_sequence_id == span.recorder_sequence_id for s in self.spans):
            raise TraceValidationError(
                f"Duplicate Span.recorder_sequence_id: {span.recorder_sequence_id!r}"
            )
        self.spans.append(span)

    def add_failure_signal(self, signal: FailureSignal) -> None:
        """Append `signal`. Rejects a duplicate `id` without mutating the trace."""
        if any(s.id == signal.id for s in self.failure_signals):
            raise TraceValidationError(f"Duplicate FailureSignal.id: {signal.id!r}")
        self.failure_signals.append(signal)

    def add_finding(self, finding: Finding) -> None:
        """Append `finding`. Validates its claim span references and
        triggering-signal reference before mutating the trace, so a
        rejected finding never partially mutates it."""
        if any(f.id == finding.id for f in self.findings):
            raise TraceValidationError(f"Duplicate Finding.id: {finding.id!r}")

        known_span_ids = {s.id for s in self.spans}
        for claim in finding.claims:
            for span_id in claim.span_ids:
                if span_id not in known_span_ids:
                    raise TraceValidationError(
                        f"Finding {finding.id!r} claim references unknown span_id {span_id!r}"
                    )

        if finding.triggering_signal_id is not None:
            known_signal_ids = {s.id for s in self.failure_signals}
            if finding.triggering_signal_id not in known_signal_ids:
                raise TraceValidationError(
                    f"Finding {finding.id!r} triggering_signal_id "
                    f"{finding.triggering_signal_id!r} does not reference an existing FailureSignal"
                )

        self.findings.append(finding)

    # -- lookup -------------------------------------------------------------

    def get_span(self, span_id: str) -> Optional[Span]:
        """Look up a span by id.

        Dataclass equality on `Span` is structural, not identity — two
        spans with identical field values compare equal even if they
        represent different moments in time. Lookup must always go
        through `Span.id`, never `==`/`in` on `Span` objects themselves.
        """
        for s in self.spans:
            if s.id == span_id:
                return s
        return None

    # -- computed status ------------------------------------------------

    @property
    def external_status(self) -> ExternalStatus:
        """Aggregate verdict across all failure signals.

        No signals -> UNKNOWN. Any FAILED -> FAILED. All PASSED -> PASSED.
        Any other mixture (e.g. PASSED + UNKNOWN, with no FAILED) -> UNKNOWN.
        """
        if not self.failure_signals:
            return ExternalStatus.UNKNOWN
        verdicts = {s.verdict for s in self.failure_signals}
        if ExternalStatus.FAILED in verdicts:
            return ExternalStatus.FAILED
        if verdicts == {ExternalStatus.PASSED}:
            return ExternalStatus.PASSED
        return ExternalStatus.UNKNOWN

    @property
    def internal_status(self) -> InternalRunStatus:
        """Aggregate result across all findings.

        Any FLAGGED -> FLAGGED. Else any INCONCLUSIVE -> INCONCLUSIVE.
        Else at least one NO_FINDING -> CLEAN_UNDER_CHECKS. Otherwise
        (no findings, or only NOT_APPLICABLE findings) -> INCONCLUSIVE:
        we never ran a check that actually produced a clean result, so
        there is nothing to call "clean".
        """
        statuses = [f.status for f in self.findings]
        if InternalCheckStatus.FLAGGED in statuses:
            return InternalRunStatus.FLAGGED
        if InternalCheckStatus.INCONCLUSIVE in statuses:
            return InternalRunStatus.INCONCLUSIVE
        if InternalCheckStatus.NO_FINDING in statuses:
            return InternalRunStatus.CLEAN_UNDER_CHECKS
        return InternalRunStatus.INCONCLUSIVE

    @property
    def checks_run(self) -> int:
        """Findings whose check actually executed (i.e. was not NOT_APPLICABLE)."""
        return sum(1 for f in self.findings if f.status != InternalCheckStatus.NOT_APPLICABLE)

    @property
    def checks_not_applicable(self) -> int:
        return sum(1 for f in self.findings if f.status == InternalCheckStatus.NOT_APPLICABLE)

    @property
    def checks_inconclusive(self) -> int:
        return sum(1 for f in self.findings if f.status == InternalCheckStatus.INCONCLUSIVE)

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "context": self.context.to_dict(),
            "spans": [s.to_dict() for s in self.spans],
            "failure_signals": [s.to_dict() for s in self.failure_signals],
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        data = require_dict(data, "Trace")
        check_known_keys(data, _ALLOWED_KEYS, "Trace")

        try:
            schema_version = data["schema_version"]
        except KeyError as exc:
            raise TraceValidationError(f"Trace missing required field: {exc}") from exc
        if schema_version != SCHEMA_VERSION:
            raise SchemaVersionMismatchError(
                f"Trace schema_version {schema_version!r} does not match "
                f"supported version {SCHEMA_VERSION!r}"
            )

        try:
            trace_id = data["id"]
        except KeyError as exc:
            raise TraceValidationError(f"Trace missing required field: {exc}") from exc

        context = TraceContext.from_dict(data["context"]) if "context" in data else TraceContext()
        spans = [Span.from_dict(s) for s in data.get("spans", [])]
        failure_signals = [FailureSignal.from_dict(s) for s in data.get("failure_signals", [])]
        findings = [Finding.from_dict(f) for f in data.get("findings", [])]

        # Re-running __post_init__ via normal construction re-validates all
        # cross-field invariants (duplicate ids, claim/signal references)
        # against the just-deserialized data, not just at the Python-object
        # construction sites above.
        return cls(
            id=trace_id,
            schema_version=schema_version,
            context=context,
            spans=spans,
            failure_signals=failure_signals,
            findings=findings,
        )
