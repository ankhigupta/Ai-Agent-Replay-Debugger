"""Trace Incompleteness check.

Looks for an observable structural pattern only: a Span whose `parent_id`
references a span id that does not exist anywhere in the Trace. Phase 1
deliberately allows this ("dangling parent_id") precisely so this check
can detect and report it. It does NOT say the trace is invalid, that the
missing span caused anything, or that the execution is unreliable — only
that this specific structural reference problem was (or wasn't) observed.
"""

from __future__ import annotations

from typing import Optional

from agentdebug.analysis.check import CheckDefinition, TraceCheck, missing_capabilities
from agentdebug.analysis.ordering import ordered_spans
from agentdebug.core.claim import Claim
from agentdebug.core.enums import Capability, ClaimType, InternalCheckStatus
from agentdebug.core.finding import Finding
from agentdebug.core.trace import Trace


class TraceIncompletenessCheck(TraceCheck):
    """FLAGs any Span whose `parent_id` does not match an existing Span.id
    in the same Trace."""

    definition = CheckDefinition(
        id="trace_incompleteness",
        basis=(
            "Detects structural span-reference problems such as spans "
            "referencing missing parents."
        ),
        required_capabilities=(Capability.SPAN_IO,),
        known_false_positive_modes=(
            "A recorder or integration may intentionally omit spans that "
            "were not observable.",
        ),
    )

    def run(
        self, trace: Trace, *, available_capabilities: Optional[frozenset[Capability]] = None
    ) -> Finding:
        missing = missing_capabilities(self.definition, available_capabilities)
        if missing:
            return Finding(
                check_id=self.definition.id,
                status=InternalCheckStatus.NOT_APPLICABLE,
                capability_gaps=missing,
            )

        known_span_ids = {s.id for s in trace.spans}
        dangling = [
            s
            for s in ordered_spans(trace.spans)
            if s.parent_id is not None and s.parent_id not in known_span_ids
        ]

        claims = [
            Claim(
                type=ClaimType.TRACE_INCOMPLETE,
                span_ids=[s.id],
                fields={"span_id": s.id, "missing_parent_id": s.parent_id},
            )
            for s in dangling
        ]

        if claims:
            return Finding(check_id=self.definition.id, status=InternalCheckStatus.FLAGGED, claims=claims)
        return Finding(check_id=self.definition.id, status=InternalCheckStatus.NO_FINDING)
