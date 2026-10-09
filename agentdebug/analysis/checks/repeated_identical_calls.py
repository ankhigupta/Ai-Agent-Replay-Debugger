"""Repeated Identical Calls check.

Looks for an observable pattern only: three or more consecutive TOOL spans
with identical canonicalized operation and input (see
`agentdebug.analysis.canonical.canonical_call_identity`). "Consecutive"
means adjacent among TOOL-kind spans in deterministic
`(start_time, recorder_sequence_id)` order — other span kinds interleaved
between them do not break the run. It does NOT claim an infinite loop, a
bug, wasted tokens, or incorrectness — only that the repeated-call pattern
was (or wasn't) observed.
"""

from __future__ import annotations

from typing import Optional

from agentdebug.analysis.canonical import canonical_call_identity
from agentdebug.analysis.check import CheckDefinition, TraceCheck, missing_capabilities
from agentdebug.analysis.ordering import ordered_spans
from agentdebug.core.claim import Claim
from agentdebug.core.enums import Capability, ClaimType, InternalCheckStatus, SpanKind
from agentdebug.core.finding import Finding
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace

_MIN_RUN_LENGTH = 3


class RepeatedIdenticalCallsCheck(TraceCheck):
    """FLAGs a maximal run of 3+ consecutive TOOL spans sharing the same
    canonicalized (name, input) identity."""

    definition = CheckDefinition(
        id="repeated_identical_calls",
        basis=(
            "Detects three or more consecutive TOOL spans with identical "
            "canonicalized operation and input."
        ),
        required_capabilities=(Capability.TOOL_CALLS,),
        known_false_positive_modes=(
            "Repeated calls may be intentional or required by the application.",
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

        tool_spans = [s for s in ordered_spans(trace.spans) if s.kind == SpanKind.TOOL]

        claims: list[Claim] = []
        run: list[Span] = []
        run_identity: Optional[str] = None

        def flush() -> None:
            if len(run) >= _MIN_RUN_LENGTH:
                claims.append(
                    Claim(
                        type=ClaimType.REPEATED_IDENTICAL_CALLS,
                        span_ids=[s.id for s in run],
                        fields={
                            "operation": run[0].name,
                            "input": run[0].input,
                            "occurrence_count": len(run),
                        },
                    )
                )

        for span in tool_spans:
            identity = canonical_call_identity(span)
            if identity == run_identity:
                run.append(span)
            else:
                flush()
                run = [span]
                run_identity = identity
        flush()

        if claims:
            return Finding(check_id=self.definition.id, status=InternalCheckStatus.FLAGGED, claims=claims)
        return Finding(check_id=self.definition.id, status=InternalCheckStatus.NO_FINDING)
