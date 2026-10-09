"""Swallowed Tool Error check.

Looks for an observable pattern only: a TOOL span whose status is ERROR,
for which no later sibling TOOL span with the same operation identity is
observed. It does NOT claim the agent noticed the error, ignored it,
caused it, should have retried, or that the run was incorrect — only that
this specific structural pattern was (or wasn't) present. See
`known_false_positive_modes` on the check's `CheckDefinition` for general,
non-Trace-specific limitations of this pattern.
"""

from __future__ import annotations

from typing import Optional

from agentdebug.analysis.canonical import tool_operation_identity
from agentdebug.analysis.check import CheckDefinition, TraceCheck, missing_capabilities
from agentdebug.analysis.ordering import ordered_spans, sort_key
from agentdebug.core.claim import Claim
from agentdebug.core.enums import Capability, ClaimType, InternalCheckStatus, SpanKind, SpanStatus
from agentdebug.core.finding import Finding
from agentdebug.core.trace import Trace


class SwallowedToolErrorCheck(TraceCheck):
    """FLAGs a TOOL span with ERROR status that has no later matching
    sibling TOOL span (same `parent_id`, same `tool_operation_identity`)."""

    definition = CheckDefinition(
        id="swallowed_tool_error",
        basis=(
            "Detects a tool span with ERROR status for which no subsequent "
            "matching retry or fallback sibling is observed."
        ),
        required_capabilities=(Capability.TOOL_STATUS, Capability.TOOL_CALLS),
        known_false_positive_modes=(
            "A valid recovery may occur through an operation not identifiable "
            "as the same tool operation.",
            "The agent may intentionally choose not to retry after an error.",
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
        failed_spans = [s for s in tool_spans if s.status == SpanStatus.ERROR]

        claims: list[Claim] = []
        for failed in failed_spans:
            recovered = any(
                candidate.parent_id == failed.parent_id
                and sort_key(candidate) > sort_key(failed)
                and tool_operation_identity(candidate) == tool_operation_identity(failed)
                for candidate in tool_spans
            )
            if not recovered:
                claims.append(
                    Claim(
                        type=ClaimType.TOOL_ERROR_SWALLOWED,
                        span_ids=[failed.id],
                        fields={
                            "tool_span_id": failed.id,
                            "operation": tool_operation_identity(failed),
                            "error_status": failed.status.value,
                            "retry_or_fallback_observed": False,
                        },
                    )
                )

        if claims:
            return Finding(check_id=self.definition.id, status=InternalCheckStatus.FLAGGED, claims=claims)
        return Finding(check_id=self.definition.id, status=InternalCheckStatus.NO_FINDING)
