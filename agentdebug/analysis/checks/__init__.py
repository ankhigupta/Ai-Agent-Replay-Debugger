"""The three V1 Phase 3 deterministic checks.

`ALL_CHECKS` is a plain tuple of instances for tests/demos that want to
run "the checks that exist" without hand-maintaining an import list. It is
not a registry: no lookup-by-id, no discovery mechanism, no execution
orchestration — "run all checks and aggregate findings" is explicitly out
of scope for Phase 3.
"""

from __future__ import annotations

from agentdebug.analysis.check import TraceCheck
from agentdebug.analysis.checks.repeated_identical_calls import RepeatedIdenticalCallsCheck
from agentdebug.analysis.checks.swallowed_tool_error import SwallowedToolErrorCheck
from agentdebug.analysis.checks.trace_incompleteness import TraceIncompletenessCheck

ALL_CHECKS: tuple[TraceCheck, ...] = (
    SwallowedToolErrorCheck(),
    RepeatedIdenticalCallsCheck(),
    TraceIncompletenessCheck(),
)

__all__ = [
    "ALL_CHECKS",
    "RepeatedIdenticalCallsCheck",
    "SwallowedToolErrorCheck",
    "TraceIncompletenessCheck",
]
