"""Phase 3: deterministic Trace checks.

    Evidence -> deterministic analysis -> structured Claim -> Finding

A `TraceCheck` only reports what pattern was (or wasn't) observed in a
Trace's already-captured evidence — never why, never whether the agent
was "correct", and never a causal explanation. V1 checks make zero LLM
calls and are fully deterministic.

See `agentdebug.analysis.check` for the `TraceCheck`/`CheckDefinition`
contract (including the `available_capabilities` design note) and
`agentdebug.analysis.checks` for the three V1 checks.
"""

from __future__ import annotations

from agentdebug.analysis.check import CheckDefinition, TraceCheck

__all__ = ["CheckDefinition", "TraceCheck"]
