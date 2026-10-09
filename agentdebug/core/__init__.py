"""Core data model for AgentDebug: Trace, Span, Claim, Finding, and related types.

Phase 1 only: data structures, validation, and serialization. No recording
infrastructure, deterministic checks, localization, or framework
integrations yet — those are later phases, built on top of this module,
not inside it.
"""

from __future__ import annotations

from agentdebug.core.claim import Claim
from agentdebug.core.context import TraceContext
from agentdebug.core.enums import (
    Capability,
    ClaimType,
    DiagnosisConfidence,
    ExternalStatus,
    InternalCheckStatus,
    InternalRunStatus,
    SpanKind,
    SpanStatus,
)
from agentdebug.core.exceptions import (
    AgentDebugError,
    SchemaVersionMismatchError,
    TraceValidationError,
)
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.finding import Finding
from agentdebug.core.serialization import SCHEMA_VERSION, from_json, is_json_serializable, to_json
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace

__all__ = [
    "AgentDebugError",
    "Capability",
    "Claim",
    "ClaimType",
    "DiagnosisConfidence",
    "ExternalStatus",
    "FailureSignal",
    "Finding",
    "InternalCheckStatus",
    "InternalRunStatus",
    "SCHEMA_VERSION",
    "SchemaVersionMismatchError",
    "Span",
    "SpanKind",
    "SpanStatus",
    "Trace",
    "TraceContext",
    "TraceValidationError",
    "from_json",
    "is_json_serializable",
    "to_json",
]
