"""AgentDebug: framework-independent, evidence-backed debugging for AI-agent executions.

    Agent -> Recorder -> Trace + Spans -> deterministic checks
          -> (with a FailureSignal) failure-driven localization -> Findings

This top-level package re-exports AgentDebug's stable, developer-facing
API — everything a normal caller needs for the walkthrough above, plus the
exceptions and base classes needed to handle errors or extend the SDK
(`TraceCheck`/`CheckDefinition` to write a custom check, `LocalizationAnalyzer`
to write a custom analyzer, `EvidenceSink` to write a custom evidence
sink) — so `from agentdebug import Recorder, SpanKind, ...` works without
reaching into submodules. Every name here is re-exported from its real
home (`agentdebug.core`, `agentdebug.recording`, `agentdebug.analysis.check`,
`agentdebug.analysis.checks`, `agentdebug.analysis.localization`,
`agentdebug.evidence`), which remain the source of truth and are still
directly importable. Internal helpers (e.g. deterministic-ordering/
canonicalization utilities, the Phase 1 serialization internals) are
deliberately left out of this top-level surface.

Framework integrations (e.g. `agentdebug.integrations.langgraph`) are
never imported here, so `import agentdebug` never requires a framework —
`agentdebug` has zero required third-party dependencies. LangGraph support
is an optional extra (`pip install agentdebug[langgraph]`); import
`agentdebug.integrations.langgraph` directly to use it.

"""

from __future__ import annotations

from agentdebug.analysis.check import CheckDefinition, TraceCheck
from agentdebug.analysis.checks import (
    ALL_CHECKS,
    RepeatedIdenticalCallsCheck,
    SwallowedToolErrorCheck,
    TraceIncompletenessCheck,
)
from agentdebug.analysis.localization import (
    FirstObservedOccurrenceAnalyzer,
    LocalizationAnalyzer,
    LocalizationTarget,
    ValueType,
)
from agentdebug.core import (
    AgentDebugError,
    Capability,
    Claim,
    ClaimType,
    DiagnosisConfidence,
    ExternalStatus,
    FailureSignal,
    Finding,
    InternalCheckStatus,
    InternalRunStatus,
    SCHEMA_VERSION,
    SchemaVersionMismatchError,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
    TraceContext,
    TraceValidationError,
    from_json,
    is_json_serializable,
    to_json,
)
from agentdebug.evidence import EvidenceSink, EvidenceSinkError, JsonlEvidenceSink
from agentdebug.recording import ActiveSpan, Recorder, RecorderError

__all__ = [
    "ALL_CHECKS",
    "ActiveSpan",
    "AgentDebugError",
    "Capability",
    "CheckDefinition",
    "Claim",
    "ClaimType",
    "DiagnosisConfidence",
    "EvidenceSink",
    "EvidenceSinkError",
    "ExternalStatus",
    "FailureSignal",
    "Finding",
    "FirstObservedOccurrenceAnalyzer",
    "InternalCheckStatus",
    "InternalRunStatus",
    "JsonlEvidenceSink",
    "LocalizationAnalyzer",
    "LocalizationTarget",
    "Recorder",
    "RecorderError",
    "RepeatedIdenticalCallsCheck",
    "SCHEMA_VERSION",
    "SchemaVersionMismatchError",
    "Span",
    "SpanKind",
    "SpanStatus",
    "SwallowedToolErrorCheck",
    "Trace",
    "TraceCheck",
    "TraceContext",
    "TraceIncompletenessCheck",
    "TraceValidationError",
    "ValueType",
    "from_json",
    "is_json_serializable",
    "to_json",
]
