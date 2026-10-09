"""Closed enums for the AgentDebug core data model.

All enums are `str, Enum` so `.value` is also the serialized JSON form —
serialized output must always show plain strings like "TOOL", never a
Python enum repr like "SpanKind.TOOL". Constructing a member from an
unknown value (e.g. `SpanKind("BOGUS")`) raises `ValueError` via Python's
own `Enum` machinery; deserialization code wraps that into
`TraceValidationError` (see `serialization.enum_from_value`).
"""

from __future__ import annotations

from enum import Enum


class SpanKind(str, Enum):
    """What kind of work a Span represents."""

    LLM = "LLM"
    TOOL = "TOOL"
    RETRIEVAL = "RETRIEVAL"
    STATE = "STATE"
    ERROR = "ERROR"
    CUSTOM = "CUSTOM"


class SpanStatus(str, Enum):
    """Outcome of a single Span."""

    OK = "OK"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class ExternalStatus(str, Enum):
    """A verdict sourced from outside the trace (e.g. a failing test),
    not something AgentDebug itself concluded.

    Used by `FailureSignal.verdict` and the computed `Trace.external_status`.
    """

    FAILED = "FAILED"
    PASSED = "PASSED"
    UNKNOWN = "UNKNOWN"


class InternalCheckStatus(str, Enum):
    """The result of running a single deterministic check, as recorded on
    one `Finding`. See `Finding` for the status-conditioned shape each
    value implies."""

    FLAGGED = "FLAGGED"
    NO_FINDING = "NO_FINDING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class InternalRunStatus(str, Enum):
    """The aggregate result across all of a Trace's Findings.

    Deliberately a separate enum from `InternalCheckStatus`: "how did this
    one check go" and "how did the whole run go" are different questions
    with different answer shapes (there is no per-check "clean", only a
    run-level one), so they are not merged into a single enum.
    """

    FLAGGED = "FLAGGED"
    INCONCLUSIVE = "INCONCLUSIVE"
    CLEAN_UNDER_CHECKS = "CLEAN_UNDER_CHECKS"


class ClaimType(str, Enum):
    """What kind of factual evidence a `Claim` represents."""

    TOOL_ERROR_SWALLOWED = "TOOL_ERROR_SWALLOWED"
    REPEATED_IDENTICAL_CALLS = "REPEATED_IDENTICAL_CALLS"
    TRACE_INCOMPLETE = "TRACE_INCOMPLETE"
    VALUE_FIRST_OBSERVED = "VALUE_FIRST_OBSERVED"
    VALUE_NOT_OBSERVED = "VALUE_NOT_OBSERVED"


class Capability(str, Enum):
    """A category of evidence a check may depend on but which a given
    trace might not provide. Reported via `Finding.capability_gaps`
    instead of a check silently skipping or guessing."""

    TOOL_STATUS = "TOOL_STATUS"
    TOOL_CALLS = "TOOL_CALLS"
    SPAN_IO = "SPAN_IO"
    PRE_RUN_CONTEXT = "PRE_RUN_CONTEXT"
    RETRIEVAL_DOCS = "RETRIEVAL_DOCS"


class DiagnosisConfidence(str, Enum):
    """How confidently a check's evidence points at a single cause.

    Phase 1 only defines the type; no analyzer sets or interprets it yet.
    """

    UNIQUE = "UNIQUE"
    AMBIGUOUS = "AMBIGUOUS"
    NONE = "NONE"
