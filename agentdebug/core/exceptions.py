"""Core exception hierarchy for AgentDebug."""

from __future__ import annotations


class AgentDebugError(Exception):
    """Base class for all AgentDebug errors."""


class TraceValidationError(AgentDebugError):
    """Raised when a Trace or one of its component models is malformed.

    Covers both direct construction (e.g. a negative `start_time`) and
    deserialization (e.g. an unknown field, a missing required field, or
    a dangling claim/signal reference).
    """


class SchemaVersionMismatchError(AgentDebugError):
    """Raised when deserialized data's `schema_version` does not exactly
    match the currently supported `SCHEMA_VERSION`.

    Phase 1 does not implement migrations: a mismatch is always fatal,
    never coerced or silently accepted.
    """
