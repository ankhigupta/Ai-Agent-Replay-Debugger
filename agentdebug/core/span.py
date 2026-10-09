"""Span: one recorded unit of work within a Trace.

Covers the same kind of ground as the older replay-debugger's
`ExecutionStep`, but is a fresh model, not a rename or subclass of it —
AgentDebug's Trace/Span model is the new foundation, independent of the
old Execution/Step/Checkpoint architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentdebug.core.enums import SpanKind, SpanStatus
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.serialization import (
    check_known_keys,
    enum_from_value,
    is_json_serializable,
    require_dict,
)

_ALLOWED_KEYS = frozenset(
    {
        "id",
        "parent_id",
        "kind",
        "name",
        "status",
        "start_time",
        "end_time",
        "recorder_sequence_id",
        "input",
        "output",
        "metadata",
    }
)


@dataclass(frozen=True, kw_only=True)
class Span:
    """A single recorded unit of work within a Trace.

    `parent_id`, when set, is intentionally NOT validated against other
    spans at construction time — a dangling reference (pointing at a span
    id that doesn't exist, anywhere) is explicitly allowed. Rejecting it
    eagerly would make that defect unrepresentable, and a later Trace
    Incompleteness check needs to be able to detect and report exactly
    this condition.
    """

    id: str
    parent_id: Optional[str] = None
    kind: SpanKind
    name: str
    status: SpanStatus
    start_time: float
    end_time: float
    recorder_sequence_id: int
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise TraceValidationError("Span.id must be a non-empty string")
        if self.parent_id is not None and not isinstance(self.parent_id, str):
            raise TraceValidationError("Span.parent_id must be a string or None")
        if not isinstance(self.name, str) or not self.name:
            raise TraceValidationError("Span.name must be a non-empty string")
        if self.start_time < 0:
            raise TraceValidationError("Span.start_time must be >= 0")
        if self.end_time < self.start_time:
            raise TraceValidationError("Span.end_time must be >= start_time")
        if self.recorder_sequence_id < 0:
            raise TraceValidationError("Span.recorder_sequence_id must be >= 0")
        if self.input is not None and not is_json_serializable(self.input):
            raise TraceValidationError("Span.input must be JSON serializable")
        if self.output is not None and not is_json_serializable(self.output):
            raise TraceValidationError("Span.output must be JSON serializable")
        if not is_json_serializable(self.metadata):
            raise TraceValidationError("Span.metadata values must be JSON serializable")

    @property
    def duration(self) -> float:
        """Wall-clock duration of this span. Always derived, never stored."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "kind": self.kind.value,
            "name": self.name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "recorder_sequence_id": self.recorder_sequence_id,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Span":
        data = require_dict(data, "Span")
        check_known_keys(data, _ALLOWED_KEYS, "Span")
        try:
            return cls(
                id=data["id"],
                parent_id=data.get("parent_id"),
                kind=enum_from_value(SpanKind, data["kind"], "kind", "Span"),
                name=data["name"],
                status=enum_from_value(SpanStatus, data["status"], "status", "Span"),
                start_time=data["start_time"],
                end_time=data["end_time"],
                recorder_sequence_id=data["recorder_sequence_id"],
                input=data.get("input"),
                output=data.get("output"),
                metadata=data.get("metadata", {}),
            )
        except KeyError as exc:
            raise TraceValidationError(f"Span missing required field: {exc}") from exc
