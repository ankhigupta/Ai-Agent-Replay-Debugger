"""FailureSignal: an externally-sourced verdict about whether a run failed.

Distinct from Findings: a FailureSignal is evidence fed *into* AgentDebug
from outside (e.g. "pytest says this assertion failed"), not a conclusion
AgentDebug reached on its own.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from agentdebug.core.enums import ExternalStatus
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.serialization import (
    check_known_keys,
    enum_from_value,
    is_json_serializable,
    require_dict,
)

_ALLOWED_KEYS = frozenset({"id", "source", "verdict", "expected", "actual", "detail"})


@dataclass(frozen=True, kw_only=True)
class FailureSignal:
    """An externally-sourced pass/fail verdict attached to a Trace."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    verdict: ExternalStatus
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise TraceValidationError("FailureSignal.source must be a non-empty string")
        if self.expected is not None and not is_json_serializable(self.expected):
            raise TraceValidationError("FailureSignal.expected must be JSON serializable")
        if self.actual is not None and not is_json_serializable(self.actual):
            raise TraceValidationError("FailureSignal.actual must be JSON serializable")
        if self.detail is not None and not isinstance(self.detail, str):
            raise TraceValidationError("FailureSignal.detail must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "verdict": self.verdict.value,
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureSignal":
        data = require_dict(data, "FailureSignal")
        check_known_keys(data, _ALLOWED_KEYS, "FailureSignal")
        try:
            kwargs: dict[str, Any] = {
                "source": data["source"],
                "verdict": enum_from_value(ExternalStatus, data["verdict"], "verdict", "FailureSignal"),
                "expected": data.get("expected"),
                "actual": data.get("actual"),
                "detail": data.get("detail"),
            }
        except KeyError as exc:
            raise TraceValidationError(f"FailureSignal missing required field: {exc}") from exc
        if "id" in data:
            kwargs["id"] = data["id"]
        return cls(**kwargs)
