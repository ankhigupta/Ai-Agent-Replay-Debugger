"""Claim: a single factual, structured piece of evidence inside a Finding.

Claims hold facts observed directly in the trace (e.g. "span-2's output had
refund_eligible=false"), never narrative causal conclusions like "span-2
caused the failure". Keeping claims strictly factual is what makes a
Finding evidence-backed rather than merely asserted — there is no claim
rendering / narrative layer in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentdebug.core.enums import ClaimType
from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.serialization import (
    check_known_keys,
    enum_from_value,
    is_json_serializable,
    require_dict,
)

_ALLOWED_KEYS = frozenset({"type", "span_ids", "fields"})


@dataclass(frozen=True, kw_only=True)
class Claim:
    """A single structured, factual piece of evidence.

    `span_ids` are referenced by id only; Claim does not check that those
    spans actually exist anywhere — that referential check happens at
    `Trace` level, where the full span set is known.
    """

    type: ClaimType
    span_ids: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for span_id in self.span_ids:
            if not isinstance(span_id, str) or not span_id:
                raise TraceValidationError("Claim.span_ids entries must be non-empty strings")
        if not is_json_serializable(self.fields):
            raise TraceValidationError(
                "Claim.fields must consist only of JSON primitives and nested "
                "dict/list structures of JSON primitives"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "span_ids": list(self.span_ids),
            "fields": self.fields,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Claim":
        data = require_dict(data, "Claim")
        check_known_keys(data, _ALLOWED_KEYS, "Claim")
        try:
            return cls(
                type=enum_from_value(ClaimType, data["type"], "type", "Claim"),
                span_ids=list(data.get("span_ids", [])),
                fields=data.get("fields", {}),
            )
        except KeyError as exc:
            raise TraceValidationError(f"Claim missing required field: {exc}") from exc
