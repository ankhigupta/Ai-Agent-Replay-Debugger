"""JSON serialization helpers and schema versioning for AgentDebug core models.

Serialization is intentionally strict:

- Unknown fields raise `TraceValidationError` rather than being silently
  dropped (`check_known_keys`), at every model level, not just `Trace`.
- `schema_version` must match `SCHEMA_VERSION` exactly on deserialization
  (enforced in `Trace.from_dict`) — no coercion, no migration in Phase 1.
- Enum values are validated through `enum_from_value`, which turns a raw
  `ValueError` from an unknown enum string into a `TraceValidationError`.

This module intentionally does NOT import `Trace` (or any other model) at
module level: `Trace` imports from here (`SCHEMA_VERSION`, `check_known_keys`,
`require_dict`), so importing `Trace` here too would be circular. `from_json`
imports it locally, inside the function body, once it's actually needed.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Iterable

from agentdebug.core.exceptions import TraceValidationError

SCHEMA_VERSION = "1.0.0"


def is_json_serializable(value: Any) -> bool:
    """Return True if `value` consists only of JSON-compatible data.

    Allowed: str, int, float, bool, None, and dict/list/tuple containers
    of only these, recursively. Dict keys must be strings. Anything else
    (arbitrary objects, sets, etc.) is rejected.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(is_json_serializable(v) for v in value)
    return False


def require_dict(data: Any, type_name: str) -> dict[str, Any]:
    """Validate that `data` is a JSON object before field-by-field parsing."""
    if not isinstance(data, dict):
        raise TraceValidationError(
            f"{type_name} must deserialize from a JSON object, got {type(data).__name__}"
        )
    return data


def check_known_keys(data: dict[str, Any], allowed: Iterable[str], type_name: str) -> None:
    """Reject any key in `data` that isn't in `allowed`.

    Guards against silent schema drift: a field renamed or removed
    upstream fails loudly here instead of being ignored.
    """
    unknown = set(data) - set(allowed)
    if unknown:
        raise TraceValidationError(f"Unknown field(s) for {type_name}: {sorted(unknown)}")


def enum_from_value(enum_cls: type[Enum], value: Any, field_name: str, type_name: str) -> Enum:
    """Convert a raw serialized value to an enum member, or raise
    `TraceValidationError` (not a bare `ValueError`) if it isn't valid."""
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise TraceValidationError(
            f"Invalid {field_name} for {type_name}: {value!r}"
        ) from exc


def to_json(trace: "Trace") -> str:  # type: ignore[name-defined]
    """Serialize a Trace to a JSON string via its `to_dict()`."""
    return json.dumps(trace.to_dict())


def from_json(data: str) -> "Trace":  # type: ignore[name-defined]
    """Deserialize a JSON string (as produced by `to_json`) into a Trace."""
    from agentdebug.core.trace import Trace

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise TraceValidationError(f"Invalid JSON: {exc}") from exc
    return Trace.from_dict(parsed)
