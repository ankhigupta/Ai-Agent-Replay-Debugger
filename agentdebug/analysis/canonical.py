"""Deterministic canonicalization helpers for Phase 3 checks.

Neither function uses Python's built-in `hash()` (not stable across
processes) or depends on dict/set iteration order beyond what
`json.dumps(..., sort_keys=True)` normalizes. List order inside
`span.input` is preserved as-is and treated as semantically meaningful —
canonicalization never reorders lists, only normalizes dict key order.
"""

from __future__ import annotations

import json
from typing import Any

from agentdebug.core.span import Span


def tool_operation_identity(span: Span) -> str:
    """Deterministic identity for "is this the same tool operation".

    V1 definition: `span.name`, exactly. This is a deliberate, documented
    simplification — no semantic matching, no fuzzy comparison, no
    inspection of tool input. Two spans with the same name are treated as
    the same operation for retry/fallback matching even if their inputs
    differ; two spans with different names are never treated as the same
    operation even if they are semantically equivalent calls to the same
    underlying tool under a different label. `span.name` is a required,
    non-empty field on every valid `Span`, so this identity is always
    available once a span exists at all.
    """
    return span.name


def canonical_call_identity(span: Span) -> str:
    """Deterministic identity for "is this call identical to that one".

    Combines `span.name` and `span.input` into a stable JSON string via
    `json.dumps(..., sort_keys=True, separators=(",", ":"))`. Dict key
    order is normalized; list order is preserved. Relies on `span.input`
    already being JSON-serializable, which Phase 1's `Span.__post_init__`
    guarantees for any constructed `Span`.
    """
    canonical_value: dict[str, Any] = {"name": span.name, "input": span.input}
    return json.dumps(canonical_value, sort_keys=True, separators=(",", ":"))
