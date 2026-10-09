"""Deterministic span ordering shared by every Phase 3 check.

Every check must examine spans in `(start_time, recorder_sequence_id)`
order. `Trace.spans` list order is NOT guaranteed chronological (callers
could in principle construct a Trace with spans in any order), so checks
must never rely on it directly. Never use UUID ordering, dict/set
iteration order, object identity, or wall-clock "now" for ordering either.
"""

from __future__ import annotations

from typing import Iterable

from agentdebug.core.span import Span


def sort_key(span: Span) -> tuple[float, int]:
    """The canonical deterministic ordering key for a span."""
    return (span.start_time, span.recorder_sequence_id)


def ordered_spans(spans: Iterable[Span]) -> list[Span]:
    """Return `spans` sorted by `sort_key` (ascending: earliest first)."""
    return sorted(spans, key=sort_key)
