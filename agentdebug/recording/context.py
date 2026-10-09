"""contextvars-based active trace/span tracking for the Recorder.

Two independent `ContextVar`s carry the "where am I right now" state
needed to infer `parent_id` automatically: which `Trace` spans are
currently being recorded into, and which `Span` (if any) is the innermost
currently-active one.

Why `contextvars` and not a single global or thread-local:

- An asyncio Task copies the current `Context` at the moment it is
  created (`asyncio.create_task`/`ensure_future`), and that copy becomes
  the Task's own, independent `Context` for its lifetime. So a value set
  inside one Task is invisible to a sibling Task started earlier, and
  changes inside a Task never leak back out to its caller — exactly the
  isolation needed so concurrent tasks never see each other's active span.
- A new OS thread also starts with its own default `Context` (values set
  via `ContextVar.set` in one thread are not visible from another unless
  explicitly copied), so the same mechanism incidentally isolates
  concurrent threads too, with no extra thread-local machinery needed.

This module only exposes the raw get/bind/reset primitives; `Recorder`
(in `agentdebug.recording.recorder`) is what sequences them into span/trace
lifecycles.
"""

from __future__ import annotations

import contextvars
from typing import Optional

from agentdebug.core.trace import Trace

_active_trace: "contextvars.ContextVar[Optional[Trace]]" = contextvars.ContextVar(
    "agentdebug_active_trace", default=None
)
_active_span_id: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "agentdebug_active_span_id", default=None
)


def get_current_trace() -> Optional[Trace]:
    """Return the Trace active in this task/thread's context, if any."""
    return _active_trace.get()


def bind_active_trace(trace: Optional[Trace]) -> "contextvars.Token[Optional[Trace]]":
    """Set the active trace (or clear it, with None).

    Returns a Token that `reset_active_trace` can use to restore whatever
    was active before this call.
    """
    return _active_trace.set(trace)


def reset_active_trace(token: "contextvars.Token[Optional[Trace]]") -> None:
    """Restore the active trace to what it was before the matching `bind_active_trace`."""
    _active_trace.reset(token)


def get_current_span_id() -> Optional[str]:
    """Return the id of the innermost currently-active Span, if any."""
    return _active_span_id.get()


def bind_active_span_id(span_id: Optional[str]) -> "contextvars.Token[Optional[str]]":
    """Set the active span id (or clear it, with None).

    Returns a Token that `reset_active_span_id` can use to restore
    whatever was active before this call.
    """
    return _active_span_id.set(span_id)


def reset_active_span_id(token: "contextvars.Token[Optional[str]]") -> None:
    """Restore the active span id to what it was before the matching `bind_active_span_id`."""
    _active_span_id.reset(token)


def clear() -> None:
    """Force both active-trace and active-span-id to None in the current
    context. Not used by normal Recorder operation (which always restores
    the prior value via tokens); provided for test isolation, so tests
    don't leak active state to each other within the same thread/context.
    """
    _active_trace.set(None)
    _active_span_id.set(None)
