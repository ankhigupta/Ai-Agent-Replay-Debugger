"""Recorder: captures structured execution evidence into a Trace.

The Recorder answers only "what happened?" — it builds Trace/Span records
from a running agent. It never decides whether the agent was correct,
never diagnoses failures, and never produces Claims or Findings; that is
later analysis, built on top of what the Recorder captures, never inside
it.

Usage:

    recorder = Recorder()
    trace = recorder.start_trace()

    with recorder.span("agent", kind=SpanKind.CUSTOM):
        with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}) as span:
            result = get_order("ORD-42")
            span.set_output(result)

    recorder.end_trace()

`parent_id` is never passed explicitly: nesting `recorder.span()` calls
infers it from whichever span is currently active (see
`agentdebug.recording.context`).
"""

from __future__ import annotations

import contextlib
import itertools
import threading
import time
import uuid
from typing import Any, Iterator, Optional

from agentdebug.core.context import TraceContext
from agentdebug.core.enums import SpanKind, SpanStatus
from agentdebug.core.failure_signal import FailureSignal
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace
from agentdebug.recording.context import (
    bind_active_span_id,
    bind_active_trace,
    get_current_span_id,
    get_current_trace,
    reset_active_span_id,
    reset_active_trace,
)


class RecorderError(Exception):
    """Raised for Recorder usage errors, e.g. recording a span with no
    active trace. Distinct from `TraceValidationError`, which is about
    malformed Trace/Span *data*, not Recorder call-sequencing mistakes.
    """


class ActiveSpan:
    """Mutable, in-progress recording state for one span.

    Exists only between `Recorder.span()`'s `__enter__` and `__exit__`.
    Once the span finishes, its data is used to construct an immutable
    `agentdebug.core.span.Span` and this object is discarded — it is
    never itself added to a Trace, and never touched after the `with`
    block exits.
    """

    def __init__(
        self,
        *,
        id: str,
        parent_id: Optional[str],
        kind: SpanKind,
        name: str,
        start_time: float,
        recorder_sequence_id: int,
        input: Any,
        metadata: dict[str, Any],
    ) -> None:
        self.id = id
        self.parent_id = parent_id
        self.kind = kind
        self.name = name
        self.status = SpanStatus.OK
        self.start_time = start_time
        self.recorder_sequence_id = recorder_sequence_id
        self.input = input
        self.output: Any = None
        self.metadata = metadata

    def set_output(self, output: Any) -> None:
        """Record this span's output. Only used if the span completes
        successfully; ignored if the `with` block raises."""
        self.output = output


class Recorder:
    """Captures Trace/Span evidence for one or more agent executions.

    Concurrency is split three ways:

    - Active trace/span *context* (which Trace, which parent span) lives
      in `contextvars.ContextVar`s (`agentdebug.recording.context`), so
      each asyncio Task and each OS thread sees its own isolated view of
      "what's currently active" — no cross-task or cross-thread
      contamination.
    - `recorder_sequence_id` allocation is global to one Recorder instance
      (order needs to be meaningful across everything that Recorder has
      ever produced) and is protected by a single `threading.Lock`, so
      concurrent threads never receive duplicate or out-of-order sequence
      ids.
    - `Trace.add_span`/`add_finding`/`add_failure_signal` already validate
      before mutating (a Phase 1 invariant), so a span is only appended
      after its full lifecycle — including exception handling — has
      completed. No additional locking is added around `Trace` mutation
      itself: this is adequate for normal asyncio/threaded recording, not
      a concurrent storage engine. If multiple threads called
      `Trace.add_span` on the very same `Trace` at the same instant,
      CPython's GIL keeps the underlying `list.append` from corrupting,
      but no attempt is made to serialize interleaved multi-thread writes
      beyond that.
    """

    def __init__(self) -> None:
        self._sequence_lock = threading.Lock()
        self._sequence_counter = itertools.count(0)

    def _next_sequence_id(self) -> int:
        with self._sequence_lock:
            return next(self._sequence_counter)

    # -- trace lifecycle --------------------------------------------------

    def start_trace(self, *, id: Optional[str] = None, context: Optional[TraceContext] = None) -> Trace:
        """Start a new Trace and make it the active trace for this
        task/thread's context. `id` defaults to a new UUID4; `context`
        defaults to `TraceContext()` (Phase 1 defaults)."""
        trace = Trace(
            id=id if id is not None else str(uuid.uuid4()),
            context=context if context is not None else TraceContext(),
        )
        bind_active_trace(trace)
        return trace

    def end_trace(self) -> Trace:
        """End the active trace's recording session and return it.

        Clears the active-trace context for this task/thread; further
        `span()` calls here will raise `RecorderError` until a new trace
        is started. Phase 1's `Trace` has no "ended_at"/lifecycle-status
        field of its own, so "ending" means ending the recording session,
        not mutating the returned `Trace` object.
        """
        trace = self._require_active_trace()
        bind_active_trace(None)
        return trace

    def current_trace(self) -> Optional[Trace]:
        """Return the Trace active in this task/thread's context, or None."""
        return get_current_trace()

    def add_failure_signal(self, signal: FailureSignal) -> None:
        """Attach an externally-created FailureSignal to the active trace.

        Only attaches what's given — the Recorder never infers failures
        on its own.
        """
        trace = self._require_active_trace()
        trace.add_failure_signal(signal)

    def _require_active_trace(self) -> Trace:
        trace = get_current_trace()
        if trace is None:
            raise RecorderError("No active trace in this task/thread. Call start_trace() first.")
        return trace

    # -- span lifecycle -----------------------------------------------------

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        kind: SpanKind,
        input: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Iterator[ActiveSpan]:
        """Record one Span, with `parent_id` inferred from whichever span
        is currently active (if any), appended to the currently active
        Trace on exit.

        On a normal exit: status is `SpanStatus.OK`, and `span.output`
        (set via `span.set_output(...)`, if called) is recorded.

        On an exception: status is `SpanStatus.ERROR`, a JSON-serializable
        `{"exception": {"type": ..., "message": ...}}` entry is merged
        into the span's metadata, the span is still added to the Trace,
        and the original exception is re-raised unchanged — the Recorder
        never swallows exceptions.
        """
        trace = self._require_active_trace()
        parent_id = get_current_span_id()
        active = ActiveSpan(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            kind=kind,
            name=name,
            start_time=time.monotonic(),
            recorder_sequence_id=self._next_sequence_id(),
            input=input,
            metadata=dict(metadata) if metadata else {},
        )
        token = bind_active_span_id(active.id)
        try:
            try:
                yield active
            except Exception as exc:
                active.status = SpanStatus.ERROR
                active.metadata["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                trace.add_span(self._finalize(active, time.monotonic()))
                raise
            else:
                trace.add_span(self._finalize(active, time.monotonic()))
        finally:
            reset_active_span_id(token)

    @staticmethod
    def _finalize(active: ActiveSpan, end_time: float) -> Span:
        return Span(
            id=active.id,
            parent_id=active.parent_id,
            kind=active.kind,
            name=active.name,
            status=active.status,
            start_time=active.start_time,
            end_time=end_time,
            recorder_sequence_id=active.recorder_sequence_id,
            input=active.input,
            output=active.output,
            metadata=active.metadata,
        )
