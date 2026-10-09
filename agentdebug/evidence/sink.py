"""EvidenceSink: the minimal write contract for persisting Trace evidence.

    Trace -> EvidenceSink -> durable, inspectable evidence artifact

Deliberately small: a sink's only job is "take a Trace, persist it
somewhere durable". It is not a repository layer, not a database
abstraction, not a storage manager, and not an event bus — those are
explicitly out of scope. See `agentdebug.core.serialization` for the
actual Trace <-> JSON serialization every sink implementation builds on;
this module does not duplicate it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentdebug.core.exceptions import AgentDebugError
from agentdebug.core.trace import Trace


class EvidenceSinkError(AgentDebugError):
    """Raised for sink usage errors (e.g. writing after `close()`).

    Distinct from `TraceValidationError` (malformed Trace data) and from
    whatever the underlying storage raises for its own failures (e.g. a
    real `OSError` opening a bad path, or a `RuntimeError` from a broken
    serialization) — those propagate completely unchanged, never wrapped
    or swallowed.
    """


class EvidenceSink(ABC):
    """Minimal contract for persisting Trace evidence durably.

    Implementations decide the storage format and location; callers only
    ever need `write_trace` and `close` — and, where natural, the context
    manager protocol, which here just calls `close()` on exit.
    """

    @abstractmethod
    def write_trace(self, trace: Trace) -> None:
        """Persist one complete Trace record.

        Raises `EvidenceSinkError` if called after `close()`. If building
        the record itself fails (e.g. the Trace can't be serialized),
        that failure propagates to the caller unchanged — nothing here
        ever writes a partial or fabricated record.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources. Safe to call more than once."""
        raise NotImplementedError

    def __enter__(self) -> "EvidenceSink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
