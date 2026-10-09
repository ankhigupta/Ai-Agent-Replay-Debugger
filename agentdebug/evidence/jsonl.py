"""JsonlEvidenceSink: append-only JSONL persistence for AgentDebug Traces.

    Trace -> agentdebug.core.serialization.to_json() -> one JSONL line

One Trace is one JSONL line, written via the EXISTING Phase 1 serializer
(`to_json`, i.e. `json.dumps(trace.to_dict())`) — this module does not
reconstruct Trace fields or invent a parallel serialization format, and
does not add a separate "jsonl schema version"; the line IS a Trace
record, carrying Trace's own `schema_version`. A Trace's
`spans`/`failure_signals`/`findings` are already part of `to_dict()`, so
they are written automatically — there is no separate spans/findings
JSONL stream, by design (one Trace is one unit of evidence).

Concurrency: a single `JsonlEvidenceSink` instance guards its writes with
a `threading.Lock`, so sharing one instance across threads *within one
process* is safe. This is NOT a multi-process logging system: there is no
cross-process file locking, and nothing here claims atomicity if two
separate processes append to the same path concurrently. That is an
explicit, documented limitation for V1, not an oversight.

Durability: each write is `flush()`-ed, so a reader opening the file
immediately after `write_trace()` returns sees the new line without
needing `close()` first. This is NOT an `fsync()`-level guarantee — data
can still be lost on an OS-level crash before the page cache reaches
disk. That stronger guarantee is out of scope for V1.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Union

from agentdebug.core.serialization import to_json
from agentdebug.core.trace import Trace
from agentdebug.evidence.sink import EvidenceSink, EvidenceSinkError


class JsonlEvidenceSink(EvidenceSink):
    """Appends one JSON line per `write_trace()` call to a file.

    The destination is opened in append mode (`"a"`), so an existing file
    is never truncated — traces written by a previous run remain, and
    `JsonlEvidenceSink(path)` can always be reopened to keep adding to the
    same evidence file.

    `create_parents` (default `False`): the parent directory is NOT
    created automatically — this matches ordinary `open()` behavior, so a
    missing parent directory raises `FileNotFoundError` just like it would
    for any other file write, with no silent directory creation. Pass
    `create_parents=True` to opt into `Path.mkdir(parents=True,
    exist_ok=True)` first.
    """

    def __init__(self, path: Union[str, Path], *, create_parents: bool = False) -> None:
        self._path = Path(path)
        if create_parents:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def write_trace(self, trace: Trace) -> None:
        """Write `trace` as one complete JSON line, then flush.

        The full line is built (via the existing `to_json`) BEFORE the
        lock is acquired or the file is touched, so a serialization
        failure never reaches the file at all — no partial or corrupt
        line is ever possible.
        """
        record = to_json(trace)
        with self._lock:
            if self._closed:
                raise EvidenceSinkError("Cannot write_trace: sink is already closed")
            self._file.write(record + "\n")
            self._file.flush()

    def close(self) -> None:
        """Close the underlying file. Idempotent: calling this again after
        the sink is already closed has no further effect."""
        with self._lock:
            if self._closed:
                return
            self._file.close()
            self._closed = True
