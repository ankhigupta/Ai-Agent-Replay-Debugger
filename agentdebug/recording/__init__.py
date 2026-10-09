"""Phase 2: the Recorder runtime that captures Trace/Span evidence.

Answers "what happened?" only — building Trace/Span records from a
running agent. Never decides correctness, never diagnoses failures, never
produces Claims or Findings; that is later analysis, built on top of this,
never inside it.
"""

from __future__ import annotations

from agentdebug.recording.recorder import ActiveSpan, Recorder, RecorderError

__all__ = ["ActiveSpan", "Recorder", "RecorderError"]
