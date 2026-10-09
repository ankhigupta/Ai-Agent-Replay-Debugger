"""Phase 6: durable, inspectable evidence export for AgentDebug Traces.

    Trace -> EvidenceSink -> JSONL file

Not a database, not an observability platform — a sink's only job is
"persist this Trace as a durable artifact for later evaluation and
benchmarking". See `sink.py` for the minimal `EvidenceSink` contract and
`jsonl.py` for the only V1 implementation, `JsonlEvidenceSink`.
"""

from __future__ import annotations

from agentdebug.evidence.jsonl import JsonlEvidenceSink
from agentdebug.evidence.sink import EvidenceSink, EvidenceSinkError

__all__ = ["EvidenceSink", "EvidenceSinkError", "JsonlEvidenceSink"]
