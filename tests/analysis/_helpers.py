"""Shared span/trace construction helpers for Phase 3 and 4 analysis tests.

Not a test module itself (no `Test*` classes, filename doesn't match
`test*.py`), so `unittest discover` never collects it directly.
"""

from __future__ import annotations

from typing import Any, Optional

from agentdebug.core.context import TraceContext
from agentdebug.core.enums import SpanKind, SpanStatus
from agentdebug.core.finding import Finding
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace


def make_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.TOOL,
    status: SpanStatus = SpanStatus.OK,
    parent_id: Optional[str] = None,
    start_time: float,
    recorder_sequence_id: int,
    input: Any = None,
    output: Any = None,
    id: Optional[str] = None,
) -> Span:
    return Span(
        id=id if id is not None else f"span-{recorder_sequence_id}",
        parent_id=parent_id,
        kind=kind,
        name=name,
        status=status,
        start_time=start_time,
        end_time=start_time + 1.0,
        recorder_sequence_id=recorder_sequence_id,
        input=input,
        output=output,
        metadata={},
    )


def make_trace(
    *, id: str = "t1", spans: Optional[list[Span]] = None, context: Optional[TraceContext] = None
) -> Trace:
    return Trace(
        id=id, spans=list(spans) if spans else [], context=context if context is not None else TraceContext()
    )


def finding_content(finding: Finding) -> dict[str, Any]:
    """`finding.to_dict()` with the identity field (`id`) removed, so two
    Findings can be compared on analytical content only."""
    data = finding.to_dict()
    data.pop("id")
    return data
