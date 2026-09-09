"""Framework-neutral representation of a single step within an agent execution.

A "step" is any discrete unit of work an agent performs while producing a
result: calling an LLM, invoking a tool, retrieving documents, updating
internal state, delegating to a sub-agent, etc. This model intentionally
avoids any vocabulary tied to a specific orchestration framework (e.g. no
"node", "task", or "chain" terms borrowed from a particular library) so that
adapters for different frameworks can all map onto the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class StepType(str, Enum):
    """Generic categories of work a step can represent.

    Kept deliberately framework-agnostic: no LangGraph/CrewAI-specific
    step kinds. Anything that doesn't fit a specific category should use
    OTHER rather than growing this enum per-framework.
    """

    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    STATE_UPDATE = "state_update"
    AGENT = "agent"
    OTHER = "other"


@dataclass
class ExecutionStep:
    """A single recorded unit of work within an Execution.

    Attributes:
        step_id: Unique identifier for this step within its execution.
        name: Human-readable label (e.g. tool name, node name).
        type: Generic category of work this step performs.
        input: Arbitrary input payload captured for this step.
        output: Arbitrary output payload captured for this step, if completed.
        started_at: Timestamp when the step began executing.
        ended_at: Timestamp when the step finished, or None if still running.
        metadata: Free-form, adapter-specific extra data that doesn't belong
            in the core fields (e.g. token usage, model name, latency).
    """

    step_id: str
    name: str
    type: StepType
    input: Any = None
    output: Any = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
