"""Framework-neutral representation of a full agent execution.

An Execution is the top-level record for a single run of an agent, framework
notwithstanding. It aggregates the steps taken and any checkpoints captured
along the way, so that a UI or replay engine can work against one consistent
shape regardless of whether the underlying agent was built with LangGraph,
CrewAI, or something else entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from app.models.checkpoint import Checkpoint
from app.models.step import ExecutionStep


class ExecutionStatus(str, Enum):
    """Lifecycle state of an Execution."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Execution:
    """A single, framework-neutral record of an agent run.

    Attributes:
        execution_id: Unique identifier for this execution.
        agent_name: Human-readable name of the agent that was run.
        framework: Identifier of the originating framework (e.g.
            "langgraph", "crewai"). Purely descriptive metadata — no code
            in this module depends on its value.
        started_at: Timestamp when the execution began.
        ended_at: Timestamp when the execution finished, or None if still
            running.
        status: Current lifecycle status of the execution.
        input: Arbitrary input payload the execution was started with.
        steps: Ordered list of steps captured during the execution.
        checkpoints: List of state checkpoints captured during the
            execution.
    """

    execution_id: str
    agent_name: str
    framework: str
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    input: Any = None
    steps: list[ExecutionStep] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
