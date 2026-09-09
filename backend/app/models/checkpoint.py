"""Framework-neutral representation of a captured state snapshot.

A checkpoint is a point-in-time snapshot of an execution's internal state,
associated with the step that produced it. Not every framework exposes
checkpoints (some only expose step input/output), so this model is kept
separate and optional rather than folded into ExecutionStep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Checkpoint:
    """A snapshot of execution state captured at a particular point.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        step_id: Identifier of the ExecutionStep this checkpoint was
            captured at or after, if the originating framework ties
            checkpoints to individual steps. Optional because not every
            framework's checkpoint granularity aligns 1:1 with a single
            step (e.g. interval-based or multi-step checkpoints).
        state: Arbitrary snapshot of the agent/execution state at this
            point in time. Shape is adapter/framework-specific.
        metadata: Free-form, adapter-specific extra data (e.g. checkpoint
            source, storage backend, versioning info).
    """

    checkpoint_id: str
    step_id: Optional[str] = None
    state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
