"""TraceContext: inputs and state captured before any span executes.

Deliberately distinct from STATE-kind Spans: `initial_state` is a single
snapshot of state as it existed before the run started, while a STATE span
represents a state *change* that happened during execution. Conflating the
two would make it impossible to tell, during debugging, whether a value
was there from the start or was produced mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentdebug.core.exceptions import TraceValidationError
from agentdebug.core.serialization import check_known_keys, is_json_serializable, require_dict

_ALLOWED_KEYS = frozenset({"initial_state", "user_input", "system_prompt", "config"})


@dataclass(frozen=True, kw_only=True)
class TraceContext:
    """Inputs and pre-run state a Trace was started with.

    Attributes:
        initial_state: Snapshot of state as it existed before any span
            ran. Must be JSON serializable.
        user_input: The input the run was triggered with, if any. Must be
            JSON serializable when not None.
        system_prompt: The system prompt the run was configured with, if
            any.
        config: Arbitrary run configuration. Must be JSON serializable.
    """

    initial_state: dict[str, Any] = field(default_factory=dict)
    user_input: Optional[Any] = None
    system_prompt: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_json_serializable(self.initial_state):
            raise TraceValidationError("TraceContext.initial_state must be JSON serializable")
        if not is_json_serializable(self.config):
            raise TraceValidationError("TraceContext.config must be JSON serializable")
        if self.user_input is not None and not is_json_serializable(self.user_input):
            raise TraceValidationError("TraceContext.user_input must be JSON serializable")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise TraceValidationError("TraceContext.system_prompt must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": self.initial_state,
            "user_input": self.user_input,
            "system_prompt": self.system_prompt,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceContext":
        data = require_dict(data, "TraceContext")
        check_known_keys(data, _ALLOWED_KEYS, "TraceContext")
        return cls(
            initial_state=data.get("initial_state", {}),
            user_input=data.get("user_input"),
            system_prompt=data.get("system_prompt"),
            config=data.get("config", {}),
        )
