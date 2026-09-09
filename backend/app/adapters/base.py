"""Framework-neutral adapter contract for the AI Agent Replay Debugger.

An "adapter" is the integration layer between one specific agent framework
(LangGraph, CrewAI, etc.) and the rest of the debugger. Nothing in this
module imports or depends on any concrete framework — concrete adapters
live elsewhere and implement this interface.

Responsibility boundary (adapter vs. debugger core):

    An AgentAdapter exposes framework-level MECHANISMS only — capturing a
    run, reading back its steps/checkpoints, and (where the framework
    supports it) forking a new branch of execution from a checkpoint and
    running it. An adapter has no notion of *why* a fork or replay is
    being performed.

    The debugger core owns all debugging/experiment semantics: forming a
    hypothesis about a failure, deciding which checkpoint to branch from
    and what state to override, invoking the adapter's mechanisms to
    produce a branch and a replayed execution, comparing the original and
    replayed executions, detecting divergence, and classifying the result
    (e.g. SUPPORTED / REFUTED / INCONCLUSIVE). None of that vocabulary or
    logic belongs in this module or in any concrete adapter — an adapter
    that starts reasoning about "interventions", "hypotheses", or
    "experiment results" has taken on responsibility that belongs to the
    debugger core instead.

Not every framework can support every capability (e.g. some frameworks may
not expose checkpoints, or may not allow forking state mid-run). Rather
than forcing every adapter to fake support it doesn't have, adapters must
explicitly declare what they support via `AdapterCapabilities`, and callers
are expected to check capabilities before relying on a given method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from app.models.checkpoint import Checkpoint
from app.models.execution import Execution
from app.models.step import ExecutionStep


@dataclass(frozen=True)
class AdapterCapabilities:
    """Declares which debugger-facing mechanisms a given adapter supports.

    Every field defaults to False so that a capability must be explicitly
    opted into by an adapter rather than assumed. Callers should inspect an
    adapter's capabilities before invoking the corresponding methods.

    Attributes:
        execution_capture: Adapter can observe and record a live execution
            as it runs.
        trace_inspection: Adapter can return the recorded steps of a
            previously captured execution.
        checkpoint_access: Adapter can return state checkpoints captured
            during an execution.
        state_forking: Adapter can create a new framework-level branch from
            a checkpoint, optionally with a state override applied. This is
            a mechanical capability only — it says nothing about why a
            fork is being made; that reasoning belongs to the debugger
            core.
        replay: Adapter can execute an execution starting from a
            checkpoint/fork/anchor point.
        record_replay: Adapter supports deterministic record/replay, i.e.
            re-running an execution reproduces the same steps/outputs
            rather than merely re-invoking the agent from scratch.
    """

    execution_capture: bool = False
    trace_inspection: bool = False
    checkpoint_access: bool = False
    state_forking: bool = False
    replay: bool = False
    record_replay: bool = False


class AgentAdapter(ABC):
    """Abstract interface every framework-specific adapter must implement.

    Implementations translate a concrete framework's native execution
    representation into the framework-neutral models defined in
    `app.models` (Execution, ExecutionStep, Checkpoint). This class defines
    only framework-level mechanisms — capture, retrieval, forking, and
    replay execution. It defines no debugging/experiment semantics: no
    hypothesis, intervention, comparison, divergence, or verdict logic.
    That logic is the debugger core's responsibility and is orchestrated
    on top of these mechanisms, not inside them. See the module docstring
    for the full responsibility boundary.
    """

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Return the set of capabilities this adapter supports."""
        raise NotImplementedError

    @abstractmethod
    def capture_execution(self, agent: Any, input: Any) -> Execution:
        """Run `agent` with `input` and capture it as an Execution.

        Only meaningful if `capabilities.execution_capture` is True.
        """
        raise NotImplementedError

    @abstractmethod
    def get_execution(self, execution_id: str) -> Execution:
        """Return the previously captured Execution for `execution_id`."""
        raise NotImplementedError

    @abstractmethod
    def get_steps(self, execution_id: str) -> list[ExecutionStep]:
        """Return the recorded steps for `execution_id`.

        Only meaningful if `capabilities.trace_inspection` is True.
        """
        raise NotImplementedError

    @abstractmethod
    def get_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        """Return the recorded checkpoints for `execution_id`.

        Only meaningful if `capabilities.checkpoint_access` is True.
        """
        raise NotImplementedError

    @abstractmethod
    def fork_from_checkpoint(
        self, checkpoint_id: str, state_override: dict[str, Any]
    ) -> str:
        """Create a new framework-level branch from a checkpoint.

        This is a mechanical operation: given a checkpoint and a
        framework-neutral state override, the adapter asks the underlying
        framework to materialize a new branch/fork whose state incorporates
        that override. The adapter does not interpret what the override
        means or why it is being applied — deciding what to change and why
        is debugger-core logic (e.g. a "controlled intervention"), not an
        adapter concern.

        Returns an adapter-defined anchor identifier (e.g. a new checkpoint
        or branch id) that can be passed to `replay_from`. Only meaningful
        if `capabilities.state_forking` is True.
        """
        raise NotImplementedError

    @abstractmethod
    def replay_from(self, anchor_id: str) -> Execution:
        """Execute the framework-level mechanism to run from an anchor.

        `anchor_id` identifies a checkpoint, fork, or other framework-level
        starting point (such as one returned by `fork_from_checkpoint`).
        This method only performs the run and returns the resulting
        Execution — it does not compare that Execution to any other run,
        detect divergence, or classify an outcome; that is debugger-core
        logic. Only meaningful if `capabilities.replay` is True. Whether
        the run is deterministic (reproducing recorded outputs) or simply
        re-invokes the agent depends on `capabilities.record_replay`.
        """
        raise NotImplementedError
