"""LangGraph implementation of the framework-neutral agent adapter.

This module implements only the framework-level mechanisms the debugger
core needs: capturing a LangGraph run, and reading back the resulting
Execution/steps/checkpoints. It does not interpret *why* anything happened
— no hypothesis, intervention, comparison, or verdict logic lives here (see
`app.adapters.base` for that boundary).

Unlike `app.adapters.base` and `app.models`, this module DOES import
LangGraph directly — it is the LangGraph-specific adapter, so depending on
`langgraph` here is the point, not a violation of the framework-neutral
boundary.

Design, verified empirically against a real langgraph 1.2.11 /
langgraph-checkpoint 4.2.0 install (see `backend/experiments/langgraph_checkpoint_poc.py`
for the original exploration; the findings below were re-confirmed while
building this adapter, because the POC's assumption about a `writes` key
in checkpoint metadata turned out to be wrong for this version):

- `agent` passed to `capture_execution` is a compiled LangGraph graph
  (`StateGraph(...).compile(checkpointer=...)`). The adapter contract
  types this as `Any` because that is a framework detail; this module is
  the one place that assumption is made concrete.
- The adapter owns thread identity: it uses the generated `execution_id`
  as the LangGraph `thread_id`, since the adapter contract has no separate
  thread/session concept and LangGraph requires one per checkpointed run.
- Per-node output is captured via `agent.stream(input, config,
  stream_mode="updates")`, NOT `agent.invoke()`. Confirmed directly:
  `graph.get_state_history()` checkpoint metadata in this version is only
  `{"source", "step", "parents"}` — there is no `writes` field naming which
  node(s) produced a checkpoint. `stream_mode="updates"` is the API that
  actually yields `{node_name: node_output}` per node execution.
- Checkpoint-to-node attribution is derived, not assumed: each
  `StateSnapshot.next` names the node(s) about to run after that
  checkpoint. So the node(s) that produced checkpoint N are exactly the
  ones named in checkpoint (N-1)'s `next`. This was verified with a
  fan-out graph (two nodes racing into one join node): both nodes'
  `stream(..., stream_mode="updates")` chunks arrive as separate updates,
  but both are consumed by the SAME single checkpoint transition — a
  concretely observed one-to-many relationship, which is why
  `Checkpoint.step_id` is left unset rather than assigned to one node.
- The very first checkpoint transition (from the raw-input snapshot, whose
  `next` is `(START,)`) is LangGraph's own input-merge step, not a
  user-defined node; it produces no `stream_mode="updates"` chunk and is
  not turned into an ExecutionStep.
- LangGraph does not report a step's semantic category (LLM vs. tool vs.
  retrieval, etc.). Every step captured here is recorded as
  `StepType.OTHER`, with raw LangGraph step/source metadata preserved on
  `ExecutionStep.metadata` so a future revision can refine classification
  without this module guessing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import START

from app.adapters.base import AdapterCapabilities, AgentAdapter
from app.models.checkpoint import Checkpoint
from app.models.execution import Execution, ExecutionStatus
from app.models.step import ExecutionStep, StepType

FRAMEWORK_NAME = "langgraph"


class LangGraphAdapter(AgentAdapter):
    """Adapter that exposes LangGraph execution mechanisms to the debugger.

    Storage is a plain in-memory dict keyed by execution_id. This is a
    first production slice, not a persistence layer — process restarts
    lose everything, and nothing here is thread-safe. A database-backed
    store is a separate, later concern (explicitly out of scope here).
    """

    def __init__(self) -> None:
        self._executions: dict[str, Execution] = {}

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Return the mechanisms currently supported by this adapter."""
        return AdapterCapabilities(
            execution_capture=True,
            trace_inspection=True,
            checkpoint_access=True,
            state_forking=False,
            replay=False,
            record_replay=False,
        )

    def capture_execution(self, agent: Any, input: Any) -> Execution:
        """Run a compiled LangGraph graph and capture it as an Execution.

        Executes the graph via `agent.stream(input, config,
        stream_mode="updates")` on a dedicated thread (the generated
        execution_id) — this is the call that actually runs the graph;
        there is no separate `invoke()` call, which would otherwise
        execute it twice. Per-node updates from the stream and the
        resulting checkpoint history are then translated into the common
        `Execution` / `ExecutionStep` / `Checkpoint` models.

        If the agent raises during execution, the exception is not
        swallowed: it is recorded (status FAILED, error captured in a
        synthetic trailing step, using whatever partial checkpoints/steps
        exist up to the failure) and then re-raised, so callers cannot
        mistake a failed run for a successful one by only checking the
        return value.
        """
        execution_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": execution_id}}
        agent_name = getattr(agent, "name", None) or type(agent).__name__
        started_at = datetime.now(timezone.utc)

        execution = Execution(
            execution_id=execution_id,
            agent_name=agent_name,
            framework=FRAMEWORK_NAME,
            started_at=started_at,
            status=ExecutionStatus.RUNNING,
            input=input,
        )
        self._executions[execution_id] = execution

        stream_chunks: list[tuple[str, Any]] = []
        error: Optional[BaseException] = None
        try:
            for update in agent.stream(input, thread_config, stream_mode="updates"):
                if isinstance(update, dict):
                    stream_chunks.extend(update.items())
        except Exception as exc:  # noqa: BLE001 - deliberately captured, then re-raised below
            error = exc

        # Capture whatever checkpoint history exists regardless of success/
        # failure — a failed run may still have produced useful checkpoints
        # for the steps that completed before the failure.
        checkpoints, steps = self._translate_history(
            agent, thread_config, execution_id, stream_chunks
        )
        execution.checkpoints = checkpoints
        execution.steps = steps
        execution.ended_at = datetime.now(timezone.utc)

        if error is not None:
            execution.status = ExecutionStatus.FAILED
            execution.steps.append(
                ExecutionStep(
                    step_id=f"{execution_id}:error",
                    name="execution_error",
                    type=StepType.OTHER,
                    input=None,
                    output=None,
                    started_at=execution.ended_at,
                    ended_at=execution.ended_at,
                    metadata={
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
            )
            raise error

        execution.status = ExecutionStatus.COMPLETED
        return execution

    def get_execution(self, execution_id: str) -> Execution:
        """Return the previously captured Execution for `execution_id`."""
        try:
            return self._executions[execution_id]
        except KeyError:
            raise KeyError(f"No captured execution found for execution_id={execution_id!r}") from None

    def get_steps(self, execution_id: str) -> list[ExecutionStep]:
        """Return the recorded steps for `execution_id`."""
        return self.get_execution(execution_id).steps

    def get_checkpoints(self, execution_id: str) -> list[Checkpoint]:
        """Return the recorded checkpoints for `execution_id`."""
        return self.get_execution(execution_id).checkpoints

    def fork_from_checkpoint(
        self,
        checkpoint_id: str,
        state_override: dict[str, Any],
    ) -> str:
        raise NotImplementedError

    def replay_from(self, anchor_id: str) -> Execution:
        raise NotImplementedError

    # -- internal translation helpers ------------------------------------

    def _translate_history(
        self,
        agent: Any,
        thread_config: dict[str, Any],
        execution_id: str,
        stream_chunks: list[tuple[str, Any]],
    ) -> tuple[list[Checkpoint], list[ExecutionStep]]:
        """Translate LangGraph state history + stream updates into common models.

        Returns (checkpoints, steps) in chronological order.

        Checkpoints come from `get_state_history` (observed newest-first;
        reversed here to process chronologically). Steps come from
        `stream_chunks` (already chronological), attributed to the
        checkpoint whose transition produced them: the node(s) that
        produced checkpoint[i] are exactly the ones named in
        checkpoint[i-1]'s `next` (see module docstring). The very first
        transition, out of the raw-input snapshot (`next == (START,)`), is
        LangGraph's own bootstrap and consumes no stream chunk.
        """
        raw_history = list(agent.get_state_history(thread_config))
        chronological = list(reversed(raw_history))

        checkpoints: list[Checkpoint] = []
        steps: list[ExecutionStep] = []
        chunk_cursor = 0

        for index, snapshot in enumerate(chronological):
            checkpoint_id, id_synthesized = self._checkpoint_id_from_config(
                snapshot.config, execution_id, index
            )
            metadata = getattr(snapshot, "metadata", None) or {}
            checkpoints.append(
                Checkpoint(
                    checkpoint_id=checkpoint_id,
                    # Deliberately left unset: a checkpoint transition can
                    # be produced by more than one node (see module
                    # docstring), so there is no single owning step here.
                    step_id=None,
                    state=snapshot.values,
                    metadata={
                        "langgraph_step": metadata.get("step"),
                        "langgraph_source": metadata.get("source"),
                        "next": list(getattr(snapshot, "next", None) or ()),
                        "checkpoint_id_synthesized": id_synthesized,
                    },
                )
            )

            if index == 0:
                continue  # raw-input snapshot: nothing produced it

            previous_next = list(getattr(chronological[index - 1], "next", None) or ())
            if previous_next == [START]:
                continue  # LangGraph's own input-merge bootstrap, not a user node

            expected_count = len(previous_next)
            correlation_uncertain = False
            if expected_count == 0:
                # No 'next' recorded on the prior snapshot to tell us how
                # many node writes to expect. Consume one chunk if
                # available rather than silently dropping it, and flag
                # the attribution as uncertain.
                expected_count = 1 if chunk_cursor < len(stream_chunks) else 0
                correlation_uncertain = True

            consumed = stream_chunks[chunk_cursor : chunk_cursor + expected_count]
            chunk_cursor += len(consumed)
            if len(consumed) != expected_count:
                correlation_uncertain = True

            for node_name, node_output in consumed:
                steps.append(
                    ExecutionStep(
                        step_id=f"{checkpoint_id}:{node_name}",
                        name=node_name,
                        # LangGraph does not expose a node's semantic
                        # category; see module docstring.
                        type=StepType.OTHER,
                        input=None,
                        output=node_output,
                        metadata={
                            "checkpoint_id": checkpoint_id,
                            "langgraph_step": metadata.get("step"),
                            "langgraph_source": metadata.get("source"),
                            "correlation_uncertain": correlation_uncertain,
                        },
                    )
                )

        # Any stream chunks left unconsumed (shouldn't normally happen)
        # are still recorded, rather than silently dropped, with no
        # checkpoint correlation.
        for offset, (node_name, node_output) in enumerate(stream_chunks[chunk_cursor:]):
            steps.append(
                ExecutionStep(
                    step_id=f"{execution_id}:uncorrelated:{chunk_cursor + offset}:{node_name}",
                    name=node_name,
                    type=StepType.OTHER,
                    input=None,
                    output=node_output,
                    metadata={"checkpoint_id": None, "correlation_uncertain": True},
                )
            )

        return checkpoints, steps

    @staticmethod
    def _checkpoint_id_from_config(
        config: Any, execution_id: str, index: int
    ) -> tuple[str, bool]:
        """Extract a checkpoint id from a LangGraph state-snapshot config.

        Returns (checkpoint_id, was_synthesized). Falls back to a
        synthetic, clearly-marked id if the expected shape isn't present,
        rather than raising and losing the whole capture.
        """
        if isinstance(config, dict):
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                checkpoint_id = configurable.get("checkpoint_id")
                if checkpoint_id:
                    return str(checkpoint_id), False
        return f"{execution_id}:synthetic-checkpoint:{index}", True
