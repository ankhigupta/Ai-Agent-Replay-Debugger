"""Tests for LangGraphAdapter's capture/retrieval translation logic.

Requires `langgraph` to be importable (adapter.py imports `langgraph.graph`
directly, since it is the LangGraph-specific adapter) — run these with the
interpreter that has it installed, e.g.:

    backend/.venv/bin/python -m unittest tests.test_langgraph_adapter -v

Two layers of coverage:

1. Fake-graph unit tests (`FakeGraphTranslationTests`): a minimal
   duck-typed stub (`_FakeGraph`/`_FakeSnapshot`) stands in for a compiled
   LangGraph graph, exercising the adapter's own translation logic (state
   history + stream updates -> Checkpoint/ExecutionStep, error handling,
   id synthesis, checkpoint<->node attribution) without depending on real
   LangGraph execution behavior. Fast and deterministic.

2. Real-LangGraph integration test (`RealLangGraphIntegrationTests`): runs
   the adapter against an actual compiled `StateGraph`, the same way the
   POC does, including a fan-out (parallel) graph to exercise the
   one-to-many checkpoint/node case for real.
"""

from __future__ import annotations

import operator
import unittest
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from app.adapters.langgraph.adapter import LangGraphAdapter
from app.models.checkpoint import Checkpoint
from app.models.execution import ExecutionStatus
from app.models.step import ExecutionStep, StepType

try:
    from langgraph.graph import START
except ImportError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"langgraph not importable, cannot test LangGraphAdapter: {exc}")


@dataclass
class _FakeSnapshot:
    """Duck-typed stand-in for a LangGraph StateSnapshot."""

    values: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    next: tuple[str, ...] = ()


class _FakeGraph:
    """Duck-typed stand-in for a compiled LangGraph graph.

    Only implements the two methods LangGraphAdapter actually calls:
    `stream` (stream_mode="updates") and `get_state_history`. `history`
    must be supplied newest-first, matching what the adapter assumes of
    the real API. `updates` is the list of {node_name: output} dicts the
    fake "stream" yields, in chronological order.
    """

    def __init__(
        self,
        history: list[_FakeSnapshot],
        updates: list[dict[str, Any]] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self._history = history
        self._updates = updates or []
        self._stream_error = stream_error

    def stream(self, input: Any, config: Any, stream_mode: str = "updates"):
        assert stream_mode == "updates"
        for update in self._updates:
            yield update
        if self._stream_error is not None:
            raise self._stream_error

    def get_state_history(self, config: Any) -> list[_FakeSnapshot]:
        return list(self._history)


def _config(checkpoint_id: str | None) -> dict[str, Any]:
    configurable: dict[str, Any] = {"thread_id": "irrelevant-in-fake"}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


class FakeGraphTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = LangGraphAdapter()

    def test_capabilities_declared_correctly(self) -> None:
        caps = self.adapter.capabilities
        self.assertTrue(caps.execution_capture)
        self.assertTrue(caps.trace_inspection)
        self.assertTrue(caps.checkpoint_access)
        self.assertFalse(caps.state_forking)
        self.assertFalse(caps.replay)
        self.assertFalse(caps.record_replay)

    def test_capture_execution_linear_history_one_node_per_checkpoint(self) -> None:
        # newest-first, as the adapter assumes of graph.get_state_history
        history = [
            _FakeSnapshot(
                values={"x": 2},
                config=_config("cp2"),
                metadata={"step": 2, "source": "loop", "parents": {}},
                next=(),
            ),
            _FakeSnapshot(
                values={"x": 1},
                config=_config("cp1"),
                metadata={"step": 1, "source": "loop", "parents": {}},
                next=("node_b",),
            ),
            _FakeSnapshot(
                values={"x": 0},
                config=_config("cp0"),
                metadata={"step": 0, "source": "loop", "parents": {}},
                next=("node_a",),
            ),
            _FakeSnapshot(
                values={},
                config=_config("cp-input"),
                metadata={"step": -1, "source": "input", "parents": {}},
                next=(START,),
            ),
        ]
        updates = [{"node_a": {"x": 1}}, {"node_b": {"x": 2}}]
        graph = _FakeGraph(history=history, updates=updates)

        execution = self.adapter.capture_execution(graph, {"x": 0})

        self.assertEqual(execution.status, ExecutionStatus.COMPLETED)
        self.assertEqual(execution.framework, "langgraph")
        self.assertEqual(execution.input, {"x": 0})
        self.assertIsNotNone(execution.started_at)
        self.assertIsNotNone(execution.ended_at)

        # chronological order: cp-input -> cp0 -> cp1 -> cp2
        self.assertEqual(
            [c.checkpoint_id for c in execution.checkpoints], ["cp-input", "cp0", "cp1", "cp2"]
        )
        for checkpoint in execution.checkpoints:
            self.assertIsInstance(checkpoint, Checkpoint)
            # Deliberately unset: see adapter module docstring.
            self.assertIsNone(checkpoint.step_id)
            self.assertFalse(checkpoint.metadata["checkpoint_id_synthesized"])

        # cp-input -> cp0 is LangGraph's own bootstrap (next == (START,)):
        # no ExecutionStep. cp0 -> cp1 is node_a. cp1 -> cp2 is node_b.
        self.assertEqual(len(execution.steps), 2)
        step_names = [s.name for s in execution.steps]
        self.assertEqual(step_names, ["node_a", "node_b"])
        for step in execution.steps:
            self.assertIsInstance(step, ExecutionStep)
            self.assertEqual(step.type, StepType.OTHER)
            self.assertFalse(step.metadata["correlation_uncertain"])
        self.assertEqual(execution.steps[0].output, {"x": 1})
        self.assertEqual(execution.steps[0].step_id, "cp1:node_a")
        self.assertEqual(execution.steps[1].output, {"x": 2})
        self.assertEqual(execution.steps[1].step_id, "cp2:node_b")

        # round trip through the storage accessors
        self.assertEqual(self.adapter.get_execution(execution.execution_id), execution)
        self.assertEqual(self.adapter.get_steps(execution.execution_id), execution.steps)
        self.assertEqual(self.adapter.get_checkpoints(execution.execution_id), execution.checkpoints)

    def test_single_checkpoint_can_yield_multiple_steps_parallel_writes(self) -> None:
        """Two nodes racing into one join, as verified against real LangGraph:
        separate stream-update chunks, but ONE checkpoint transition
        (`next` on the prior snapshot names both). This is why
        Checkpoint.step_id stays unset rather than picking one owning step."""
        history = [
            _FakeSnapshot(
                values={"a": 1, "b": 1},
                config=_config("cp1"),
                metadata={"step": 0, "source": "loop", "parents": {}},
                next=(),
            ),
            _FakeSnapshot(
                values={"a": 0, "b": 0},
                config=_config("cp0"),
                metadata={"step": -1, "source": "input", "parents": {}},
                next=("node_a", "node_b"),
            ),
        ]
        updates = [{"node_a": 1}, {"node_b": 1}]
        graph = _FakeGraph(history=history, updates=updates)

        execution = self.adapter.capture_execution(graph, {"a": 0, "b": 0})

        self.assertEqual(len(execution.checkpoints), 2)
        self.assertEqual(len(execution.steps), 2)
        self.assertEqual({s.name for s in execution.steps}, {"node_a", "node_b"})
        # Both steps trace back to the SAME checkpoint - the one-to-many case.
        self.assertTrue(all(s.metadata["checkpoint_id"] == "cp1" for s in execution.steps))
        self.assertTrue(all(not s.metadata["correlation_uncertain"] for s in execution.steps))

    def test_missing_checkpoint_id_is_synthesized_and_flagged(self) -> None:
        history = [
            _FakeSnapshot(values={"x": 0}, config={"configurable": {}}, metadata={}, next=()),
        ]
        graph = _FakeGraph(history=history, updates=[])

        execution = self.adapter.capture_execution(graph, {"x": 0})

        self.assertEqual(len(execution.checkpoints), 1)
        checkpoint = execution.checkpoints[0]
        self.assertTrue(checkpoint.metadata["checkpoint_id_synthesized"])
        self.assertIn(execution.execution_id, checkpoint.checkpoint_id)

    def test_capture_execution_failure_is_recorded_not_swallowed(self) -> None:
        boom = RuntimeError("node_a exploded")
        history = [
            _FakeSnapshot(
                values={"x": 0},
                config=_config("cp0"),
                metadata={"step": -1, "source": "input", "parents": {}},
                next=("node_a",),
            ),
        ]
        graph = _FakeGraph(history=history, updates=[], stream_error=boom)

        with self.assertRaises(RuntimeError):
            self.adapter.capture_execution(graph, {"x": 0})

        # The exception propagates (not hidden), but the partial execution
        # was still recorded as FAILED before re-raising.
        self.assertEqual(len(self.adapter._executions), 1)
        stored = next(iter(self.adapter._executions.values()))
        self.assertEqual(stored.status, ExecutionStatus.FAILED)
        error_steps = [s for s in stored.steps if s.name == "execution_error"]
        self.assertEqual(len(error_steps), 1)
        self.assertEqual(error_steps[0].metadata["error_type"], "RuntimeError")
        self.assertEqual(error_steps[0].metadata["error_message"], "node_a exploded")
        # Checkpoint captured before the failure is still present.
        self.assertEqual(len(stored.checkpoints), 1)

    def test_get_execution_unknown_id_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            self.adapter.get_execution("does-not-exist")

    def test_fork_and_replay_not_yet_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.adapter.fork_from_checkpoint("cp0", {"x": 1})
        with self.assertRaises(NotImplementedError):
            self.adapter.replay_from("cp0")


class RealLangGraphIntegrationTests(unittest.TestCase):
    """Runs the adapter against real compiled LangGraph graphs."""

    def test_capture_execution_against_real_langgraph_linear(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph

        class _State(TypedDict):
            counter: int

        def node_a(state: _State) -> dict:
            return {"counter": state["counter"] + 1}

        def node_b(state: _State) -> dict:
            return {"counter": state["counter"] + 10}

        builder = StateGraph(_State)
        builder.add_node("node_a", node_a)
        builder.add_node("node_b", node_b)
        builder.add_edge(START, "node_a")
        builder.add_edge("node_a", "node_b")
        builder.add_edge("node_b", END)
        graph = builder.compile(checkpointer=MemorySaver())

        adapter = LangGraphAdapter()
        execution = adapter.capture_execution(graph, {"counter": 0})

        self.assertEqual(execution.status, ExecutionStatus.COMPLETED)
        self.assertGreater(len(execution.checkpoints), 0)
        for checkpoint in execution.checkpoints:
            self.assertIsNone(checkpoint.step_id)

        step_names = [s.name for s in execution.steps]
        self.assertEqual(step_names, ["node_a", "node_b"])
        for step in execution.steps:
            self.assertEqual(step.type, StepType.OTHER)
            self.assertFalse(step.metadata["correlation_uncertain"])
        self.assertEqual(execution.steps[0].output, {"counter": 1})
        self.assertEqual(execution.steps[1].output, {"counter": 11})

        # Final checkpoint's state should reflect both nodes having run.
        final_state = execution.checkpoints[-1].state
        self.assertEqual(final_state["counter"], 11)

        self.assertEqual(adapter.get_steps(execution.execution_id), execution.steps)
        self.assertEqual(adapter.get_checkpoints(execution.execution_id), execution.checkpoints)

    def test_capture_execution_against_real_langgraph_parallel(self) -> None:
        """Fan-out graph: confirms the one-to-many checkpoint/node case for
        real, not just via the fake stub above."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph

        class _State(TypedDict):
            log: Annotated[list[str], operator.add]

        def node_a(state: _State) -> dict:
            return {"log": ["a"]}

        def node_b(state: _State) -> dict:
            return {"log": ["b"]}

        def node_join(state: _State) -> dict:
            return {"log": ["join"]}

        builder = StateGraph(_State)
        builder.add_node("node_a", node_a)
        builder.add_node("node_b", node_b)
        builder.add_node("node_join", node_join)
        builder.add_edge(START, "node_a")
        builder.add_edge(START, "node_b")
        builder.add_edge("node_a", "node_join")
        builder.add_edge("node_b", "node_join")
        builder.add_edge("node_join", END)
        graph = builder.compile(checkpointer=MemorySaver())

        adapter = LangGraphAdapter()
        execution = adapter.capture_execution(graph, {"log": []})

        self.assertEqual(execution.status, ExecutionStatus.COMPLETED)
        step_names = {s.name for s in execution.steps}
        self.assertEqual(step_names, {"node_a", "node_b", "node_join"})
        for step in execution.steps:
            self.assertFalse(step.metadata["correlation_uncertain"])

        # node_a and node_b must be attributed to the SAME checkpoint
        # (one-to-many), while node_join is attributed to a later one.
        a_step = next(s for s in execution.steps if s.name == "node_a")
        b_step = next(s for s in execution.steps if s.name == "node_b")
        join_step = next(s for s in execution.steps if s.name == "node_join")
        self.assertEqual(a_step.metadata["checkpoint_id"], b_step.metadata["checkpoint_id"])
        self.assertNotEqual(a_step.metadata["checkpoint_id"], join_step.metadata["checkpoint_id"])

        final_state = execution.checkpoints[-1].state
        self.assertEqual(sorted(final_state["log"]), sorted(["a", "b", "join"]))


if __name__ == "__main__":
    unittest.main()
