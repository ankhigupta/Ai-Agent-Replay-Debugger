"""LangGraph -> AgentDebug Trace/Span translation.

    LangGraph graph -> LangGraphAdapter.capture() -> AgentDebug Trace
        -> existing Phase 3 checks + Phase 4 localization (unchanged)

This module is the ONLY place in this integration that imports LangGraph.
`agentdebug.core`/`recording`/`analysis` know nothing about it. This is a
fresh translation to the AgentDebug `Trace`/`Span` model — it does not
extend or depend on any prior, replay-oriented LangGraph adapter design
(none remain in this repository's final V1).

Design, verified empirically against the installed langgraph 1.2.11 /
langgraph-checkpoint 4.2.0 (commands run and their output are in this
phase's report, not reproduced here):

- Per-node execution evidence comes from `graph.stream(input, config,
  stream_mode="updates")`, which yields one `{node_name: node_output}` dict
  per node as it completes — this is the actual execution-activity
  evidence, confirmed NOT the same thing as a checkpoint (a checkpoint can
  be produced by more than one node at once in a fan-out, and the very
  first checkpoint transition is LangGraph's own input-merge bootstrap,
  not a user node — see `_translate_steps`).
- `graph.get_state_history(config)` requires a checkpointer; calling it on
  a graph compiled without one raises `ValueError: No checkpointer set`.
  Checked via `graph.checkpointer is not None` (a real, observed public
  attribute) BEFORE calling it, rather than relying on catching that error.
- A node's call input is not directly exposed by `stream_mode="updates"`
  (which yields only each node's own output). It IS honestly derivable
  from checkpoint history, though: LangGraph invokes node(s) with the
  current graph state as their argument, and `StateSnapshot.values` from
  the checkpoint immediately BEFORE a transition is exactly that state —
  not a reconstruction or a guess, but the literal argument value. This is
  only available when a checkpointer is configured.
- `StateGraph.add_node(name, fn, metadata={...})` is a real, public
  LangGraph API (confirmed via `inspect.signature`), and the resulting
  metadata is readable back from the compiled graph via
  `graph.get_graph().nodes[name].metadata` (confirmed empirically). This
  integration uses that — NOT auto-detection — as the only legitimate way
  to assign a non-CUSTOM `SpanKind` to a node: LangGraph's generic
  stream/checkpoint APIs carry no semantic category for a node at all, so
  guessing one would violate "never claim a category the evidence doesn't
  establish". A graph author who knows a node makes a tool call can say so
  explicitly via `metadata={"span_kind": "TOOL"}`; nodes without this
  declaration are `SpanKind.CUSTOM` (the honest default — "updates state"
  describes literally every LangGraph node equally, so it is not more
  specific evidence than CUSTOM).
- LangGraph subgraph nesting is NOT handled in this version: a flat
  `StateGraph`'s nodes are genuinely siblings, not parent/child, so every
  captured Span has `parent_id=None`. That is accurate for a flat graph,
  not a limitation being papered over — representing non-existent nesting
  would be the actual violation here.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from agentdebug.core.context import TraceContext
from agentdebug.core.enums import Capability, SpanKind, SpanStatus
from agentdebug.core.serialization import is_json_serializable
from agentdebug.core.span import Span
from agentdebug.core.trace import Trace

try:
    from langgraph.graph import START
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "agentdebug.integrations.langgraph requires the 'langgraph' package. "
        "Install it to use this integration; agentdebug.core/recording/analysis "
        "do not require it."
    ) from exc

DEFAULT_NODE_SPAN_KIND_KEY = "span_kind"


def _json_safe(value: Any) -> Any:
    """Mirrors `agentdebug.analysis.localization.first_observed._json_safe`:
    Span.input/output must be JSON-compatible (a Phase 1 invariant), so a
    non-JSON-safe LangGraph value is reported via `str()` rather than
    raising or being silently dropped."""
    return value if is_json_serializable(value) else str(value)


class LangGraphAdapter:
    """Captures one LangGraph execution as a framework-independent
    AgentDebug `Trace`.

    `node_span_kind_key` is the `add_node(..., metadata={...})` key this
    adapter reads to decide a node's `SpanKind` (default `"span_kind"`,
    value must be a valid `SpanKind` member name, e.g. `"TOOL"`). Nodes
    without a recognized value are `SpanKind.CUSTOM`.
    """

    def __init__(self, graph: Any, *, node_span_kind_key: str = DEFAULT_NODE_SPAN_KIND_KEY) -> None:
        self._graph = graph
        self._node_span_kind_key = node_span_kind_key
        self._traces: dict[str, Trace] = {}

    @property
    def capabilities(self) -> frozenset[Capability]:
        """What this adapter can actually expose for THIS graph.

        `PRE_RUN_CONTEXT` is always available: the input passed to
        `capture()` is itself genuine pre-run state, independent of
        checkpointing. `SPAN_IO`/`TOOL_STATUS`/`TOOL_CALLS` all depend on
        checkpoint history being available (it is what lets this adapter
        honestly derive span input and attribute a failure to a node, see
        module docstring) — without a checkpointer configured on the
        graph, none of those three are advertised, even though node
        *output* alone would technically still be capturable via
        `stream()`. This is a deliberate simplification (a finer-grained
        capability split is possible but not worth the taxonomy growth
        for V1) — see this phase's report for the explicit test of this
        degraded case.
        """
        capabilities = {Capability.PRE_RUN_CONTEXT}
        if getattr(self._graph, "checkpointer", None) is not None:
            capabilities.update({Capability.SPAN_IO, Capability.TOOL_STATUS, Capability.TOOL_CALLS})
        return frozenset(capabilities)

    def get_trace(self, trace_id: str) -> Trace:
        """Return a previously captured Trace by id (mirrors the old
        adapter's `get_execution`)."""
        try:
            return self._traces[trace_id]
        except KeyError:
            raise KeyError(f"No captured trace found for trace_id={trace_id!r}") from None

    def capture(self, input: Any, *, trace_id: Optional[str] = None) -> Trace:
        """Run the wrapped graph with `input` and return an AgentDebug Trace.

        Executes via `graph.stream(input, config, stream_mode="updates")`
        — the one call that actually runs the graph; there is no separate
        `invoke()`, which would otherwise run it twice. If the graph
        raises, the exception is not swallowed: the partial Trace built so
        far (including an ERROR span when the evidence supports
        attributing it, see `_error_span`) is still stored and retrievable
        via `get_trace(trace_id)`, and the original exception is then
        re-raised.
        """
        graph = self._graph
        has_checkpointer = getattr(graph, "checkpointer", None) is not None
        resolved_trace_id = trace_id or str(uuid.uuid4())
        thread_config: dict[str, Any] = (
            {"configurable": {"thread_id": resolved_trace_id}} if has_checkpointer else {}
        )

        # `TraceContext.initial_state` must stay a dict; if `input` isn't a
        # JSON-safe dict, fall back to an honest empty state rather than
        # collapsing it into some other shape.
        initial_state = input if isinstance(input, dict) and is_json_serializable(input) else {}
        trace = Trace(id=resolved_trace_id, context=TraceContext(initial_state=initial_state))
        self._traces[resolved_trace_id] = trace

        stream_chunks: list[tuple[str, Any]] = []
        error: Optional[BaseException] = None
        try:
            for update in graph.stream(input, thread_config, stream_mode="updates"):
                if isinstance(update, dict):
                    stream_chunks.extend(update.items())
        except Exception as exc:  # noqa: BLE001 - deliberately captured, then re-raised below
            error = exc

        history: list[Any] = []
        if has_checkpointer:
            history = list(reversed(list(graph.get_state_history(thread_config))))

        sequence = _Sequence()
        node_kinds = self._node_kinds()
        for span in self._translate_steps(history, stream_chunks, node_kinds, has_checkpointer, sequence):
            trace.add_span(span)

        if error is not None:
            trace.add_span(self._error_span(history, error, node_kinds, sequence))
            raise error

        return trace

    # -- internal translation helpers ------------------------------------

    def _node_kinds(self) -> dict[str, SpanKind]:
        """Node name -> explicitly-declared SpanKind, read from
        `add_node(..., metadata={self._node_span_kind_key: "..."})` via
        `graph.get_graph().nodes[name].metadata` (see module docstring).
        Never guesses: an unrecognized or missing value is simply absent
        from this mapping, and the caller defaults to CUSTOM.
        """
        kinds: dict[str, SpanKind] = {}
        try:
            structural = self._graph.get_graph()
        except Exception:  # noqa: BLE001 - structural introspection is best-effort
            return kinds
        for node_id, node in getattr(structural, "nodes", {}).items():
            metadata = getattr(node, "metadata", None) or {}
            raw_kind = metadata.get(self._node_span_kind_key)
            if raw_kind is None:
                continue
            try:
                kinds[node_id] = SpanKind(raw_kind)
            except ValueError:
                continue
        return kinds

    def _translate_steps(
        self,
        history: list[Any],
        stream_chunks: list[tuple[str, Any]],
        node_kinds: dict[str, SpanKind],
        has_checkpointer: bool,
        sequence: "_Sequence",
    ) -> list[Span]:
        """Translate stream updates (+ checkpoint history, if available)
        into Spans, in chronological order. See module docstring for the
        checkpoint<->node attribution rule and the `START`-bootstrap skip.
        """
        if not has_checkpointer:
            # No checkpoint history -> no derivable input and no `next`
            # evidence to correlate chunks to transitions, but each stream
            # chunk is still real node-execution evidence on its own.
            spans = []
            for node_name, node_output in stream_chunks:
                spans.append(
                    self._make_span(
                        node_name,
                        node_kinds.get(node_name, SpanKind.CUSTOM),
                        input=None,
                        output=node_output,
                        sequence=sequence,
                        metadata={"langgraph_checkpoint_id": None},
                    )
                )
            return spans

        spans = []
        chunk_cursor = 0
        for index in range(1, len(history)):
            previous, current = history[index - 1], history[index]
            previous_next = list(getattr(previous, "next", None) or ())

            if index == 1 and previous_next == [START]:
                continue  # LangGraph's own input-merge bootstrap, not a user node

            correlation_uncertain = False
            expected_count = len(previous_next)
            if expected_count == 0:
                expected_count = 1 if chunk_cursor < len(stream_chunks) else 0
                correlation_uncertain = True
            consumed = stream_chunks[chunk_cursor : chunk_cursor + expected_count]
            chunk_cursor += len(consumed)
            if len(consumed) != expected_count:
                correlation_uncertain = True

            checkpoint_id, _ = self._checkpoint_id_from_config(getattr(current, "config", None))
            node_input = getattr(previous, "values", None)

            for node_name, node_output in consumed:
                spans.append(
                    self._make_span(
                        node_name,
                        node_kinds.get(node_name, SpanKind.CUSTOM),
                        input=node_input,
                        output=node_output,
                        sequence=sequence,
                        metadata={
                            "langgraph_checkpoint_id": checkpoint_id,
                            "correlation_uncertain": correlation_uncertain,
                        },
                    )
                )

        for offset, (node_name, node_output) in enumerate(stream_chunks[chunk_cursor:]):
            spans.append(
                self._make_span(
                    node_name,
                    node_kinds.get(node_name, SpanKind.CUSTOM),
                    input=None,
                    output=node_output,
                    sequence=sequence,
                    metadata={"langgraph_checkpoint_id": None, "correlation_uncertain": True},
                )
            )

        return spans

    def _make_span(
        self,
        name: str,
        kind: SpanKind,
        *,
        input: Any,
        output: Any,
        sequence: "_Sequence",
        metadata: dict[str, Any],
    ) -> Span:
        start_time = time.monotonic()
        end_time = time.monotonic()
        return Span(
            id=str(uuid.uuid4()),
            parent_id=None,  # flat StateGraph nodes are siblings, not nested -- see module docstring
            kind=kind,
            name=name,
            status=SpanStatus.OK,
            start_time=start_time,
            end_time=max(end_time, start_time),
            recorder_sequence_id=sequence.next(),
            input=_json_safe(input) if input is not None else None,
            output=_json_safe(output) if output is not None else None,
            metadata=metadata,
        )

    def _error_span(
        self,
        history: list[Any],
        error: BaseException,
        node_kinds: dict[str, SpanKind],
        sequence: "_Sequence",
    ) -> Span:
        """A Span with `status=ERROR` for a run that raised. Node
        attribution is best-available evidence, not certainty: the last
        checkpoint's `next` names the node(s) LangGraph was about to
        invoke when the stream stopped producing updates. If exactly one
        is named, the span is named after it and uses that node's
        explicitly-declared `SpanKind` if it has one (e.g. so a failing
        TOOL-declared node still produces a TOOL/ERROR span that
        `SwallowedToolErrorCheck` can see) — otherwise, or when
        attribution is ambiguous, `SpanKind.ERROR` is used and all
        candidates are preserved in metadata rather than guessing one.
        """
        candidates = tuple(getattr(history[-1], "next", None) or ()) if history else ()
        name = candidates[0] if len(candidates) == 1 else "<graph_execution>"
        kind = node_kinds.get(name, SpanKind.ERROR) if len(candidates) == 1 else SpanKind.ERROR
        now = time.monotonic()
        return Span(
            id=str(uuid.uuid4()),
            parent_id=None,
            kind=kind,
            name=name,
            status=SpanStatus.ERROR,
            start_time=now,
            end_time=now,
            recorder_sequence_id=sequence.next(),
            metadata={
                "exception": {"type": type(error).__name__, "message": str(error)},
                "attributed_nodes": list(candidates),
                "attribution_basis": (
                    "last observed checkpoint's 'next' field (the node(s) LangGraph was "
                    "about to invoke)"
                    if candidates
                    else "no checkpoint history available to attribute a node"
                ),
            },
        )

    @staticmethod
    def _checkpoint_id_from_config(config: Any) -> tuple[Optional[str], bool]:
        """Extract a checkpoint id from a LangGraph state-snapshot config,
        for metadata purposes only (never used as a Span identity)."""
        if isinstance(config, dict):
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                checkpoint_id = configurable.get("checkpoint_id")
                if checkpoint_id:
                    return str(checkpoint_id), False
        return None, True


class _Sequence:
    """A trivial monotonic counter for `recorder_sequence_id`.

    `capture()` processes one graph run as a single linear, synchronous
    pass over already-produced LangGraph evidence (stream chunks +
    checkpoint history) — there is no concurrency here, so this needs none
    of Phase 2 Recorder's locking, only a counter starting at 0.
    """

    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        value = self._value
        self._value += 1
        return value
