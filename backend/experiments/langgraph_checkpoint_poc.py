"""Isolated LangGraph checkpoint/fork/replay proof-of-concept.

STATUS: executed successfully against langgraph 1.2.11, langgraph-checkpoint
4.2.0, langgraph-sdk 0.4.4. This remains an isolated developer experiment,
not production adapter code — it is not imported by, and does not modify,
the common models or adapter contract. Its observed findings are specific
to the LangGraph version tested above and should be re-verified if that
version changes materially.

Purpose
-------
This is a throwaway experiment, NOT part of the production adapter. It
exists only to answer, empirically, against whatever LangGraph version is
actually installed:

  1. What shape does a LangGraph checkpoint actually have (identifiers,
     metadata)?
  2. Does forking/modifying state at a checkpoint work the way our adapter
     design assumes (`fork_from_checkpoint` / `replay_from`)?
  3. Does the original execution branch survive a fork untouched?
  4. Can a checkpoint be mapped back to a specific node/step 1:1, or is the
     relationship many-to-one / not directly available?
  5. Does continuing from a modified checkpoint actually re-run downstream
     nodes (true continuation), as opposed to either (a) a fresh run from
     the start, or (b) a cached/replayed result that never really re-executes?

No LLM calls. No production models or adapter code are imported or
modified by this script — it is fully self-contained.

How to run
----------
    cd backend
    pip install langgraph langgraph-checkpoint
    python experiments/langgraph_checkpoint_poc.py

Read every printed section top to bottom; the script is intentionally
verbose because the goal is inspection, not a clean demo. If any step
raises, the traceback is allowed to propagate (no blanket try/except) so
the exact API mismatch is visible rather than papered over. A few narrow,
labeled try/except blocks exist ONLY where we are deliberately probing
"does this method/attribute exist in this version" — those print the exact
exception instead of swallowing it.
"""

from __future__ import annotations

import sys
from typing import TypedDict


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# 0. Report the installed LangGraph version before assuming any API shape.
# ---------------------------------------------------------------------------
section("0. Environment / version introspection")

try:
    import langgraph  # noqa: F401

    print("langgraph module file:", langgraph.__file__)
    print("langgraph.__version__:", getattr(langgraph, "__version__", "<not exposed>"))
except ImportError as exc:
    print("FATAL: LangGraph is not installed in this interpreter.")
    print("Interpreter:", sys.executable)
    print("Import error:", exc)
    print("Install with: pip install langgraph langgraph-checkpoint")
    raise

try:
    import importlib.metadata as importlib_metadata

    for dist_name in ("langgraph", "langgraph-checkpoint", "langgraph-sdk"):
        try:
            print(f"{dist_name} distribution version:", importlib_metadata.version(dist_name))
        except importlib_metadata.PackageNotFoundError:
            print(f"{dist_name} distribution: not found (may be bundled differently)")
except Exception as exc:  # pragma: no cover - pure diagnostics
    print("Could not introspect package metadata:", exc)

from langgraph.graph import END, START, StateGraph

# The checkpointer import path has moved between LangGraph versions
# (older: langgraph.checkpoint.memory; some releases: langgraph.checkpoint).
# Report exactly which import path succeeded rather than assuming one.
MemorySaver = None
for module_path in ("langgraph.checkpoint.memory", "langgraph.checkpoint"):
    try:
        mod = __import__(module_path, fromlist=["MemorySaver"])
        MemorySaver = getattr(mod, "MemorySaver")
        print(f"MemorySaver resolved from: {module_path}")
        break
    except (ImportError, AttributeError) as exc:
        print(f"MemorySaver NOT found at {module_path}: {exc}")

if MemorySaver is None:
    raise ImportError(
        "Could not locate MemorySaver under any known import path. "
        "Inspect the installed langgraph-checkpoint package layout manually "
        "(e.g. `python -c \"import langgraph.checkpoint; help(langgraph.checkpoint)\"`)."
    )


# ---------------------------------------------------------------------------
# 1. Define a very small graph with several distinct, observable nodes.
# ---------------------------------------------------------------------------
section("1. Building the graph")


class PocState(TypedDict):
    counter: int
    label: str
    path: list[str]


def node_a(state: PocState) -> dict:
    print("  [node_a] running, incoming state:", state)
    return {"counter": state["counter"] + 1, "path": state["path"] + ["a"]}


def node_b(state: PocState) -> dict:
    print("  [node_b] running, incoming state:", state)
    return {"counter": state["counter"] + 10, "path": state["path"] + ["b"]}


def node_c(state: PocState) -> dict:
    print("  [node_c] running, incoming state:", state)
    return {"counter": state["counter"] + 100, "path": state["path"] + ["c"]}


def node_finalize(state: PocState) -> dict:
    """Deterministic downstream node: doubles the counter it sees.

    Used in experiment 2 to check whether a state change made at a forked
    checkpoint actually propagates into this node's output, which would
    demonstrate real continuation rather than a cached/replayed result.
    """
    print("  [node_finalize] running, incoming state:", state)
    return {"counter": state["counter"] * 2, "path": state["path"] + ["finalize"]}


builder = StateGraph(PocState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)
builder.add_node("node_finalize", node_finalize)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_c")
builder.add_edge("node_c", "node_finalize")
builder.add_edge("node_finalize", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
print("Graph compiled with checkpointer:", type(checkpointer).__name__)


# ---------------------------------------------------------------------------
# 2. Execute the graph with an initial state, on a dedicated thread.
# ---------------------------------------------------------------------------
section("2. Original execution")

original_thread_config = {"configurable": {"thread_id": "poc-original"}}
initial_state: PocState = {"counter": 0, "label": "original", "path": []}

original_result = graph.invoke(initial_state, original_thread_config)
print("Original final state (graph.invoke return value):", original_result)


# ---------------------------------------------------------------------------
# 3. Capture/inspect the checkpoint history produced by the checkpointer.
# ---------------------------------------------------------------------------
section("3. Inspecting checkpoint / state history")

history = list(graph.get_state_history(original_thread_config))
print(f"Number of checkpoints in history: {len(history)}")
print(
    "NOTE: get_state_history typically yields newest-first. Printing in "
    "that order as returned, without assuming direction — verify below."
)

for i, snapshot in enumerate(history):
    print(f"\n--- history[{i}] ---")
    print("values:", snapshot.values)
    print("next:", snapshot.next)
    print("config:", snapshot.config)
    print("metadata:", snapshot.metadata)
    print("created_at:", getattr(snapshot, "created_at", "<no created_at attr>"))
    print("parent_config:", getattr(snapshot, "parent_config", "<no parent_config attr>"))

if not history:
    raise RuntimeError(
        "No checkpoint history returned. Either the checkpointer isn't "
        "wired up correctly, or this LangGraph version exposes history "
        "through a different API — inspect `dir(graph)` for alternatives."
    )


# ---------------------------------------------------------------------------
# 4. Identify a checkpoint in the middle of the execution.
# ---------------------------------------------------------------------------
section("4. Selecting a middle checkpoint")

# history is newest-first in the LangGraph versions this was authored
# against; reverse to walk oldest -> newest and pick the true midpoint.
chronological = list(reversed(history))
mid_index = len(chronological) // 2
middle_snapshot = chronological[mid_index]

print(f"Chosen middle checkpoint index (oldest-first): {mid_index} of {len(chronological)}")
print("Middle checkpoint config:", middle_snapshot.config)
print("Middle checkpoint values (state at this point):", middle_snapshot.values)
print("Middle checkpoint metadata:", middle_snapshot.metadata)
print("Middle checkpoint 'next' (nodes queued to run after this point):", middle_snapshot.next)


# ---------------------------------------------------------------------------
# 5. Test #11: does the framework give us enough info to map a checkpoint
#    to a specific execution step/node?
# ---------------------------------------------------------------------------
section("5. Checkpoint -> step/node mapping test")

for i, snapshot in enumerate(chronological):
    meta = snapshot.metadata or {}
    print(
        f"chronological[{i}]: "
        f"step={meta.get('step', '<no step key>')} "
        f"source={meta.get('source', '<no source key>')} "
        f"writes_keys={list(meta.get('writes') or {}) if isinstance(meta.get('writes'), dict) else meta.get('writes')} "
        f"next={snapshot.next}"
    )

print(
    "\nInterpretation guide (verify against actual printed output above, "
    "do not assume): if 'writes' at each checkpoint contains exactly one "
    "node name, the checkpoint:node relationship is 1:1 for this linear "
    "graph. If any checkpoint's 'writes' contains multiple node names "
    "(possible with parallel branches, not present in this linear PoC), "
    "that would demonstrate many-to-one. If 'step'/'source'/'writes' keys "
    "are absent entirely in this installed version, that demonstrates the "
    "mapping is not directly available via metadata and would need another "
    "mechanism."
)


# ---------------------------------------------------------------------------
# 6. Test #12: can a checkpoint represent state independently of our
#    ExecutionStep model?
# ---------------------------------------------------------------------------
section("6. Checkpoint independence from ExecutionStep-shaped data")

print(
    "middle_snapshot.values keys:", list(middle_snapshot.values.keys())
    if hasattr(middle_snapshot.values, "keys") else type(middle_snapshot.values)
)
print(
    "Observation to make manually: LangGraph's checkpoint 'values' is the "
    "full graph state (here: counter/label/path), NOT a per-step "
    "input/output record. It exists whether or not we ever construct an "
    "ExecutionStep for it. This is evidence for/against treating "
    "Checkpoint.step_id as optional in our common model — confirm by "
    "checking whether checkpoint.values ever contains step-specific I/O "
    "shape (it should not; it's whole-state)."
)


# ---------------------------------------------------------------------------
# 7. Fork from the middle checkpoint by changing exactly ONE state variable.
# ---------------------------------------------------------------------------
section("7. Forking: modifying exactly one variable at the middle checkpoint")

ORIGINAL_COUNTER_AT_MIDPOINT = middle_snapshot.values["counter"]
FORKED_COUNTER_VALUE = ORIGINAL_COUNTER_AT_MIDPOINT + 1000
print(f"Original counter at midpoint: {ORIGINAL_COUNTER_AT_MIDPOINT}")
print(f"Forcing counter to: {FORKED_COUNTER_VALUE} (single-variable change, label/path untouched)")

if not hasattr(graph, "update_state"):
    raise AttributeError(
        "graph.update_state does not exist on this LangGraph version. "
        "Inspect `dir(graph)` for the equivalent (e.g. it may be named "
        "differently, or live on the checkpointer, or require an async "
        "variant `aupdate_state`)."
    )

forked_config = graph.update_state(
    middle_snapshot.config,
    {"counter": FORKED_COUNTER_VALUE},
)
print("forked_config returned by update_state:", forked_config)

forked_snapshot = graph.get_state(forked_config)
print("State at forked_config immediately after update_state:", forked_snapshot.values)
print("Forked snapshot's parent_config:", getattr(forked_snapshot, "parent_config", "<none>"))


# ---------------------------------------------------------------------------
# 8. Verify the ORIGINAL checkpoint/thread is still intact after forking.
# ---------------------------------------------------------------------------
section("8. Verifying original branch survived the fork untouched")

original_history_after_fork = list(graph.get_state_history(original_thread_config))
print(f"Original thread history length after fork: {len(original_history_after_fork)} "
      f"(was {len(history)} before fork)")

original_midpoint_again = graph.get_state(middle_snapshot.config)
print("Re-fetched original middle checkpoint values (should be unchanged):",
      original_midpoint_again.values)
assert original_midpoint_again.values["counter"] == ORIGINAL_COUNTER_AT_MIDPOINT, (
    "ORIGINAL CHECKPOINT WAS MUTATED BY THE FORK — this would be a critical "
    "finding: it means update_state does not create an independent branch "
    "and our adapter's fork_from_checkpoint cannot rely on non-destructive "
    "forking via this call pattern."
)
print("CONFIRMED: original checkpoint state unchanged after fork.")


# ---------------------------------------------------------------------------
# 9. Continue/replay execution from the modified (forked) checkpoint.
# ---------------------------------------------------------------------------
section("9. Continuing execution from the forked checkpoint")

print(
    "Invoking graph.invoke(None, forked_config) — passing None as input "
    "is the documented signal to resume from existing checkpointed state "
    "rather than starting a fresh run. If this instead behaves like a "
    "fresh run from START, the printed node execution order below will "
    "show node_a re-running, which it should NOT if this is true "
    "continuation."
)

forked_result = graph.invoke(None, forked_config)
print("Forked/modified final state:", forked_result)


# ---------------------------------------------------------------------------
# 10 & 13. Print everything requested, and check downstream propagation via
#          the deterministic node_finalize node.
# ---------------------------------------------------------------------------
section("10. Summary: identifiers, states, and paths")

print("Original thread config:", original_thread_config)
print("Middle checkpoint config (fork point):", middle_snapshot.config)
print("Forked config (post update_state):", forked_config)

print("\nChronological original path (from history metadata, oldest->newest):")
for i, snapshot in enumerate(chronological):
    print(f"  step {i}: values={snapshot.values} next={snapshot.next}")

print("\nOriginal final state:", original_result)
print("Original final path list:", original_result.get("path"))

print("\nForked/modified final state:", forked_result)
print("Forked final path list:", forked_result.get("path"))

section("13. Downstream propagation check via node_finalize")

expected_if_true_continuation = None
if "counter" in forked_result:
    print(
        "node_finalize doubles whatever counter it receives. "
        "If continuation is REAL: forked_result['counter'] should reflect "
        "FORKED_COUNTER_VALUE plus whatever node_c added, then doubled by "
        "node_finalize — i.e. it should be traceably derived from "
        f"{FORKED_COUNTER_VALUE}, not from the original {ORIGINAL_COUNTER_AT_MIDPOINT}."
    )
    print("Forked final counter:", forked_result["counter"])
    print(
        "Manually verify: does forked_result['counter'] change if you "
        "change FORKED_COUNTER_VALUE above and re-run? If yes -> real "
        "continuation. If forked_result is identical regardless -> "
        "suspect cached/replayed rather than re-executed."
    )

section("11/12 recap — explicit pass/fail flags to fill in after reading output above")

print(
    "CHECKPOINT_TO_STEP_MAPPING: inspect section 5 output manually — "
    "record whether metadata gave 1:1 / many-to-one / unavailable."
)
print(
    "CHECKPOINT_INDEPENDENT_OF_EXECUTIONSTEP: inspect section 6 output — "
    "record whether checkpoint.values is whole-state (independent) or "
    "step-shaped (coupled)."
)

section("DONE")
print("Read every section above top to bottom before drawing conclusions.")
print("Do not summarize this run's findings without having actually executed it.")
