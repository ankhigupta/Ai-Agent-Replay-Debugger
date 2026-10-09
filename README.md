# AgentDebug

Framework-independent, evidence-backed debugging for AI-agent executions.

## What it is, and what problem it solves

Agents built on LangGraph, plain Python, or anything else fail in ways
that are hard to pin down: a tool call silently errors and nobody
retries, the same call loops three times, a value quietly flips partway
through a run. When something goes wrong, the honest question usually
isn't "what caused this" — it's "what actually happened, and where did
the thing I care about first show up?"

AgentDebug answers that second question deterministically, from captured
execution evidence, with **zero LLM API calls**. It is not a causal
diagnosis tool, not a replay/auto-fix system, and not an observability
platform — see [V1 limitations](#v1-limitations) for the full, explicit
list of what this is not.

## Architecture / workflow

```
Agent (plain Python or LangGraph)
    |
    v
Recorder  -- captures execution as it happens
    |
    v
Trace + Spans  -- structured evidence (framework-independent)
    |
    v
Deterministic checks            Failure-driven localization
(no FailureSignal needed)       (requires a FailureSignal)
    |                                |
    v                                v
            Findings (evidence-backed Claims)
                    |
                    v
        JSONL evidence export (optional)
```

Every stage is a plain, inspectable Python object — a `Trace` is a
dataclass, a `Finding` is a dataclass, nothing is hidden behind an opaque
service call.

## Installation

AgentDebug has **zero required third-party dependencies**. From the
repository root (this repository doesn't publish to PyPI; install
from source):

```bash
pip install .
```

For the optional LangGraph integration:

```bash
pip install ".[langgraph]"
```

For development (editable install, so edits to `agentdebug/` take effect
immediately):

```bash
pip install -e ".[langgraph]"
```

Requires Python 3.10+.

## Quickstart: a minimal plain-Python example

```python
from agentdebug import Recorder, SpanKind, SwallowedToolErrorCheck

recorder = Recorder()
trace = recorder.start_trace()

with recorder.span("get_order", kind=SpanKind.TOOL, input={"order_id": "ORD-42"}) as span:
    span.set_output({"refund_eligible": True})

recorder.end_trace()

finding = SwallowedToolErrorCheck().run(trace)
print(finding.status.value)  # "NO_FINDING" -- nothing went wrong here
```

A complete, runnable version with five deterministic scenarios (clean,
swallowed tool error, repeated calls, a localization failure, and an
incomplete trace) lives at
[`examples/plain_python_agent.py`](examples/plain_python_agent.py):

```bash
python -m examples.plain_python_agent
```

## FailureSignal and localization

A `FailureSignal` is **external evidence** — what a test, eval, or user
report says about the outcome. AgentDebug never invents one.

```python
from agentdebug import FailureSignal, ExternalStatus, FirstObservedOccurrenceAnalyzer, LocalizationTarget

failure_signal = FailureSignal(
    source="external_evaluator", verdict=ExternalStatus.FAILED,
    expected=True, actual=False,
)

target = LocalizationTarget(value=False, field_path="refund_eligible")
finding = FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)
```

**Localization means locating the first observed occurrence of a value —
not proving a root cause.** The resulting `Finding` says "`False` was
first observed at span `update_state`"; it never says that span caused
anything, that the value was wrong, or who/what is responsible.

## Deterministic checks

Three checks ship in V1, each detecting one specific, observable trace
pattern — no `FailureSignal` required:

```python
from agentdebug import SwallowedToolErrorCheck, RepeatedIdenticalCallsCheck, TraceIncompletenessCheck, ALL_CHECKS

findings = [check.run(trace) for check in ALL_CHECKS]
```

| Check | Detects |
|---|---|
| `SwallowedToolErrorCheck` | A `TOOL` span with `ERROR` status and no matching retry/fallback sibling |
| `RepeatedIdenticalCallsCheck` | 3+ consecutive `TOOL` spans with identical canonicalized operation + input |
| `TraceIncompletenessCheck` | A span whose `parent_id` references a span that doesn't exist in the trace |

**`NO_FINDING` does not mean the execution was correct** — it means this
specific check's defined pattern wasn't observed. A trace can be
`NO_FINDING` on every check and still be externally wrong (see
`self_consistent_wrong_*` cases in `benchmark/cases.py` for exactly this).

**`INCONCLUSIVE` means the check applies but the available evidence was
insufficient to decide** (as opposed to `NOT_APPLICABLE`, which means the
integration can't expose the required evidence at all). Pass
`available_capabilities=...` to simulate or honestly report a capability
gap; see `agentdebug.core.Capability`.

To write a custom check, subclass `TraceCheck` and provide a
`CheckDefinition`; to write a custom localization analyzer, subclass
`LocalizationAnalyzer` — both are exported from the top-level package.

## Evidence export (JSONL)

```python
from agentdebug import JsonlEvidenceSink

with JsonlEvidenceSink("runs.jsonl") as sink:
    sink.write_trace(trace)
```

- **One Trace is one JSONL record.** `findings` and `failure_signals`
  already belong to `Trace`, so they're written automatically through the
  same record — there's no separate `findings.jsonl`/`spans.jsonl`.
- **Append-only.** The file is opened in append mode; writing never
  truncates or overwrites what a previous run wrote.
- **No new serialization format.** Each line is exactly
  `agentdebug.to_json(trace)` — the existing `Trace` schema
  (`schema_version`, currently `"1.0.0"`) is the only schema here. A line
  reconstructs with the same `Trace.from_dict()` used everywhere else in
  AgentDebug.
- **An evidence artifact, not a database.** No query layer, no indexing,
  no reader abstraction — just durable, independently-parseable JSON
  lines meant for later evaluation/benchmarking tooling to read with
  ordinary file I/O.
- **V1 does not guarantee multi-process concurrency.** A single
  `JsonlEvidenceSink` is thread-safe within one process (a lock guards
  each write); there is no cross-process file locking or atomicity
  guarantee if two separate processes append to the same path at once.

See `examples/evidence_export_demo.py` for a runnable example (writes to
a temporary file, not a path committed to this repo) and
`tests/evidence/test_integration.py` for the full
Recorder → Trace → FailureSignal → localization → sink → reload proof.

## Optional: LangGraph integration

Install with `pip install ".[langgraph]"`. A complete, runnable example
lives at [`examples/langgraph_agent.py`](examples/langgraph_agent.py):

```bash
python -m examples.langgraph_agent
```

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from agentdebug.integrations.langgraph import LangGraphAdapter
from agentdebug import ALL_CHECKS, FirstObservedOccurrenceAnalyzer, LocalizationTarget, FailureSignal, ExternalStatus

# 1. Create a LangGraph graph (span_kind metadata is optional but lets
#    the adapter honestly label a node TOOL/STATE/LLM instead of the
#    default CUSTOM -- LangGraph's generic node-update evidence carries
#    no semantic category on its own).
def get_order(state):
    return {"refund_eligible": state["refund_eligible"]}

builder = StateGraph(dict)
builder.add_node("get_order", get_order, metadata={"span_kind": "TOOL"})
builder.add_edge(START, "get_order")
builder.add_edge("get_order", END)
graph = builder.compile(checkpointer=MemorySaver())

# 2. Wrap it with AgentDebug.
adapter = LangGraphAdapter(graph)

# 3 & 4. Execute it and obtain the AgentDebug Trace.
trace = adapter.capture({"refund_eligible": True})

# 5. Run the existing deterministic checks, unmodified.
findings = [check.run(trace) for check in ALL_CHECKS]

# 6. Optionally supply a FailureSignal (external evidence -- the adapter
#    never invents one).
failure_signal = FailureSignal(
    source="external_evaluator", verdict=ExternalStatus.FAILED,
    expected=True, actual=False,
)

# 7. Run localization against the converted Trace.
target = LocalizationTarget(value=False, field_path="refund_eligible")
localization_finding = FirstObservedOccurrenceAnalyzer().run(trace, failure_signal, target=target)
```

**LangGraph checkpoints are not treated as execution steps.** A
`StateSnapshot` from `get_state_history()` is graph state/history
evidence; an AgentDebug `Span` is built only from evidence that actually
corresponds to a node executing (`stream(..., stream_mode="updates")`).
This distinction matters concretely for fan-out: two nodes racing into one
join produce two independent Spans sharing one checkpoint, never one
Span fabricated for "the node" that ran.

**Capability limitations are honest, not hidden.** `adapter.capabilities`
only advertises `SPAN_IO`/`TOOL_STATUS`/`TOOL_CALLS` when the graph was
compiled with a checkpointer (that's what lets the adapter derive a node's
input and attribute a failure to a node at all); without one, only
`PRE_RUN_CONTEXT` is advertised. Pass `adapter.capabilities` as
`available_capabilities` to a check or to `FirstObservedOccurrenceAnalyzer`
to get `NOT_APPLICABLE`/`INCONCLUSIVE` instead of a false negative when
evidence is genuinely missing — see
`tests/integrations/langgraph_integration/test_adapter.py`'s
`CapabilityTests` for both cases exercised end to end.

**`import agentdebug` never requires LangGraph.** The integration lives in
`agentdebug.integrations.langgraph`, imported only when you ask for it.

## Benchmark & evaluation

A separate, deterministic evaluation harness — evaluation tooling, kept
out of `agentdebug/` (SDK core). Run it from the repository root:

```bash
python -m benchmark
```

This runs the full 29-case corpus and prints a report; pass `--output-dir`
to also write `evidence.jsonl` (via the same `JsonlEvidenceSink` above) and
`report.json`. Output defaults to `.benchmark_output/` (gitignored — never
commit generated benchmark artifacts).

**The full benchmark requires the LangGraph extra.** Two of the 29 cases
capture a real LangGraph execution, so install with
`pip install -e ".[langgraph]"` first. Without LangGraph,
`python -m benchmark` exits with an error that says so — it never quietly
runs a smaller corpus. To run only the 27 cases that need no LangGraph,
ask for that explicitly:

```bash
python -m benchmark --core-only
```

The report is then labelled as a subset (27 of 29) and is not the full
benchmark. This only affects the benchmark; the `agentdebug` SDK itself
never requires LangGraph.

**Ground truth is separate from AgentDebug evidence, on purpose.** Each
benchmark case plants a known answer (e.g. "the deviation's reference span
is `update_state`") as metadata on the case itself. `benchmark/runner.py`
never passes that metadata into a `check.run(...)` or
`FirstObservedOccurrenceAnalyzer.run(...)` call — it only ever passes a
`Trace`, a `FailureSignal` built the way a real external evaluator would
build one, and (for simulating capability gaps) `available_capabilities`.
Metrics are computed by comparing what AgentDebug actually returned
against the planted ground truth *afterward*. This separation is tested
directly (`tests/benchmarking/test_cases.py`'s `GroundTruthIsolationTests`
and `test_runner.py`'s ground-truth-mutation test), not just asserted in
prose.

**Localization labels** (`ORIGIN`, `FIRST_OBSERVABLE_DEVIATION`,
`SYMPTOM`, `TERMINAL`, `NONE`) are benchmark-only ground truth describing
what kind of point a case's reference span represents in the case
author's own understanding — never a `ClaimType`, never seen by
`agentdebug`. A benchmark label of `ORIGIN` mapping to AgentDebug's
`UNIQUE_CORRECT` observation is reported as a *relationship*, not
collapsed into a single pass/fail score.

**Abstention** means AgentDebug returned `NOT_APPLICABLE` or
`INCONCLUSIVE` rather than guessing. *Abstention precision* asks: among
cases where it abstained, how often was that appropriate (evidence
actually was missing)? *False-abstention rate* asks: among cases where
evidence genuinely was available, how often did it abstain anyway? Both
are `None` (not `0.0`) when their denominator is zero.

**Self-consistent wrong cases** are the sharpest illustration of this
package's whole point: a trace can be entirely internally consistent (no
tool errors, no repeats, no dangling references) while an external
evaluator still says the outcome was wrong. AgentDebug's trace checks
correctly find nothing (there is nothing observable to flag), and
localization correctly reports *where* the disputed value was first
observed — never that the value was wrong, and never why it came out that
way. See `self_consistent_wrong_refund`/`self_consistent_wrong_category`
in `benchmark/cases.py`.

**Why this doesn't prove causality:** localization answers "where was
value X first observed", which is a different question from "what caused
X to be wrong" — the corpus's self-consistent-wrong and held-out cases
exist specifically to keep that distinction from blurring.

**Why 29 cases don't establish generalization:** this is a deliberately
small, hand-constructed corpus meant to catch regressions and illustrate
known behavior — not a statistically powered benchmark. "Held-out" here
means only "not one of the cases a check's demonstration example was
originally built from," not a held-out split in the machine-learning
sense; no claim of broader generalization is made or should be inferred
from it.

## Running the tests

From the repository root, with the dev environment set up (`pip install -e
".[langgraph]"` recommended so every test runs instead of skipping the
LangGraph-dependent ones):

```bash
python -m unittest discover -s tests
```

Without the LangGraph extra, the LangGraph-dependent tests, and the two
tests that assert the *full* 29-case corpus's size and composition, are
skipped with an explicit reason (they are not weakened or removed); the
rest, including tests of the `--core-only` subset and of the
missing-LangGraph error, still run.

Or a single area, e.g. just the core data model:

```bash
python -m unittest discover -s tests/core
```

## Project layout

```
.
  agentdebug/        the installable SDK (this is what `pip install .` ships)
    core/            Trace, Span, Finding, Claim, FailureSignal, serialization
    recording/       Recorder (captures a live execution into a Trace)
    analysis/        deterministic checks + failure-driven localization
    evidence/        JSONL evidence export
    integrations/    optional framework integrations (LangGraph today)
  examples/          runnable demonstrations (not part of the SDK distribution)
  benchmark/         deterministic evaluation harness (not part of the SDK distribution)
  tests/             the test suite (not part of the SDK distribution)
  pyproject.toml     packaging config for the `agentdebug` distribution only
  README.md, LICENSE
```

## V1 limitations

Please read this section before assuming AgentDebug does more than it
does:

- **V1 makes zero LLM API calls.** Every check and the localization
  analyzer are fully deterministic.
- **Findings are evidence-backed.** A `Finding`'s `Claim`s only ever
  contain facts directly observed in the `Trace` (a span's recorded
  input/output, a context value, a structural reference) — never inferred
  or guessed content.
- **`NO_FINDING` does not mean the execution was correct.** It means that
  specific check's defined pattern wasn't observed in the available
  evidence. A trace can pass every check and still be externally wrong.
- **`INCONCLUSIVE` means required evidence was unavailable or
  insufficient** to decide — not a confident negative result.
  `NOT_APPLICABLE` means the integration can't expose the required
  evidence at all. Neither is ever silently reinterpreted as success.
- **AgentDebug does not claim causality.** No `Claim` ever says a span
  caused a failure, that a value was "wrong," or who/what is responsible.
- **Localization means locating the first observed occurrence of a
  value/deviation — not proving a root cause.** "First observed at span
  X" is an observation about captured evidence, not a diagnosis.
- **V1 does not provide:** replay, state forking, auto-fixing, LLM-based
  diagnosis, a dashboard, or a database platform. Evidence export is a
  flat, append-only JSONL file, not a query engine.
- **The benchmark corpus is small and hand-constructed** (29 cases); it
  catches regressions and illustrates known behavior, and does not
  establish statistical generalization.
- **LangGraph subgraph nesting isn't detected** by the LangGraph
  integration yet — all spans from a flat graph are correctly
  parent-less; a graph that nests subgraphs isn't specially handled.
