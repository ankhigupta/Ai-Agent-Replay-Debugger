"""Phase 7: deterministic benchmark/evaluation harness for AgentDebug.

    corpus (BenchmarkCase, each carrying benchmark-only ground truth)
        -> runner (executes real AgentDebug paths -- ground truth never
           reaches an analyzer call)
        -> metrics (per-check precision/recall, localization distribution,
           abstention precision / false-abstention rate, capability
           coverage, overhead)
        -> report (human-readable text + machine-readable JSON)

This package is evaluation TOOLING, kept deliberately separate from
`agentdebug/` (SDK core). It answers one question: how reliably does
AgentDebug detect observable failure patterns, localize failure-related
evidence, and abstain when evidence is insufficient — never whether it can
prove causality (it can't, and this package doesn't pretend otherwise),
and never a claim of statistical generalization from a necessarily small,
hand-constructed corpus.
"""

from __future__ import annotations
