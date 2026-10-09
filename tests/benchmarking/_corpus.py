"""Shared corpus access for the benchmark tests.

The full benchmark corpus requires the optional `langgraph` extra (see
`benchmark.cases.build_corpus`). `CORPUS` is the full corpus when LangGraph
is installed; otherwise it is the explicitly requested core-only subset, so
per-case behavioural tests still run. Assertions that depend on the full
corpus's size/composition use `requires_langgraph`, which skips them with an
explicit reason instead of loosening the expected numbers.
"""

from __future__ import annotations

import unittest

from benchmark import cases

LANGGRAPH_AVAILABLE: bool = cases._LANGGRAPH_AVAILABLE

CORPUS = cases.build_corpus(include_langgraph_cases=LANGGRAPH_AVAILABLE)

requires_langgraph = unittest.skipUnless(
    LANGGRAPH_AVAILABLE,
    "the full 29-case benchmark corpus requires the 'langgraph' extra: pip install '.[langgraph]'",
)
