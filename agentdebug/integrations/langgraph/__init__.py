"""LangGraph integration: converts observable LangGraph execution evidence
into the framework-independent AgentDebug Trace/Span model.

    LangGraph graph -> LangGraphAdapter.capture() -> AgentDebug Trace
        -> existing Phase 3 checks + Phase 4 localization (unchanged)

Importing this subpackage requires `langgraph` to be installed; importing
`agentdebug` itself does not (this subpackage is never imported from
`agentdebug/__init__.py`). See `adapter.py` for the full design notes,
including why LangGraph checkpoints are NOT treated as execution steps.
"""

from __future__ import annotations

from agentdebug.integrations.langgraph.adapter import DEFAULT_NODE_SPAN_KIND_KEY, LangGraphAdapter

__all__ = ["DEFAULT_NODE_SPAN_KIND_KEY", "LangGraphAdapter"]
