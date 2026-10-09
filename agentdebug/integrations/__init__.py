"""Framework integrations for AgentDebug.

Each integration lives in its own subpackage and is the ONLY place that
framework's imports are allowed. `agentdebug.core`, `agentdebug.recording`,
and `agentdebug.analysis` must never import from here, or from any
framework — the dependency direction is strictly:

    integration (e.g. agentdebug.integrations.langgraph) -> agentdebug core

never the reverse. `import agentdebug` does not import this package or any
integration, so the core SDK stays usable without any framework installed.
"""

from __future__ import annotations
