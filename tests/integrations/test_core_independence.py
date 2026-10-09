"""Verifies the Phase 5B architectural constraint: AgentDebug core must
not know LangGraph exists.

`sys.modules['langgraph'] = None` makes Python raise ImportError the
instant anything tries `import langgraph` (the standard trick for
simulating "package not installed" without actually uninstalling
anything), run in a subprocess so it can't contaminate this test
process's already-imported modules.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_SCRIPT = """
import sys
sys.modules["langgraph"] = None
import agentdebug
import agentdebug.core
import agentdebug.recording
import agentdebug.analysis
import agentdebug.analysis.checks
import agentdebug.analysis.localization
print("OK")
"""


class CoreIndependenceTests(unittest.TestCase):
    def test_core_recording_and_analysis_import_without_langgraph(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)

    def test_importing_agentdebug_does_not_import_langgraph_integration(self) -> None:
        script = (
            "import sys\n"
            "import agentdebug\n"
            "assert 'agentdebug.integrations' not in sys.modules, "
            "'import agentdebug must not pull in the LangGraph integration'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
