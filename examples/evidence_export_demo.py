"""Evidence export demonstration.

    Recorder (reused from examples.plain_python_agent) -> Trace
        -> JsonlEvidenceSink -> JSONL file -> Trace.from_dict() -> same evidence

Reuses `examples.plain_python_agent.build_trace` rather than duplicating
agent logic -- this script's only job is to show the export/reload step on
top of an already-proven Trace.

Writes to a temporary directory that is removed when the script exits, not
a path committed to this repo.

Run as a module from the repository root:

    python -m examples.evidence_export_demo
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentdebug.core.trace import Trace
from agentdebug.evidence import JsonlEvidenceSink
from examples.plain_python_agent import Scenario, build_trace


def main() -> None:
    trace1 = build_trace(Scenario.CLEAN)
    trace2 = build_trace(Scenario.SWALLOWED_TOOL_ERROR)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs.jsonl"

        with JsonlEvidenceSink(path) as sink:
            sink.write_trace(trace1)
            sink.write_trace(trace2)

        lines = path.read_text(encoding="utf-8").splitlines()
        print(f"Wrote {len(lines)} JSONL record(s) to {path}")

        for i, line in enumerate(lines):
            record = json.loads(line)
            restored = Trace.from_dict(record)
            print(
                f"  record {i}: id={restored.id!r} schema_version={record['schema_version']!r} "
                f"spans={len(restored.spans)}"
            )
            assert restored.to_dict() == json.loads(line)  # proves round-trip fidelity

        # tmp directory (and the file in it) is removed here on exit --
        # nothing is left behind in the repo.


if __name__ == "__main__":
    main()
