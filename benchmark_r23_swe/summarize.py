#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
INSTANCE = "sympy__sympy-20590"


def find_report(mode: str) -> tuple[bool | None, str | None, dict[str, Any] | None]:
    model = f"Qwen3-14B-NEXUS-R2.3-{mode}"
    candidates = sorted(Path("logs/run_evaluation").glob(f"nexus-r23-{mode}/{model}/{INSTANCE}/report.json"))
    if not candidates:
        candidates = sorted(Path("logs/run_evaluation").rglob(f"{INSTANCE}/report.json"))
        candidates = [path for path in candidates if f"R2.3-{mode}" in str(path)]
    if not candidates:
        return None, None, None
    path = candidates[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    resolved = bool(payload.get(INSTANCE, {}).get("resolved", False))
    return resolved, str(path), payload


def main() -> None:
    modes = {}
    for mode in ["A", "B", "C"]:
        resolved, report_path, report = find_report(mode)
        generation = json.loads((OUT / "generation_metadata.json").read_text(encoding="utf-8"))
        modes[mode] = {
            "resolved": resolved,
            "report_path": report_path,
            "report": report,
            "git_apply_check": generation["apply_checks"][mode],
        }
    summary = {
        "format": "nexus-r23-swebench-official-harness/1",
        "instance_id": INSTANCE,
        "n": 1,
        "modes": modes,
        "boundary": "One-instance official-harness smoke test; not a leaderboard score.",
    }
    (OUT / "SWE_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# NEXUS R2.3 SWE-bench Verified smoke test",
        "",
        f"Instance: `{INSTANCE}`",
        "",
        "| Mode | git apply check | Official harness resolved |",
        "|---|---:|---:|",
    ]
    for mode in ["A", "B", "C"]:
        value = modes[mode]
        resolved = "not completed" if value["resolved"] is None else str(value["resolved"])
        lines.append(f"| {mode} | {value['git_apply_check']['valid']} | {resolved} |")
    lines.extend(
        [
            "",
            "This validates the complete issue-to-patch-to-Docker-test pipeline on one official instance. It is not statistically meaningful as a SWE-bench score.",
        ]
    )
    (OUT / "SWE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
