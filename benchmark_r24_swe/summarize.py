#!/usr/bin/env python3
# ruff: noqa
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
INSTANCE = "sympy__sympy-20590"


def find_report(mode: str) -> tuple[bool | None, str | None, dict[str, Any] | None]:
    model = f"Qwen3-14B-NEXUS-R2.4-{mode}"
    candidates = sorted(
        Path("logs/run_evaluation").glob(
            f"nexus-r24-{mode}/{model}/{INSTANCE}/report.json"
        )
    )
    if not candidates:
        candidates = [
            path
            for path in Path("logs/run_evaluation").rglob(f"{INSTANCE}/report.json")
            if f"R2.4-{mode}" in str(path)
        ]
    if not candidates:
        return None, None, None
    path = candidates[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    resolved = bool(payload.get(INSTANCE, {}).get("resolved", False))
    return resolved, str(path), payload


def main() -> None:
    generation = json.loads(
        (OUT / "generation_metadata.json").read_text(encoding="utf-8")
    )
    modes = {}
    for mode in ["A", "B", "C"]:
        resolved, report_path, report = find_report(mode)
        exit_path = Path("runtime_r24_swe/harness") / f"{mode}.exit_code"
        exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
        modes[mode] = {
            "generation_valid": generation["results"][mode]["valid"],
            "patch_sha256": generation["results"][mode].get("patch_sha256"),
            "changed_paths": generation["results"][mode].get("changed_paths", []),
            "harness_exit_code": exit_code,
            "resolved": resolved,
            "report_path": report_path,
            "report": report,
        }
    infrastructure_complete = all(value["resolved"] is not None for value in modes.values())
    summary = {
        "format": "nexus-r24-swebench-official-harness/1",
        "instance_id": INSTANCE,
        "n": 1,
        "infrastructure_complete": infrastructure_complete,
        "C_decision": generation["C_decision"],
        "modes": modes,
        "boundary": "One-instance official-harness pipeline test; not a leaderboard score.",
    }
    (OUT / "SWE_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# NEXUS R2.4 SWE-bench Verified canonical-edit test",
        "",
        f"Instance: `{INSTANCE}`",
        "",
        "| Mode | Canonical patch valid | Official harness resolved | Harness exit |",
        "|---|---:|---:|---:|",
    ]
    for mode in ["A", "B", "C"]:
        value = modes[mode]
        resolved = "not completed" if value["resolved"] is None else str(value["resolved"])
        lines.append(
            f"| {mode} | {value['generation_valid']} | {resolved} | "
            f"{value['harness_exit_code']} |"
        )
    lines.extend(
        [
            "",
            f"Official harness infrastructure complete: **{infrastructure_complete}**.",
            f"C decision: **{generation['C_decision']}**.",
            "",
            "Every submitted patch was emitted by git after deterministic exact-edit validation; raw model diff text is never submitted.",
        ]
    )
    (OUT / "SWE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
