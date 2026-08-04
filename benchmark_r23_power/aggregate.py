#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
RESULTS_ROOT = ROOT / "downloaded"
OUT = ROOT / "aggregate"


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_delta(
    left: list[int],
    right: list[int],
    seed: int,
    samples: int = 30000,
) -> list[float]:
    rng = random.Random(seed)
    n = len(left)
    values: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(right[i] - left[i] for i in indices) / n)
    values.sort()
    lower = values[int(samples * 0.025)]
    upper = values[int(samples * 0.975) - 1]
    return [round(lower, 6), round(upper, 6)]


def pair_stats(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    seed: int,
) -> dict[str, Any]:
    left_scores = [int(row[left]["correct"]) for row in rows]
    right_scores = [int(row[right]["correct"]) for row in rows]
    wins = sum(r > l for l, r in zip(left_scores, right_scores))
    losses = sum(r < l for l, r in zip(left_scores, right_scores))
    delta = sum(right_scores) / len(rows) - sum(left_scores) / len(rows)
    return {
        "left": left,
        "right": right,
        "delta_accuracy": round(delta, 6),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "exact_sign_p": round(exact_sign_p(wins, losses), 8),
        "paired_bootstrap_95_ci": bootstrap_delta(left_scores, right_scores, seed),
    }


def mode_summary(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(row[mode]["correct"]) for row in rows)
    valid = sum(int(row[mode]["valid"]) for row in rows)
    calls = sum(int(row[mode]["llm_calls"]) for row in rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 6),
        "valid": valid,
        "valid_rate": round(valid / total, 6),
        "invalid": total - valid,
        "llm_calls": calls,
        "calls_per_task": round(calls / total, 4),
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    seed = int(manifest["seed"])
    expected_total = sum(
        int(spec["sample_n"]) for spec in manifest["benchmarks"].values()
    )

    response_files = sorted(RESULTS_ROOT.rglob("responses.jsonl"))
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for path in response_files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["benchmark"], row["stable_id"])
                if key in rows_by_key:
                    raise RuntimeError(f"duplicate result row: {key}")
                rows_by_key[key] = row
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    if len(rows) != expected_total:
        raise RuntimeError(
            f"incomplete benchmark: expected {expected_total} rows, found {len(rows)}"
        )

    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
        by_category[f"{row['benchmark']}::{row['category']}"] .append(row)

    modes = {mode: mode_summary(rows, mode) for mode in ["A", "B", "C"]}
    primary = pair_stats(rows, "B", "C", seed + 1)
    secondary = pair_stats(rows, "A", "C", seed + 2)
    ci_lower = primary["paired_bootstrap_95_ci"][0]
    if primary["delta_accuracy"] < 0:
        claim = "matched_compute_regression"
    elif (
        primary["delta_accuracy"] > 0
        and ci_lower > 0
        and primary["exact_sign_p"] < 0.05
    ):
        claim = "statistically_supported_matched_compute_gain"
    else:
        claim = "matched_compute_gain_not_proven"

    proposal_correct = sum(int(row["C"]["proposal_correct"]) for row in rows)
    final_correct = sum(int(row["C"]["correct"]) for row in rows)
    beneficial = sum(
        int(not row["C"]["proposal_correct"] and row["C"]["correct"])
        for row in rows
    )
    harmful = sum(
        int(row["C"]["proposal_correct"] and not row["C"]["correct"])
        for row in rows
    )
    changed = sum(
        int(row["C"]["proposal_answer"] != row["C"]["answer"])
        for row in rows
    )
    tool_activations = sum(
        int(row["C"]["resolution"]["telemetry"]["typed_exact_arithmetic"])
        for row in rows
    )
    tool_overrides = sum(
        int(row["C"]["resolution"]["telemetry"]["tool_override"])
        for row in rows
    )
    contradictions = sum(
        int(row["C"]["resolution"]["blackboard"]["answer_contradiction"])
        for row in rows
    )

    per_benchmark: dict[str, Any] = {}
    for benchmark, subset in sorted(by_benchmark.items()):
        per_benchmark[benchmark] = {
            "A": mode_summary(subset, "A"),
            "B": mode_summary(subset, "B"),
            "C": mode_summary(subset, "C"),
            "C_vs_B": pair_stats(subset, "B", "C", seed + len(benchmark)),
            "C_vs_A": pair_stats(subset, "A", "C", seed + 100 + len(benchmark)),
        }

    per_category: dict[str, Any] = {}
    for category, subset in sorted(by_category.items()):
        per_category[category] = {
            mode: mode_summary(subset, mode) for mode in ["A", "B", "C"]
        }

    summary = {
        "format": "nexus-r23-power-aggregate/1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runner_sha256": hash_file(ROOT / "run_power.py"),
        "aggregate_sha256": hash_file(Path(__file__)),
        "total_tasks": len(rows),
        "expected_tasks": expected_total,
        "complete": len(rows) == expected_total,
        "claim": claim,
        "modes": modes,
        "primary_C_vs_B": primary,
        "secondary_C_vs_A": secondary,
        "nexus_runtime_audit": {
            "proposal_correct": proposal_correct,
            "final_correct": final_correct,
            "beneficial_corrections": beneficial,
            "harmful_corrections": harmful,
            "answer_changes": changed,
            "typed_exact_tool_activations": tool_activations,
            "tool_overrides": tool_overrides,
            "answer_contradictions_detected": contradictions,
        },
        "per_benchmark": per_benchmark,
        "per_category": per_category,
        "selected_task_fingerprints": [
            {
                "benchmark": row["benchmark"],
                "stable_id": row["stable_id"],
                "question_sha256": row["question_sha256"],
            }
            for row in rows
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ALL_RESPONSES.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "NEXUS_R23_POWER_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# NEXUS R2.3 matched-compute power benchmark",
        "",
        f"- Tasks: **{len(rows)}/{expected_total}**",
        f"- Claim gate: **{claim}**",
        f"- Model SHA-256: `{manifest['model']['sha256']}`",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`",
        "",
        "## Overall",
        "",
        "| Mode | Correct | Accuracy | Valid rate | LLM calls/task |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ["A", "B", "C"]:
        item = modes[mode]
        lines.append(
            f"| {mode} | {item['correct']}/{item['total']} | "
            f"{item['accuracy']:.2%} | {item['valid_rate']:.2%} | "
            f"{item['calls_per_task']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Primary matched-compute comparison: C vs B",
            "",
            f"- Delta: **{primary['delta_accuracy']:+.2%}**",
            f"- Wins / losses / ties: **{primary['wins']} / {primary['losses']} / {primary['ties']}**",
            f"- Exact two-sided p: **{primary['exact_sign_p']}**",
            f"- Paired bootstrap 95% CI: **[{primary['paired_bootstrap_95_ci'][0]:+.2%}, {primary['paired_bootstrap_95_ci'][1]:+.2%}]**",
            "",
            "## Secondary comparison: C vs A",
            "",
            f"- Delta: **{secondary['delta_accuracy']:+.2%}**",
            f"- Wins / losses / ties: **{secondary['wins']} / {secondary['losses']} / {secondary['ties']}**",
            f"- Exact two-sided p: **{secondary['exact_sign_p']}**",
            f"- Paired bootstrap 95% CI: **[{secondary['paired_bootstrap_95_ci'][0]:+.2%}, {secondary['paired_bootstrap_95_ci'][1]:+.2%}]**",
            "",
            "## NEXUS runtime audit",
            "",
            f"- Proposal correct: **{proposal_correct}/{len(rows)}**",
            f"- Final correct: **{final_correct}/{len(rows)}**",
            f"- Beneficial corrections: **{beneficial}**",
            f"- Harmful corrections: **{harmful}**",
            f"- Answer changes: **{changed}**",
            f"- Typed exact activations: **{tool_activations}**",
            f"- Tool overrides: **{tool_overrides}**",
            f"- Contradictions detected: **{contradictions}**",
            "",
            "## Per benchmark",
            "",
            "| Benchmark | A | B | C | C-B delta | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for benchmark, item in per_benchmark.items():
        lines.append(
            f"| {benchmark} | {item['A']['accuracy']:.2%} | "
            f"{item['B']['accuracy']:.2%} | {item['C']['accuracy']:.2%} | "
            f"{item['C_vs_B']['delta_accuracy']:+.2%} | "
            f"{item['C_vs_B']['exact_sign_p']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The primary scientific comparison is C versus B because both use exactly two calls to the same frozen model. A is a lower-compute reference. This 90-item run is independently sourced and preregistered, but it is not an official full-dataset leaderboard submission. A positive raw delta is not called an intelligence gain unless the preregistered confidence-interval and exact-test gate is passed.",
        ]
    )
    (OUT / "NEXUS_R23_POWER_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
