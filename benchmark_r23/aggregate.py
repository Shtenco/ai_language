#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "collected"
OUTPUT = ROOT / "aggregate"
SEED = 230804


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_delta(left: list[int], right: list[int], seed: int, samples: int = 30000) -> list[float]:
    rng = random.Random(seed)
    n = len(left)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(right[i] - left[i] for i in indices) / n)
    values.sort()
    return [round(values[int(0.025 * samples)], 4), round(values[int(0.975 * samples) - 1], 4)]


def pair_stats(rows: list[dict[str, Any]], left: str, right: str, seed: int) -> dict[str, Any]:
    left_scores = [int(row[left]["correct"]) for row in rows]
    right_scores = [int(row[right]["correct"]) for row in rows]
    wins = sum(r > l for l, r in zip(left_scores, right_scores))
    losses = sum(r < l for l, r in zip(left_scores, right_scores))
    return {
        "delta_accuracy": round(sum(right_scores) / len(rows) - sum(left_scores) / len(rows), 4),
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - wins - losses,
        "exact_sign_p": round(exact_sign_p(wins, losses), 6),
        "bootstrap_95_ci": bootstrap_delta(left_scores, right_scores, seed),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(INPUT_ROOT.rglob("summary.json")):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    for responses_path in sorted(INPUT_ROOT.rglob("responses.jsonl")):
        with responses_path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if len(summaries) != 3 or not rows:
        raise RuntimeError(f"expected 3 summaries and responses, got summaries={len(summaries)} rows={len(rows)}")

    modes = {}
    for mode in ["A", "B", "C"]:
        correct = sum(int(row[mode]["correct"]) for row in rows)
        modes[mode] = {
            "correct": correct,
            "n": len(rows),
            "accuracy": round(correct / len(rows), 4),
            "invalid_answers": sum(not row[mode]["answer"] for row in rows),
        }
    result = {
        "format": "nexus-r23-external-aggregate/1",
        "n": len(rows),
        "benchmarks": {summary["benchmark"]: summary for summary in summaries},
        "modes": modes,
        "pairs": {
            "B_minus_A": pair_stats(rows, "A", "B", SEED + 201),
            "C_minus_A": pair_stats(rows, "A", "C", SEED + 202),
            "C_minus_B": pair_stats(rows, "B", "C", SEED + 203),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# NEXUS R2.3 external intelligence-gain report",
        "",
        f"Total external tasks: **{len(rows)}**.",
        "",
        "| Mode | Correct | Accuracy | Invalid answers |",
        "|---|---:|---:|---:|",
    ]
    for mode in ["A", "B", "C"]:
        value = modes[mode]
        lines.append(
            f"| {mode} | {value['correct']}/{value['n']} | {value['accuracy']:.1%} | {value['invalid_answers']} |"
        )
    lines.extend(["", "## Per benchmark", "", "| Benchmark | A | B | C | C-A |", "|---|---:|---:|---:|---:|"])
    for summary in sorted(summaries, key=lambda item: item["benchmark"]):
        a = summary["modes"]["A"]["accuracy"]
        b = summary["modes"]["B"]["accuracy"]
        c = summary["modes"]["C"]["accuracy"]
        lines.append(f"| {summary['benchmark']} | {a:.1%} | {b:.1%} | {c:.1%} | {c-a:+.1%} |")
    lines.extend(["", "## Paired inference", ""])
    for name, value in result["pairs"].items():
        lines.append(
            f"- **{name}:** {value['delta_accuracy']:+.1%}; wins/losses/ties "
            f"{value['wins']}/{value['losses']}/{value['ties']}; "
            f"95% bootstrap CI {value['bootstrap_95_ci']}; exact sign p={value['exact_sign_p']}."
        )
    lines.extend(
        [
            "",
            "## Honest conclusion rule",
            "",
            "The previous +37.5 percentage-point gain is considered replicated only if C-A is positive, its paired 95% bootstrap interval excludes zero, and the gain is not confined to one benchmark. Mode C spends three inference calls versus one in A/B, so this measures the full system, not parameter efficiency.",
            "",
            "This 45-task pilot is independent and externally sourced, but it is not a replacement for official full-dataset leaderboard submissions.",
        ]
    )
    (OUTPUT / "NEXUS_R2.3_EXTERNAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
