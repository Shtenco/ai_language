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
SEED = 24080417
EXPECTED = {"mmlu_pro", "gpqa", "arc_challenge"}


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_delta(left: list[int], right: list[int], seed: int, samples: int = 50000) -> list[float]:
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
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(INPUT_ROOT.rglob("summary.json"))
    ]
    by_benchmark = {summary["benchmark"]: summary for summary in summaries}
    missing = sorted(EXPECTED - set(by_benchmark))
    if missing:
        raise RuntimeError(f"missing benchmark summaries: {missing}")

    rows: list[dict[str, Any]] = []
    for path in sorted(INPUT_ROOT.rglob("responses.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    completed = sorted(
        name for name, summary in by_benchmark.items() if summary.get("status") == "COMPLETED"
    )
    blocked = {
        name: summary for name, summary in by_benchmark.items() if summary.get("status") != "COMPLETED"
    }
    if not rows:
        raise RuntimeError("no completed benchmark responses were collected")

    modes = {}
    for mode in ["A", "B", "C"]:
        correct = sum(int(row[mode]["correct"]) for row in rows)
        modes[mode] = {
            "correct": correct,
            "n": len(rows),
            "accuracy": round(correct / len(rows), 4),
            "invalid_answers": sum(not row[mode]["answer"] for row in rows),
        }
    pairs = {
        "B_minus_A": pair_stats(rows, "A", "B", SEED + 201),
        "C_minus_A": pair_stats(rows, "A", "C", SEED + 202),
        "C_minus_B": pair_stats(rows, "B", "C", SEED + 203),
    }
    corrections = sum((not row["B"]["correct"]) and row["C"]["correct"] for row in rows)
    regressions = sum(row["B"]["correct"] and (not row["C"]["correct"]) for row in rows)
    non_negative_everywhere = all(
        by_benchmark[name]["pairs"]["C_minus_A"]["delta_accuracy"] >= 0 for name in completed
    )
    all_completed = set(completed) == EXPECTED
    ci_excludes_zero = pairs["C_minus_A"]["bootstrap_95_ci"][0] > 0
    replicated = all_completed and ci_excludes_zero and non_negative_everywhere and corrections > regressions

    result = {
        "format": "nexus-r24-external-aggregate/1",
        "n": len(rows),
        "completed_benchmarks": completed,
        "blocked_benchmarks": blocked,
        "benchmarks": by_benchmark,
        "modes": modes,
        "pairs": pairs,
        "guarded_consensus": {
            "B_to_C_corrections": corrections,
            "B_to_C_regressions": regressions,
            "net": corrections - regressions,
        },
        "claim_gate": {
            "all_three_completed": all_completed,
            "aggregate_C_minus_A_ci_excludes_zero": ci_excludes_zero,
            "C_minus_A_non_negative_on_every_completed_benchmark": non_negative_everywhere,
            "more_corrections_than_regressions": corrections > regressions,
            "R2.2_gain_replicated": replicated,
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# NEXUS R2.4 external guarded-consensus report",
        "",
        f"Completed external tasks: **{len(rows)}** across {', '.join(completed)}.",
        "",
        "| Mode | Correct | Accuracy | Invalid |",
        "|---|---:|---:|---:|",
    ]
    for mode in ["A", "B", "C"]:
        value = modes[mode]
        lines.append(f"| {mode} | {value['correct']}/{value['n']} | {value['accuracy']:.1%} | {value['invalid_answers']} |")
    lines.extend([
        "",
        "## Per benchmark",
        "",
        "| Benchmark | Status | A | B | C | C-A | C corrections/regressions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for name in sorted(EXPECTED):
        summary = by_benchmark[name]
        if summary.get("status") != "COMPLETED":
            lines.append(f"| {name} | {summary.get('status')} | — | — | — | — | — |")
            continue
        a = summary["modes"]["A"]["accuracy"]
        b = summary["modes"]["B"]["accuracy"]
        c = summary["modes"]["C"]["accuracy"]
        guard = summary["guarded_consensus"]
        lines.append(
            f"| {name} | completed | {a:.1%} | {b:.1%} | {c:.1%} | {c-a:+.1%} | "
            f"{guard['B_to_C_corrections']}/{guard['B_to_C_regressions']} |"
        )
    lines.extend(["", "## Paired inference", ""])
    for name, value in pairs.items():
        lines.append(
            f"- **{name}:** {value['delta_accuracy']:+.1%}; wins/losses/ties "
            f"{value['wins']}/{value['losses']}/{value['ties']}; "
            f"95% bootstrap CI {value['bootstrap_95_ci']}; exact sign p={value['exact_sign_p']}."
        )
    lines.extend([
        "",
        "## Claim gate",
        "",
        f"R2.2 gain replicated: **{replicated}**.",
        "",
        "The claim remains false unless every preregistered benchmark completes, aggregate C-A has a positive 95% interval, no benchmark regresses, and guarded consensus produces more corrections than regressions.",
    ])
    (OUTPUT / "NEXUS_R2.4_EXTERNAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
