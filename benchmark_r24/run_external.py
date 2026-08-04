#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
API_URL = os.environ.get("NEXUS_LLM_API", "http://127.0.0.1:8080/v1/chat/completions")
MODEL = os.environ.get("NEXUS_LLM_MODEL", "qwen3-14b")
LETTERS = "ABCDEFGHIJ"
AUTH_BLOCKED_EXIT = 42


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_llm(messages: list[dict[str, str]], max_tokens: int, seed: int) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=1200) as response:
                raw = json.loads(response.read().decode("utf-8"))
            return {
                "content": raw["choices"][0]["message"]["content"],
                "usage": raw.get("usage", {}),
                "latency_s": round(time.time() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(8 * (attempt + 1))
    return {
        "content": "",
        "usage": {},
        "latency_s": round(time.time() - started, 3),
        "error": last_error,
    }


def parse_jsonish(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"answer": value}
    except Exception:  # noqa: BLE001
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {"answer": value}
        except Exception:  # noqa: BLE001
            pass
    return {"answer": text}


def extract_letter(value: Any, option_count: int) -> str:
    text = str(value or "").upper().strip()
    patterns = [
        r"(?:ANSWER|FINAL|OPTION|CHOICE|RECOMMENDED_ANSWER)\s*[:=\-]?\s*\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?\s*[\.!]?$",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and LETTERS.index(match.group(1)) < option_count:
            return match.group(1)
    return ""


def extract_confidence(obj: dict[str, Any], default: float = 0.5) -> float:
    try:
        value = float(obj.get("confidence", default))
    except (TypeError, ValueError):
        return default
    return round(max(0.0, min(1.0, value)), 4)


def format_question(task: dict[str, Any]) -> str:
    options = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(task["options"]))
    return f"Question:\n{task['question']}\n\nOptions:\n{options}"


def prompt_a(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nClosed-book multiple-choice evaluation. Do not use external tools. "
                "Return JSON only: {\"answer\":\"A\",\"confidence\":0.0}."
            ),
        },
        {"role": "user", "content": format_question(task)},
    ]


def prompt_b(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are NEXUS Solver-1 in a closed-book benchmark. Decompose the problem, identify "
                "governing facts or equations, eliminate incompatible options, check units, scope, sign, causal "
                "direction, and option mapping. Return JSON only with answer, confidence, rationale. Do not invent citations."
            ),
        },
        {"role": "user", "content": format_question(task)},
    ]


def prompt_challenger(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are NEXUS Solver-2. Independently solve the multiple-choice question without seeing "
                "another solver's answer. Search specifically for traps involving quantifiers, exceptions, units, "
                "signs, causal direction, and near-miss options. Return JSON only with answer, confidence, rationale."
            ),
        },
        {"role": "user", "content": format_question(task)},
    ]


def prompt_arbitrator(task: dict[str, Any], solver_one: str, solver_two: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are the NEXUS disagreement arbitrator. Both candidate solutions are fallible. "
                "Re-check the original question, identify the exact point of disagreement, and select one option. "
                "Do not change an answer merely for novelty. Return JSON only with answer, confidence, "
                "winner, concrete_defect, decision_basis."
            ),
        },
        {
            "role": "user",
            "content": f"{format_question(task)}\n\nSolver-1:\n{solver_one}\n\nSolver-2:\n{solver_two}",
        },
    ]


def stable_sample(tasks: list[dict[str, Any]], benchmark: str, n: int, seed: int) -> list[dict[str, Any]]:
    ranked = []
    for task in tasks:
        key = sha256_text(f"{seed}|{benchmark}|{task['stable_id']}|{task['question']}")
        ranked.append((key, task))
    ranked.sort(key=lambda item: item[0])
    return [task for _, task in ranked[:n]]


def normalize_mmlu(row: dict[str, Any]) -> dict[str, Any]:
    options = [str(value) for value in row["options"]]
    if row.get("answer_index") is not None:
        answer = LETTERS[int(row["answer_index"])]
    else:
        answer = str(row["answer"]).strip().upper()
    stable_id = str(row.get("question_id", sha256_text(str(row["question"]))[:16]))
    return {
        "stable_id": stable_id,
        "question": str(row["question"]),
        "options": options,
        "answer": answer,
        "category": str(row.get("category", "unknown")),
    }


def normalize_gpqa(row: dict[str, Any], seed: int) -> dict[str, Any]:
    question = str(row.get("Question", row.get("question", "")))
    correct = str(row.get("Correct Answer", row.get("correct_answer", "")))
    incorrect = []
    for key in ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
        if key in row:
            incorrect.append(str(row[key]))
    if not incorrect:
        incorrect = [str(value) for value in row.get("incorrect_answers", [])]
    choices = [(correct, True)] + [(value, False) for value in incorrect]
    stable_id = str(row.get("Record ID", row.get("record_id", sha256_text(question)[:16])))
    rng = random.Random(int(sha256_text(f"{seed}|gpqa|{stable_id}")[:16], 16))
    rng.shuffle(choices)
    answer_index = next(i for i, (_, is_correct) in enumerate(choices) if is_correct)
    return {
        "stable_id": stable_id,
        "question": question,
        "options": [value for value, _ in choices],
        "answer": LETTERS[answer_index],
        "category": str(row.get("High-level domain", row.get("domain", "science"))),
    }


def normalize_arc(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("question", "")
    if isinstance(question, dict):
        question = question.get("stem", question.get("text", ""))
    choices = row["choices"]
    texts = [str(value) for value in choices["text"]]
    labels = [str(value).upper() for value in choices["label"]]
    answer_key = str(row["answerKey"]).upper()
    if answer_key in labels:
        answer_index = labels.index(answer_key)
    elif answer_key.isdigit() and int(answer_key) - 1 < len(texts):
        answer_index = int(answer_key) - 1
    else:
        raise ValueError(f"ARC answer key not found: {answer_key!r} in {labels!r}")
    return {
        "stable_id": str(row.get("id", sha256_text(str(question))[:16])),
        "question": str(question),
        "options": texts,
        "answer": LETTERS[answer_index],
        "category": "science",
    }


def is_auth_blocked(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    needles = ["gated", "authenticated", "authentication", "access request", "401", "403"]
    return any(needle in text for needle in needles)


def blocked_summary(benchmark: str, manifest_sha256: str, model: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "format": "nexus-r24-external-results/1",
        "benchmark": benchmark,
        "status": "AUTH_BLOCKED",
        "manifest_sha256": manifest_sha256,
        "model": model,
        "reason": f"{type(exc).__name__}: {exc}",
        "required_secret": "HF_TOKEN with approved access to the gated dataset",
    }


def load_tasks(name: str, spec: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_id = spec["dataset"]
    token = os.environ.get("HF_TOKEN") or None
    source_sha = HfApi(token=token).dataset_info(dataset_id).sha
    dataset = load_dataset(
        dataset_id,
        spec["config"],
        split=spec["split"],
        revision=source_sha,
        token=token,
    )
    if name == "mmlu_pro":
        normalized = [normalize_mmlu(dict(row)) for row in dataset]
    elif name == "gpqa":
        normalized = [normalize_gpqa(dict(row), seed) for row in dataset]
    elif name == "arc_challenge":
        normalized = [normalize_arc(dict(row)) for row in dataset]
    else:
        raise ValueError(f"unsupported benchmark: {name}")
    excluded = {str(value) for value in spec.get("exclude_stable_ids_seen_in_r23", [])}
    eligible = [task for task in normalized if task["stable_id"] not in excluded]
    selected = stable_sample(eligible, name, int(spec["sample_n"]), seed)
    source = {
        "dataset": dataset_id,
        "config": spec["config"],
        "split": spec["split"],
        "repository_sha": source_sha,
        "full_split_size": len(normalized),
        "excluded_seen_in_r23": sorted(excluded),
        "eligible_split_size": len(eligible),
        "selected_n": len(selected),
        "selected_fingerprints": [
            {"stable_id": task["stable_id"], "question_sha256": sha256_text(task["question"])}
            for task in selected
        ],
    }
    return selected, source


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_delta(a: list[int], b: list[int], seed: int, samples: int = 30000) -> list[float]:
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(b[i] - a[i] for i in indices) / n)
    values.sort()
    lo = values[int(0.025 * samples)]
    hi = values[int(0.975 * samples) - 1]
    return [round(lo, 4), round(hi, 4)]


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


def usage_tokens(raw: dict[str, Any]) -> int:
    usage = raw.get("usage") or {}
    for key in ("total_tokens", "tokens"):
        if key not in usage:
            continue
        try:
            return int(usage[key])
        except (TypeError, ValueError):
            return 0
    return 0


def write_blocked(out: Path, payload: dict[str, Any]) -> None:
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(
        "# NEXUS R2.4 external shard\n\n"
        f"Status: **{payload['status']}**\n\n"
        f"Reason: `{payload['reason']}`\n\n"
        "No benchmark score was manufactured or substituted.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["mmlu_pro", "gpqa", "arc_challenge"])
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    seed = int(manifest["seed"])
    benchmark = args.benchmark
    out = ROOT / "results" / benchmark
    out.mkdir(parents=True, exist_ok=True)

    try:
        tasks, source = load_tasks(benchmark, manifest["benchmarks"][benchmark], seed)
    except Exception as exc:  # noqa: BLE001
        if is_auth_blocked(exc):
            payload = blocked_summary(benchmark, manifest_sha256, manifest["model"], exc)
            write_blocked(out, payload)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            raise SystemExit(AUTH_BLOCKED_EXIT if args.probe_only else 0) from exc
        raise

    if args.probe_only:
        probe = {"status": "READY", "benchmark": benchmark, "source": source, "selected_count": len(tasks)}
        (out / "probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(probe, ensure_ascii=False), flush=True)
        return

    with (out / "selected_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    rows: list[dict[str, Any]] = []
    with (out / "responses.jsonl").open("w", encoding="utf-8") as sink:
        for index, task in enumerate(tasks, 1):
            option_count = len(task["options"])
            task_seed = seed + index * 10

            a_raw = call_llm(prompt_a(task), max_tokens=72, seed=task_seed + 1)
            a_obj = parse_jsonish(a_raw["content"])
            a_answer = extract_letter(a_obj.get("answer", a_raw["content"]), option_count)

            b_raw = call_llm(prompt_b(task), max_tokens=256, seed=task_seed + 2)
            b_obj = parse_jsonish(b_raw["content"])
            b_answer = extract_letter(b_obj.get("answer", b_raw["content"]), option_count)

            challenger_raw = call_llm(prompt_challenger(task), max_tokens=256, seed=task_seed + 3)
            challenger_obj = parse_jsonish(challenger_raw["content"])
            challenger_answer = extract_letter(challenger_obj.get("answer", challenger_raw["content"]), option_count)

            arbitration_used = not b_answer or not challenger_answer or b_answer != challenger_answer
            arbitrator_raw: dict[str, Any] = {"content": "", "usage": {}, "latency_s": 0.0, "skipped": True}
            arbitrator_obj: dict[str, Any] = {}
            if arbitration_used:
                arbitrator_raw = call_llm(
                    prompt_arbitrator(task, b_raw["content"], challenger_raw["content"]),
                    max_tokens=192,
                    seed=task_seed + 4,
                )
                arbitrator_obj = parse_jsonish(arbitrator_raw["content"])
                arbitrator_answer = extract_letter(arbitrator_obj.get("answer", arbitrator_raw["content"]), option_count)
                c_answer = arbitrator_answer or b_answer or challenger_answer
                c_confidence = extract_confidence(arbitrator_obj)
                decision = "arbitrated"
            else:
                c_answer = b_answer
                c_confidence = round((extract_confidence(b_obj) + extract_confidence(challenger_obj)) / 2, 4)
                decision = "independent_consensus"

            expected = task["answer"]
            row = {
                "index": index,
                "benchmark": benchmark,
                "stable_id": task["stable_id"],
                "category": task["category"],
                "question_sha256": sha256_text(task["question"]),
                "expected": expected,
                "A": {
                    "answer": a_answer,
                    "correct": a_answer == expected,
                    "confidence": extract_confidence(a_obj),
                    "parsed": a_obj,
                    "raw": a_raw,
                },
                "B": {
                    "answer": b_answer,
                    "correct": b_answer == expected,
                    "confidence": extract_confidence(b_obj),
                    "parsed": b_obj,
                    "raw": b_raw,
                },
                "C": {
                    "answer": c_answer,
                    "correct": c_answer == expected,
                    "confidence": c_confidence,
                    "decision": decision,
                    "arbitration_used": arbitration_used,
                    "solver_one": b_raw,
                    "solver_one_parsed": b_obj,
                    "solver_two": challenger_raw,
                    "solver_two_parsed": challenger_obj,
                    "arbitrator": arbitrator_raw,
                    "arbitrator_parsed": arbitrator_obj,
                },
            }
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{index:02d}/{len(tasks)}] {benchmark} id={task['stable_id']} expected={expected} "
                f"A={a_answer} B={b_answer} S2={challenger_answer} C={c_answer} decision={decision}",
                flush=True,
            )

    modes: dict[str, Any] = {}
    for mode in ["A", "B", "C"]:
        correct = sum(int(row[mode]["correct"]) for row in rows)
        if mode == "A":
            latencies = [row[mode]["raw"]["latency_s"] for row in rows]
            tokens = [usage_tokens(row[mode]["raw"]) for row in rows]
        elif mode == "B":
            latencies = [row[mode]["raw"]["latency_s"] for row in rows]
            tokens = [usage_tokens(row[mode]["raw"]) for row in rows]
        else:
            latencies = [
                row[mode]["solver_one"]["latency_s"]
                + row[mode]["solver_two"]["latency_s"]
                + row[mode]["arbitrator"]["latency_s"]
                for row in rows
            ]
            tokens = [
                usage_tokens(row[mode]["solver_one"])
                + usage_tokens(row[mode]["solver_two"])
                + usage_tokens(row[mode]["arbitrator"])
                for row in rows
            ]
        brier = sum((float(row[mode]["confidence"]) - float(row[mode]["correct"])) ** 2 for row in rows) / len(rows)
        modes[mode] = {
            "correct": correct,
            "n": len(rows),
            "accuracy": round(correct / len(rows), 4),
            "invalid_answers": sum(not row[mode]["answer"] for row in rows),
            "mean_latency_s": round(sum(latencies) / len(latencies), 3),
            "total_reported_tokens": sum(tokens),
            "brier_score": round(brier, 4),
        }
    modes["C"]["arbitration_count"] = sum(row["C"]["arbitration_used"] for row in rows)
    modes["C"]["arbitration_rate"] = round(modes["C"]["arbitration_count"] / len(rows), 4)

    corrections = sum((not row["B"]["correct"]) and row["C"]["correct"] for row in rows)
    regressions = sum(row["B"]["correct"] and (not row["C"]["correct"]) for row in rows)
    summary = {
        "format": "nexus-r24-external-results/1",
        "benchmark": benchmark,
        "status": "COMPLETED",
        "manifest_sha256": manifest_sha256,
        "source": source,
        "model": manifest["model"],
        "seed": seed,
        "modes": modes,
        "guarded_consensus": {
            "B_to_C_corrections": corrections,
            "B_to_C_regressions": regressions,
            "net": corrections - regressions,
        },
        "pairs": {
            "B_minus_A": pair_stats(rows, "A", "B", seed + 101),
            "C_minus_A": pair_stats(rows, "A", "C", seed + 102),
            "C_minus_B": pair_stats(rows, "B", "C", seed + 103),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# NEXUS R2.4 — {benchmark} guarded-consensus panel",
        "",
        f"Dataset repository SHA: `{source['repository_sha']}`",
        f"Selected tasks: **{len(rows)}** of {source['full_split_size']}",
        "",
        "| Mode | Correct | Accuracy | Invalid | Mean latency | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ["A", "B", "C"]:
        value = modes[mode]
        lines.append(
            f"| {mode} | {value['correct']}/{value['n']} | {value['accuracy']:.1%} | "
            f"{value['invalid_answers']} | {value['mean_latency_s']:.1f}s | {value['brier_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"C arbitration rate: **{modes['C']['arbitration_rate']:.1%}**.",
            f"B→C corrections/regressions: **{corrections}/{regressions}**.",
            "",
            "## Paired deltas",
            "",
        ]
    )
    for name, value in summary["pairs"].items():
        lines.append(
            f"- **{name}:** {value['delta_accuracy']:+.1%}; wins/losses/ties "
            f"{value['wins']}/{value['losses']}/{value['ties']}; "
            f"95% bootstrap CI {value['bootstrap_95_ci']}; exact sign p={value['exact_sign_p']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This panel was frozen after the R2.3 negative-control result and uses a different seed. "
            "It tests a preregistered architectural correction, not post-hoc rescoring of R2.3 tasks.",
            "Mode C reuses the structured Solver-1 call and adds an independent challenger; a third call is made only on disagreement.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
