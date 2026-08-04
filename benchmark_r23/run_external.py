#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
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
        r"(?:ANSWER|FINAL|OPTION|CHOICE)\s*[:=\-]?\s*\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?\s*[\.!]?$",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and LETTERS.index(match.group(1)) < option_count:
            return match.group(1)
    return ""


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
                "/no_think\nYou are NEXUS structured solver in a closed-book benchmark. "
                "Decompose the problem, identify governing facts or equations, eliminate each incompatible option, "
                "check units and logical direction, then return JSON only with keys answer, confidence, rationale. "
                "The answer must be one option letter. Do not invent citations."
            ),
        },
        {"role": "user", "content": format_question(task)},
    ]


def prompt_proposal(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are the NEXUS proposal engine. Solve the closed-book multiple-choice question. "
                "Explicitly compare plausible options and expose uncertain assumptions. Return JSON only with "
                "answer, confidence, rationale, assumptions."
            ),
        },
        {"role": "user", "content": format_question(task)},
    ]


def prompt_critic(task: dict[str, Any], proposal: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are an adversarial NEXUS critic. Attack the proposed solution for factual, mathematical, "
                "causal, scope, sign, unit, and option-mapping errors. Re-solve independently. Return JSON only "
                "with proposed_answer, recommended_answer, defects, confidence."
            ),
        },
        {
            "role": "user",
            "content": f"{format_question(task)}\n\nProposal to audit:\n{proposal}",
        },
    ]


def prompt_final(task: dict[str, Any], proposal: str, critique: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are the NEXUS evidence arbitrator. The proposal and critique are fallible. "
                "Resolve their disagreement from the original question and options only. Prefer a correction only "
                "when the critic identifies a concrete defect. Return JSON only with answer, confidence, decision_basis."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{format_question(task)}\n\nProposal:\n{proposal}\n\nAdversarial critique:\n{critique}"
            ),
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
        raw = row.get("incorrect_answers", [])
        incorrect = [str(value) for value in raw]
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


def load_tasks(name: str, spec: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_id = spec["dataset"]
    source_sha = HfApi().dataset_info(dataset_id).sha
    dataset = load_dataset(dataset_id, spec["config"], split=spec["split"], revision=source_sha)
    if name == "mmlu_pro":
        normalized = [normalize_mmlu(dict(row)) for row in dataset]
    elif name == "gpqa":
        normalized = [normalize_gpqa(dict(row), seed) for row in dataset]
    elif name == "arc_challenge":
        normalized = [normalize_arc(dict(row)) for row in dataset]
    else:
        raise ValueError(f"unsupported benchmark: {name}")
    selected = stable_sample(normalized, name, int(spec["sample_n"]), seed)
    source = {
        "dataset": dataset_id,
        "config": spec["config"],
        "split": spec["split"],
        "repository_sha": source_sha,
        "full_split_size": len(normalized),
        "selected_n": len(selected),
        "selected_fingerprints": [
            {
                "stable_id": task["stable_id"],
                "question_sha256": sha256_text(task["question"]),
            }
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


def bootstrap_delta(a: list[int], b: list[int], seed: int, samples: int = 20000) -> list[float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["mmlu_pro", "gpqa", "arc_challenge"])
    args = parser.parse_args()

    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    seed = int(manifest["seed"])
    benchmark = args.benchmark
    out = ROOT / "results" / benchmark
    out.mkdir(parents=True, exist_ok=True)

    tasks, source = load_tasks(benchmark, manifest["benchmarks"][benchmark], seed)
    with (out / "selected_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    rows: list[dict[str, Any]] = []
    with (out / "responses.jsonl").open("w", encoding="utf-8") as sink:
        for index, task in enumerate(tasks, 1):
            option_count = len(task["options"])
            task_seed = seed + index

            a_raw = call_llm(prompt_a(task), max_tokens=72, seed=task_seed)
            a_obj = parse_jsonish(a_raw["content"])
            a_answer = extract_letter(a_obj.get("answer", a_raw["content"]), option_count)

            b_raw = call_llm(prompt_b(task), max_tokens=256, seed=task_seed)
            b_obj = parse_jsonish(b_raw["content"])
            b_answer = extract_letter(b_obj.get("answer", b_raw["content"]), option_count)

            proposal = call_llm(prompt_proposal(task), max_tokens=256, seed=task_seed)
            critique = call_llm(prompt_critic(task, proposal["content"]), max_tokens=224, seed=task_seed)
            final = call_llm(
                prompt_final(task, proposal["content"], critique["content"]),
                max_tokens=144,
                seed=task_seed,
            )
            final_obj = parse_jsonish(final["content"])
            c_answer = extract_letter(final_obj.get("answer", final["content"]), option_count)

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
                    "parsed": a_obj,
                    "raw": a_raw,
                },
                "B": {
                    "answer": b_answer,
                    "correct": b_answer == expected,
                    "parsed": b_obj,
                    "raw": b_raw,
                },
                "C": {
                    "answer": c_answer,
                    "correct": c_answer == expected,
                    "proposal": proposal,
                    "critique": critique,
                    "final": final,
                    "parsed_final": final_obj,
                },
            }
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{index:02d}/{len(tasks)}] {benchmark} id={task['stable_id']} "
                f"expected={expected} A={a_answer} B={b_answer} C={c_answer}",
                flush=True,
            )

    modes: dict[str, Any] = {}
    for mode in ["A", "B", "C"]:
        correct = sum(int(row[mode]["correct"]) for row in rows)
        modes[mode] = {
            "correct": correct,
            "n": len(rows),
            "accuracy": round(correct / len(rows), 4),
            "invalid_answers": sum(not row[mode]["answer"] for row in rows),
            "mean_latency_s": round(
                sum(
                    row[mode]["raw"]["latency_s"]
                    if mode in {"A", "B"}
                    else row[mode]["proposal"]["latency_s"]
                    + row[mode]["critique"]["latency_s"]
                    + row[mode]["final"]["latency_s"]
                    for row in rows
                )
                / len(rows),
                3,
            ),
        }

    summary = {
        "format": "nexus-r23-external-results/1",
        "benchmark": benchmark,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source": source,
        "model": manifest["model"],
        "seed": seed,
        "modes": modes,
        "pairs": {
            "B_minus_A": pair_stats(rows, "A", "B", seed + 101),
            "C_minus_A": pair_stats(rows, "A", "C", seed + 102),
            "C_minus_B": pair_stats(rows, "B", "C", seed + 103),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# NEXUS R2.3 — {benchmark} external pilot",
        "",
        f"Dataset repository SHA: `{source['repository_sha']}`",
        f"Selected tasks: **{len(rows)}** of {source['full_split_size']}",
        "",
        "| Mode | Correct | Accuracy | Invalid | Mean inference latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ["A", "B", "C"]:
        value = modes[mode]
        lines.append(
            f"| {mode} | {value['correct']}/{value['n']} | {value['accuracy']:.1%} | "
            f"{value['invalid_answers']} | {value['mean_latency_s']:.1f}s |"
        )
    lines.extend(["", "## Paired deltas", ""])
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
            "This is a preregistered stratified external pilot, not the official full-dataset leaderboard score. ",
            "Mode C uses more inference calls than A and B; any gain is a system-level compute-and-control gain, not a change in model weights.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
