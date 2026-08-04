#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.request
from collections import defaultdict
from fractions import Fraction
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
    last_error: str | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=1200) as response:
                raw = json.loads(response.read().decode("utf-8"))
            return {
                "content": raw["choices"][0]["message"]["content"],
                "usage": raw.get("usage", {}),
                "latency_s": round(time.time() - started, 3),
                "error": None,
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


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1 and number <= 100:
        number /= 100
    return round(max(0.0, min(1.0, number)), 4)


def extract_letter(value: Any, option_count: int) -> str:
    text = str(value or "").upper().strip()
    patterns = [
        r"(?:ANSWER|FINAL|OPTION|CHOICE|ОТВЕТ|ВАРИАНТ)\s*[:=\-]?\s*\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?\s*[\.!]?$",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and LETTERS.index(match.group(1)) < option_count:
            return match.group(1)
    return ""


def parse_number(value: Any) -> Fraction | None:
    text = str(value or "").strip()
    text = text.replace("−", "-").replace("–", "-")
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?", text)
    if not matches:
        return None
    token = matches[-1].replace(",", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            denominator_int = int(denominator)
            if denominator_int == 0:
                return None
            return Fraction(int(numerator), denominator_int)
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def fraction_text(value: Fraction | None) -> str:
    if value is None:
        return ""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def canonical_answer(task: dict[str, Any], value: Any) -> str:
    if task["kind"] == "mcq":
        return extract_letter(value, len(task["options"]))
    return fraction_text(parse_number(value))


def is_valid_answer(task: dict[str, Any], answer: str) -> bool:
    if task["kind"] == "mcq":
        return answer in LETTERS[: len(task["options"])]
    return parse_number(answer) is not None


def score_answer(task: dict[str, Any], answer: str) -> bool:
    if task["kind"] == "mcq":
        return answer == task["answer"]
    return parse_number(answer) == parse_number(task["answer"])


_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_ALLOWED_UNARY: dict[type[ast.unaryop], Any] = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _eval_ast(node: ast.AST, depth: int = 0) -> Fraction:
    if depth > 24:
        raise ValueError("expression too deep")
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_ast(node.operand, depth + 1))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_ast(node.left, depth + 1)
        right = _eval_ast(node.right, depth + 1)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 12:
                raise ValueError("unsafe exponent")
            return left ** right.numerator
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    raise ValueError(f"unsupported node: {type(node).__name__}")


def safe_eval_expression(value: Any) -> Fraction | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    if "=" in text:
        text = text.rsplit("=", 1)[-1].strip()
    text = text.strip("` $€£₽₸")
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", text):
        return None
    try:
        tree = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 80:
            return None
        return _eval_ast(tree)
    except Exception:  # noqa: BLE001
        return None


def map_numeric_to_option(task: dict[str, Any], result: Fraction) -> str:
    matches: list[str] = []
    for index, option in enumerate(task.get("options", [])):
        number = parse_number(option)
        if number == result:
            matches.append(LETTERS[index])
    return matches[0] if len(matches) == 1 else ""


def concrete_defects(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = [str(item).strip() for item in value]
    elif value is None:
        raw = []
    else:
        raw = [part.strip() for part in re.split(r"[;\n]", str(value))]
    rejected = {"", "none", "no defect", "no defects", "нет", "ошибок нет"}
    return [item for item in raw if item.lower() not in rejected and len(item) >= 4]


def format_task(task: dict[str, Any]) -> str:
    if task["kind"] == "mcq":
        options = "\n".join(
            f"{LETTERS[index]}. {text}" for index, text in enumerate(task["options"])
        )
        return f"Question:\n{task['question']}\n\nOptions:\n{options}"
    return f"Problem:\n{task['question']}"


def answer_contract(task: dict[str, Any]) -> str:
    if task["kind"] == "mcq":
        return '"answer":"A" where answer is one option letter'
    return '"answer":"42" where answer is only the final numeric value'


def prompt_a(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nClosed-book benchmark. Solve directly without tools. "
                f"Return JSON only with {answer_contract(task)} and confidence between 0 and 1."
            ),
        },
        {"role": "user", "content": format_task(task)},
    ]


def prompt_b(task: dict[str, Any], role: str) -> list[dict[str, str]]:
    if role == "analytic":
        instruction = "Solve analytically, checking the governing rule and rejecting tempting shortcuts."
    else:
        instruction = "Solve independently as a skeptical examiner and look for traps or sign errors."
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nMatched-compute vanilla solver. No external tools and no access to another solver. "
                f"{instruction} Return JSON only with {answer_contract(task)}, confidence, and short rationale."
            ),
        },
        {"role": "user", "content": format_task(task)},
    ]


def prompt_proposal(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are the proposal organ of an executable NEXUS runtime. Solve closed-book. "
                "Expose assumptions and provide one machine-checkable arithmetic expression when arithmetic is relevant. "
                f"Return JSON only with {answer_contract(task)}, confidence, rationale, assumptions, equation. "
                "The equation field must contain only one arithmetic expression or be empty."
            ),
        },
        {"role": "user", "content": format_task(task)},
    ]


def prompt_critic(task: dict[str, Any], proposal: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are the independent adversarial critic organ of NEXUS. Re-solve the original task, "
                "then identify concrete mathematical, factual, causal, scope, unit, or option-mapping defects in the proposal. "
                f"Return JSON only with \"recommended_answer\" using the same format as {answer_contract(task)}, "
                "confidence, defects as a list, rationale, corrected_equation. The corrected_equation must contain only one "
                "arithmetic expression or be empty."
            ),
        },
        {
            "role": "user",
            "content": f"{format_task(task)}\n\nProposal to audit:\n{proposal}",
        },
    ]


def normalize_mmlu(row: dict[str, Any]) -> dict[str, Any]:
    options = [str(item) for item in row["options"]]
    raw_answer = row.get("answer_index", row.get("answer"))
    if isinstance(raw_answer, int) or str(raw_answer).strip().isdigit():
        answer = LETTERS[int(raw_answer)]
    else:
        answer = str(raw_answer).strip().upper()
    question = str(row["question"])
    return {
        "stable_id": str(row.get("question_id", sha256_text(question)[:20])),
        "question": question,
        "options": options,
        "answer": answer,
        "category": str(row.get("category", "unknown")),
        "kind": "mcq",
    }


def normalize_arc(row: dict[str, Any]) -> dict[str, Any]:
    question: Any = row.get("question", "")
    if isinstance(question, dict):
        question = question.get("stem", question.get("text", ""))
    choices = row["choices"]
    texts = [str(item) for item in choices["text"]]
    labels = [str(item).upper() for item in choices["label"]]
    answer_key = str(row["answerKey"]).upper()
    if answer_key in labels:
        answer_index = labels.index(answer_key)
    elif answer_key.isdigit() and 0 < int(answer_key) <= len(texts):
        answer_index = int(answer_key) - 1
    else:
        raise ValueError(f"ARC answer key {answer_key!r} not found in {labels!r}")
    return {
        "stable_id": str(row.get("id", sha256_text(str(question))[:20])),
        "question": str(question),
        "options": texts,
        "answer": LETTERS[answer_index],
        "category": "science",
        "kind": "mcq",
    }


def normalize_gsm8k(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"])
    answer = fraction_text(parse_number(row["answer"]))
    if not answer:
        raise ValueError("GSM8K answer could not be parsed")
    return {
        "stable_id": sha256_text(question)[:20],
        "question": question,
        "options": [],
        "answer": answer,
        "category": "math_word_problem",
        "kind": "numeric",
    }


def stable_rank(tasks: list[dict[str, Any]], benchmark: str, seed: int) -> list[dict[str, Any]]:
    return sorted(
        tasks,
        key=lambda task: sha256_text(
            f"{seed}|{benchmark}|{task['category']}|{task['stable_id']}|{task['question']}"
        ),
    )


def category_aware_sample(
    tasks: list[dict[str, Any]], benchmark: str, sample_n: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in stable_rank(tasks, benchmark, seed):
        groups[task["category"]].append(task)
    categories = sorted(groups)
    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < sample_n:
        added = False
        for category in categories:
            if position < len(groups[category]):
                selected.append(groups[category][position])
                added = True
                if len(selected) == sample_n:
                    break
        if not added:
            break
        position += 1
    return stable_rank(selected, benchmark, seed)


def load_tasks(
    benchmark: str, spec: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_id = spec["dataset"]
    source_sha = HfApi().dataset_info(dataset_id).sha
    dataset = load_dataset(
        dataset_id,
        spec["config"],
        split=spec["split"],
        revision=source_sha,
    )
    if benchmark == "mmlu_pro":
        normalized = [normalize_mmlu(dict(row)) for row in dataset]
    elif benchmark == "arc_challenge":
        normalized = [normalize_arc(dict(row)) for row in dataset]
    elif benchmark == "gsm8k":
        normalized = [normalize_gsm8k(dict(row)) for row in dataset]
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    selected = category_aware_sample(
        normalized,
        benchmark,
        int(spec["sample_n"]),
        seed,
    )
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
                "category": task["category"],
            }
            for task in selected
        ],
    }
    return selected, source


def vanilla_resolution(task: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_answer = canonical_answer(task, left.get("answer"))
    right_answer = canonical_answer(task, right.get("answer"))
    left_confidence = clamp_confidence(left.get("confidence"))
    right_confidence = clamp_confidence(right.get("confidence"))
    if left_answer and left_answer == right_answer:
        answer = left_answer
        basis = "agreement"
    elif is_valid_answer(task, right_answer) and right_confidence > left_confidence:
        answer = right_answer
        basis = "higher_reported_confidence"
    elif is_valid_answer(task, left_answer):
        answer = left_answer
        basis = "first_valid_or_tie"
    else:
        answer = right_answer if is_valid_answer(task, right_answer) else ""
        basis = "only_valid"
    return {
        "answer": answer,
        "basis": basis,
        "left_answer": left_answer,
        "right_answer": right_answer,
        "left_confidence": left_confidence,
        "right_confidence": right_confidence,
        "disagreement": bool(left_answer and right_answer and left_answer != right_answer),
    }


def exact_tool_records(
    task: dict[str, Any], proposal: dict[str, Any], critic: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidates = [
        ("proposal", proposal.get("equation")),
        ("critic", critic.get("corrected_equation", critic.get("equation"))),
    ]
    for source, expression in candidates:
        result = safe_eval_expression(expression)
        if result is None:
            continue
        if task["kind"] == "mcq":
            mapped = map_numeric_to_option(task, result)
        else:
            mapped = fraction_text(result)
        records.append(
            {
                "source": source,
                "expression": str(expression),
                "result": fraction_text(result),
                "mapped_answer": mapped,
                "mapped": is_valid_answer(task, mapped),
            }
        )
    return records


def nexus_resolution(task: dict[str, Any], proposal: dict[str, Any], critic: dict[str, Any]) -> dict[str, Any]:
    proposal_answer = canonical_answer(task, proposal.get("answer"))
    critic_answer = canonical_answer(
        task,
        critic.get("recommended_answer", critic.get("answer")),
    )
    proposal_confidence = clamp_confidence(proposal.get("confidence"))
    critic_confidence = clamp_confidence(critic.get("confidence"))
    defects = concrete_defects(critic.get("defects"))
    exact_records = exact_tool_records(task, proposal, critic)
    mapped = [record for record in exact_records if record["mapped"]]

    answer = ""
    basis = "no_valid_candidate"
    tool_override = False
    if mapped:
        unique = {record["mapped_answer"] for record in mapped}
        if len(unique) == 1:
            answer = next(iter(unique))
            basis = "typed_exact_consensus"
            tool_override = answer not in {proposal_answer, critic_answer}
        elif defects:
            critic_tools = [record for record in mapped if record["source"] == "critic"]
            if critic_tools:
                answer = critic_tools[-1]["mapped_answer"]
                basis = "typed_exact_critic_correction"
                tool_override = answer != proposal_answer

    if not answer and proposal_answer and proposal_answer == critic_answer:
        answer = proposal_answer
        basis = "proposal_critic_agreement"
    elif not answer and defects and is_valid_answer(task, critic_answer):
        threshold = max(0.55, proposal_confidence - 0.15)
        if critic_confidence >= threshold:
            answer = critic_answer
            basis = "concrete_defect_correction"
    if not answer and is_valid_answer(task, proposal_answer):
        if not is_valid_answer(task, critic_answer) or proposal_confidence >= critic_confidence:
            answer = proposal_answer
            basis = "proposal_higher_confidence"
    if not answer and is_valid_answer(task, critic_answer):
        answer = critic_answer
        basis = "critic_only_valid_or_higher_confidence"

    blackboard = {
        "proposal_answer": proposal_answer,
        "critic_answer": critic_answer,
        "proposal_confidence": proposal_confidence,
        "critic_confidence": critic_confidence,
        "answer_contradiction": bool(
            proposal_answer and critic_answer and proposal_answer != critic_answer
        ),
        "defects": defects,
        "exact_tool_records": exact_records,
    }
    telemetry = {
        "blackboard": 1,
        "contradiction_detector": 1,
        "typed_exact_arithmetic": int(bool(exact_records)),
        "deterministic_arbitration": 1,
        "tool_override": int(tool_override),
        "concrete_defect_count": len(defects),
    }
    return {
        "answer": answer,
        "basis": basis,
        "blackboard": blackboard,
        "telemetry": telemetry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["mmlu_pro", "arc_challenge", "gsm8k"],
    )
    parser.add_argument("--shard-index", required=True, type=int)
    args = parser.parse_args()

    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    benchmark = args.benchmark
    spec = manifest["benchmarks"][benchmark]
    shard_count = int(spec["shards"])
    if not 0 <= args.shard_index < shard_count:
        raise ValueError("invalid shard index")
    seed = int(manifest["seed"])
    selected, source = load_tasks(benchmark, spec, seed)
    tasks = selected[args.shard_index :: shard_count]

    out = ROOT / "results" / benchmark / f"shard-{args.shard_index}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (out / "selected_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    rows: list[dict[str, Any]] = []
    with (out / "responses.jsonl").open("w", encoding="utf-8") as sink:
        for local_index, task in enumerate(tasks, 1):
            global_seed = seed + args.shard_index * 1000 + local_index

            a_raw = call_llm(prompt_a(task), max_tokens=72, seed=global_seed)
            a_obj = parse_jsonish(a_raw["content"])
            a_answer = canonical_answer(task, a_obj.get("answer", a_raw["content"]))

            b_left_raw = call_llm(
                prompt_b(task, "analytic"),
                max_tokens=160,
                seed=global_seed + 10000,
            )
            b_right_raw = call_llm(
                prompt_b(task, "skeptical"),
                max_tokens=160,
                seed=global_seed + 20000,
            )
            b_left = parse_jsonish(b_left_raw["content"])
            b_right = parse_jsonish(b_right_raw["content"])
            b_resolution = vanilla_resolution(task, b_left, b_right)
            b_answer = b_resolution["answer"]

            proposal_raw = call_llm(
                prompt_proposal(task),
                max_tokens=160,
                seed=global_seed + 30000,
            )
            proposal = parse_jsonish(proposal_raw["content"])
            critic_raw = call_llm(
                prompt_critic(task, proposal_raw["content"]),
                max_tokens=160,
                seed=global_seed + 40000,
            )
            critic = parse_jsonish(critic_raw["content"])
            c_resolution = nexus_resolution(task, proposal, critic)
            c_answer = c_resolution["answer"]
            proposal_answer = canonical_answer(task, proposal.get("answer"))

            row = {
                "format": "nexus-r23-power-row/1",
                "benchmark": benchmark,
                "shard_index": args.shard_index,
                "stable_id": task["stable_id"],
                "category": task["category"],
                "kind": task["kind"],
                "question_sha256": sha256_text(task["question"]),
                "expected": task["answer"],
                "A": {
                    "answer": a_answer,
                    "correct": score_answer(task, a_answer),
                    "valid": is_valid_answer(task, a_answer),
                    "raw": a_raw,
                    "parsed": a_obj,
                    "llm_calls": 1,
                },
                "B": {
                    "answer": b_answer,
                    "correct": score_answer(task, b_answer),
                    "valid": is_valid_answer(task, b_answer),
                    "resolution": b_resolution,
                    "left": {"raw": b_left_raw, "parsed": b_left},
                    "right": {"raw": b_right_raw, "parsed": b_right},
                    "llm_calls": 2,
                },
                "C": {
                    "answer": c_answer,
                    "correct": score_answer(task, c_answer),
                    "valid": is_valid_answer(task, c_answer),
                    "proposal_answer": proposal_answer,
                    "proposal_correct": score_answer(task, proposal_answer),
                    "resolution": c_resolution,
                    "proposal": {"raw": proposal_raw, "parsed": proposal},
                    "critic": {"raw": critic_raw, "parsed": critic},
                    "llm_calls": 2,
                },
            }
            rows.append(row)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "shard": args.shard_index,
                        "item": local_index,
                        "total": len(tasks),
                        "A": int(row["A"]["correct"]),
                        "B": int(row["B"]["correct"]),
                        "C": int(row["C"]["correct"]),
                        "c_basis": c_resolution["basis"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "format": "nexus-r23-power-shard-summary/1",
        "benchmark": benchmark,
        "shard_index": args.shard_index,
        "shard_count": shard_count,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source": source,
        "tasks": len(rows),
        "accuracy": {
            mode: round(sum(int(row[mode]["correct"]) for row in rows) / len(rows), 6)
            for mode in ["A", "B", "C"]
        },
        "valid_rate": {
            mode: round(sum(int(row[mode]["valid"]) for row in rows) / len(rows), 6)
            for mode in ["A", "B", "C"]
        },
        "llm_calls": {
            mode: sum(int(row[mode]["llm_calls"]) for row in rows)
            for mode in ["A", "B", "C"]
        },
        "nexus_tool_activations": sum(
            int(row["C"]["resolution"]["telemetry"]["typed_exact_arithmetic"])
            for row in rows
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
