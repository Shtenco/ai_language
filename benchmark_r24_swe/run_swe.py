#!/usr/bin/env python3
# ruff: noqa
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"
OUT = ROOT / "results"
WORK = ROOT / "work"
API_URL = os.environ.get("NEXUS_LLM_API", "http://127.0.0.1:8080/v1/chat/completions")
MODEL = os.environ.get("NEXUS_LLM_MODEL", "qwen3-14b")

STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before", "being", "between",
    "could", "does", "from", "have", "into", "issue", "more", "should", "some", "such", "that",
    "their", "there", "these", "they", "this", "when", "where", "which", "while", "with", "would",
    "function", "class", "method", "error", "expected", "current", "return", "using", "tests", "test",
    "version", "attribute", "instances", "purpose", "assume", "changes", "introduced",
}


def run(command: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


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
            with urllib.request.urlopen(request, timeout=1800) as response:
                raw = json.loads(response.read().decode("utf-8"))
            return {
                "content": raw["choices"][0]["message"]["content"],
                "usage": raw.get("usage", {}),
                "latency_s": round(time.time() - started, 3),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(10 * (attempt + 1))
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
        return value if isinstance(value, dict) else {"edits": []}
    except Exception:  # noqa: BLE001
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {"edits": []}
        except Exception:  # noqa: BLE001
            pass
    return {"edits": [], "parse_error": text[:4000]}


def issue_identifiers(problem: str) -> list[str]:
    ordered: list[str] = []
    quoted = re.findall(r"`([^`\n]+)`", problem)
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", problem)
    candidates = quoted + tokens
    for raw in candidates:
        value = raw.strip(" .()[]{}'\"")
        lowered = value.lower()
        if lowered in STOPWORDS or len(value) < 3 or value in ordered:
            continue
        ordered.append(value)
    return ordered[:40]


def collect_context(repo: Path, problem: str, max_chars: int = 32000) -> tuple[str, list[dict[str, Any]]]:
    identifiers = issue_identifiers(problem)
    quoted_identifiers = set(re.findall(r"`([^`\n]+)`", problem))
    files = run(["git", "ls-files", "*.py"], cwd=repo).stdout.splitlines()
    ranked: list[tuple[int, str, str, list[int]]] = []
    for relative in files:
        path = repo / relative
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = text.lower()
        score = 0
        hits: list[int] = []
        lines = text.splitlines()
        for identifier in identifiers:
            needle = identifier.lower()
            count = lowered.count(needle)
            if count:
                weight = 18 if identifier in quoted_identifiers else 5
                score += min(count, 8) * weight
                for index, line in enumerate(lines):
                    if needle in line.lower():
                        hits.append(index)
                        break
            if re.match(r"^[A-Z][A-Za-z0-9_]+$", identifier):
                declaration = f"class {identifier}"
                if declaration in text:
                    score += 100
            if needle in Path(relative).stem.lower():
                score += 35
        if score:
            ranked.append((score, relative, text, sorted(set(hits))))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    parts: list[str] = []
    audit: list[dict[str, Any]] = []
    used = 0
    for score, relative, text, hits in ranked[:12]:
        lines = text.splitlines(keepends=True)
        center = hits[0] if hits else 0
        start = max(0, center - 80)
        end = min(len(lines), center + 160)
        snippet = "".join(lines[start:end])
        block = f"\n<<<FILE {relative} LINES {start + 1}-{end}>>>\n{snippet}<<<END FILE>>>\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 1000:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block)
        audit.append({"path": relative, "score": score, "start_line": start + 1, "end_line": end})
    return "".join(parts), audit


def plan_contract() -> str:
    return (
        "Return JSON only with this schema: "
        '{"analysis":"brief","edits":[{"path":"tracked/file.py","search":"exact existing text",'
        '"replace":"replacement text"}]}. '
        "Each search string must occur exactly once in the supplied repository context. "
        "Do not output a unified diff, Markdown, line numbers inside search strings, new files, or test claims."
    )


def prompt_a(problem: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "/no_think\nYou are a software engineer. Produce the smallest exact edit plan. " + plan_contract(),
        },
        {"role": "user", "content": f"Issue:\n{problem}\n\nRepository context:\n{context}"},
    ]


def prompt_b(problem: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are NEXUS structured code solver. Internally trace the relevant class hierarchy, "
                "identify the first bad change, preserve invariants and minimize behavioral surface. " + plan_contract()
            ),
        },
        {"role": "user", "content": f"Issue:\n{problem}\n\nRepository context:\n{context}"},
    ]


def prompt_repair(problem: str, context: str, plan: dict[str, Any], error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nRepair an edit plan rejected by deterministic repository validation. "
                "Use only exact text visible in the context. " + plan_contract()
            ),
        },
        {
            "role": "user",
            "content": (
                f"Issue:\n{problem}\n\nRepository context:\n{context}\n\nRejected plan:\n"
                f"{json.dumps(plan, ensure_ascii=False)}\n\nValidator error:\n{error}"
            ),
        },
    ]


def prompt_review(problem: str, context: str, patch: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are an independent senior maintainer. Review the canonical patch against the issue "
                "and supplied source. Return JSON only: "
                '{"accept":true,"defects":[],"corrected_edits":[]} or '
                '{"accept":false,"defects":["concrete defect"],"corrected_edits":[{"path":"...",'
                '"search":"exact existing text","replace":"..."}]}. '
                "Do not use hidden tests or gold patches."
            ),
        },
        {
            "role": "user",
            "content": f"Issue:\n{problem}\n\nRepository context:\n{context}\n\nCanonical proposal patch:\n{patch}",
        },
    ]


def reset_repo(repo: Path, base_commit: str) -> None:
    run(["git", "reset", "--hard", base_commit], cwd=repo)
    run(["git", "clean", "-fdx"], cwd=repo)


def canonicalize_plan(repo: Path, base_commit: str, plan: dict[str, Any], label: str) -> dict[str, Any]:
    reset_repo(repo, base_commit)
    edits = plan.get("edits")
    if not isinstance(edits, list) or not edits:
        return {"valid": False, "error": "plan has no edits", "patch": "", "label": label}
    if len(edits) > 8:
        return {"valid": False, "error": "plan exceeds eight edits", "patch": "", "label": label}
    changed_paths: list[str] = []
    try:
        tracked = set(run(["git", "ls-files"], cwd=repo).stdout.splitlines())
        for index, edit in enumerate(edits, 1):
            if not isinstance(edit, dict):
                raise ValueError(f"edit {index} is not an object")
            relative = str(edit.get("path", ""))
            search = str(edit.get("search", ""))
            replace = str(edit.get("replace", ""))
            path = Path(relative)
            if not relative or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"edit {index} has unsafe path {relative!r}")
            if relative not in tracked:
                raise ValueError(f"edit {index} path is not a tracked file: {relative}")
            if not search or search == replace:
                raise ValueError(f"edit {index} has empty or no-op search/replace")
            target = repo / relative
            text = target.read_text(encoding="utf-8")
            count = text.count(search)
            if count != 1:
                raise ValueError(f"edit {index} search occurs {count} times in {relative}; expected exactly once")
            target.write_text(text.replace(search, replace, 1), encoding="utf-8")
            changed_paths.append(relative)
        diff_check = run(["git", "diff", "--check"], cwd=repo, check=False)
        if diff_check.returncode:
            raise ValueError(f"git diff --check failed: {diff_check.stdout[-4000:]}")
        patch = run(["git", "diff", "--binary", "--no-ext-diff", base_commit, "--"], cwd=repo).stdout
        if not patch.strip():
            raise ValueError("canonical git diff is empty")
        patch_path = OUT / f"{label}.candidate.patch"
        patch_path.write_text(patch, encoding="utf-8")
        reset_repo(repo, base_commit)
        apply_check = run(["git", "apply", "--check", str(patch_path.resolve())], cwd=repo, check=False)
        if apply_check.returncode:
            raise ValueError(f"canonical patch failed fresh git apply --check: {apply_check.stdout[-4000:]}")
        return {
            "valid": True,
            "error": "",
            "patch": patch,
            "label": label,
            "changed_paths": sorted(set(changed_paths)),
            "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        reset_repo(repo, base_commit)
        return {"valid": False, "error": str(exc), "patch": "", "label": label}


def generate_valid_plan(
    repo: Path,
    base_commit: str,
    problem: str,
    context: str,
    messages: list[dict[str, str]],
    seed: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    raw = call_llm(messages, max_tokens=2048, seed=seed)
    plan = parse_jsonish(raw["content"])
    result = canonicalize_plan(repo, base_commit, plan, label)
    attempts.append({"raw": raw, "plan": plan, "validation": result})
    if result["valid"]:
        return plan, result, attempts
    repair_raw = call_llm(
        prompt_repair(problem, context, plan, result["error"]),
        max_tokens=2048,
        seed=seed + 1000,
    )
    repaired_plan = parse_jsonish(repair_raw["content"])
    repaired = canonicalize_plan(repo, base_commit, repaired_plan, label)
    attempts.append({"raw": repair_raw, "plan": repaired_plan, "validation": repaired})
    return repaired_plan, repaired, attempts


def prediction(instance_id: str, mode: str, patch: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": f"Qwen3-14B-NEXUS-R2.4-{mode}",
        "model_patch": patch,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dataset_id = manifest["dataset"]
    token = os.environ.get("HF_TOKEN") or None
    dataset_sha = HfApi(token=token).dataset_info(dataset_id).sha
    dataset = load_dataset(dataset_id, split=manifest["split"], revision=dataset_sha, token=token)
    instance_id = manifest["instance_ids"][0]
    matches = [dict(row) for row in dataset if str(row["instance_id"]) == instance_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {instance_id}, got {len(matches)}")
    row = matches[0]

    problem = str(row["problem_statement"])
    repo_name = str(row["repo"])
    base_commit = str(row["base_commit"])
    repo = WORK / "repo"
    run(["git", "clone", "--filter=blob:none", f"https://github.com/{repo_name}.git", str(repo)])
    run(["git", "checkout", base_commit], cwd=repo)
    reset_repo(repo, base_commit)
    context, retrieval_audit = collect_context(repo, problem)

    seed = 24080417
    _, a_result, a_attempts = generate_valid_plan(
        repo, base_commit, problem, context, prompt_a(problem, context), seed + 1, "A"
    )
    _, b_result, b_attempts = generate_valid_plan(
        repo, base_commit, problem, context, prompt_b(problem, context), seed + 2, "B"
    )
    c_plan, c_proposal, c_attempts = generate_valid_plan(
        repo, base_commit, problem, context, prompt_b(problem, context), seed + 3, "C"
    )

    review_raw: dict[str, Any] = {"content": "", "usage": {}, "latency_s": 0.0, "skipped": True}
    review_obj: dict[str, Any] = {"accept": False, "defects": ["proposal invalid"], "corrected_edits": []}
    c_final = c_proposal
    c_decision = "invalid_proposal"
    if c_proposal["valid"]:
        review_raw = call_llm(prompt_review(problem, context, c_proposal["patch"]), max_tokens=1536, seed=seed + 4)
        review_obj = parse_jsonish(review_raw["content"])
        if bool(review_obj.get("accept", False)):
            c_decision = "review_accepted_proposal"
        else:
            corrected = {"analysis": "review correction", "edits": review_obj.get("corrected_edits", [])}
            corrected_result = canonicalize_plan(repo, base_commit, corrected, "C_reviewed")
            if corrected_result["valid"]:
                c_final = corrected_result
                c_plan = corrected
                c_decision = "review_correction_validated"
            else:
                c_decision = "invalid_review_correction_fallback_to_valid_proposal"
                c_attempts.append({"raw": review_raw, "plan": corrected, "validation": corrected_result})

    results = {"A": a_result, "B": b_result, "C": c_final}
    for mode, result in results.items():
        patch = result["patch"] if result["valid"] else ""
        (OUT / f"{mode}.patch").write_text(patch, encoding="utf-8")
        (OUT / f"predictions_{mode}.jsonl").write_text(
            json.dumps(prediction(instance_id, mode, patch)) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "format": "nexus-r24-swebench-generation/1",
        "dataset": dataset_id,
        "dataset_repository_sha": dataset_sha,
        "instance_id": instance_id,
        "repo": repo_name,
        "base_commit": base_commit,
        "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        "retrieval_audit": retrieval_audit,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "gold_fields_excluded_from_prompts": ["patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"],
        "patch_protocol": "exact edit plan -> deterministic application -> canonical git diff -> fresh git apply check",
        "results": results,
        "C_decision": c_decision,
        "raw": {
            "A_attempts": a_attempts,
            "B_attempts": b_attempts,
            "C_attempts": c_attempts,
            "C_review": review_raw,
            "C_review_parsed": review_obj,
            "C_final_plan": c_plan,
        },
    }
    (OUT / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "problem_statement.txt").write_text(problem, encoding="utf-8")
    (OUT / "retrieved_context.txt").write_text(context, encoding="utf-8")
    print(
        json.dumps(
            {
                "instance_id": instance_id,
                "valid": {mode: result["valid"] for mode, result in results.items()},
                "C_decision": c_decision,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
