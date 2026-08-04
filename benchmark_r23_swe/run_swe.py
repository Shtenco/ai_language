#!/usr/bin/env python3
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


def extract_diff(text: str) -> str:
    text = (text or "").replace("\r\n", "\n")
    fence = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    starts = [
        position
        for marker in ["diff --git ", "--- a/", "*** Begin Patch"]
        if (position := text.find(marker)) >= 0
    ]
    if starts:
        text = text[min(starts) :]
    if "*** Begin Patch" in text:
        text = text.replace("*** Begin Patch\n", "").replace("*** End Patch", "")
    return text.strip() + ("\n" if text.strip() else "")


def issue_terms(problem: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", problem):
        lowered = token.lower()
        if lowered not in STOPWORDS and lowered not in terms:
            terms.append(lowered)
    return terms[:24]


def collect_context(repo: Path, problem: str, max_chars: int = 12000) -> tuple[str, list[dict[str, Any]]]:
    files = run(["git", "ls-files", "*.py"], cwd=repo).stdout.splitlines()
    terms = issue_terms(problem)
    ranked: list[tuple[int, str, str]] = []
    for relative in files:
        path = repo / relative
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = text.lower()
        score = sum(min(lowered.count(term), 8) for term in terms)
        basename = Path(relative).stem.lower()
        score += sum(5 for term in terms if term in basename)
        if score > 0:
            ranked.append((score, relative, text))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    parts: list[str] = []
    audit: list[dict[str, Any]] = []
    used = 0
    for score, relative, text in ranked[:10]:
        lines = text.splitlines()
        hit_indices = [
            i for i, line in enumerate(lines) if any(term in line.lower() for term in terms)
        ]
        center = hit_indices[0] if hit_indices else 0
        start = max(0, center - 45)
        end = min(len(lines), center + 75)
        snippet = "\n".join(f"{i + 1:05d}: {lines[i]}" for i in range(start, end))
        block = f"\nFILE: {relative}\n{snippet}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 500:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block)
        audit.append({"path": relative, "score": score, "start_line": start + 1, "end_line": end})
    return "".join(parts), audit


def base_prompt(problem: str, context: str) -> str:
    return (
        "GitHub issue:\n"
        f"{problem}\n\n"
        "Retrieved repository context (line numbers are annotations, not part of files):\n"
        f"{context}\n\n"
        "Return a valid unified git diff against the checked-out base commit. Do not include explanations, "
        "Markdown fences, test output, or fabricated files. Modify only what is necessary."
    )


def prompt_a(problem: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are a software engineer in a closed repository task. Produce the patch directly. "
                "Never claim tests passed. Output unified diff only."
            ),
        },
        {"role": "user", "content": base_prompt(problem, context)},
    ]


def prompt_b(problem: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are NEXUS structured code solver. Internally localize the defect, infer invariants, "
                "trace callers, minimize behavioral surface, and design a regression-safe fix. Output only the final "
                "unified git diff, with no prose or fences."
            ),
        },
        {"role": "user", "content": base_prompt(problem, context)},
    ]


def prompt_proposal(problem: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are the NEXUS patch proposal engine. Produce a minimal plausible unified git diff. "
                "Output diff only."
            ),
        },
        {"role": "user", "content": base_prompt(problem, context)},
    ]


def prompt_critic(problem: str, context: str, proposal: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are an adversarial senior maintainer. Review the proposed patch against the issue and "
                "repository context. Identify concrete wrong files, missing cases, API regressions, invalid assumptions, "
                "and likely test failures. Do not output a patch. Return concise plain text audit."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Issue:\n{problem}\n\nContext:\n{context}\n\nProposed patch:\n{proposal}"
            ),
        },
    ]


def prompt_final(problem: str, context: str, proposal: str, critique: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are the NEXUS final code arbitrator. Repair the proposal only where the critique is "
                "grounded in the issue or repository context. Output one complete valid unified git diff only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Issue:\n{problem}\n\nContext:\n{context}\n\nProposal:\n{proposal}\n\nCritique:\n{critique}"
            ),
        },
    ]


def apply_check(repo: Path, patch: str, label: str) -> dict[str, Any]:
    patch_path = OUT / f"{label}.patch"
    patch_path.write_text(patch, encoding="utf-8")
    if not patch.strip():
        return {"valid": False, "output": "empty patch"}
    completed = run(["git", "apply", "--check", str(patch_path.resolve())], cwd=repo, check=False)
    return {"valid": completed.returncode == 0, "output": completed.stdout[-12000:]}


def prediction(instance_id: str, mode: str, patch: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": f"Qwen3-14B-NEXUS-R2.3-{mode}",
        "model_patch": patch,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dataset_id = manifest["dataset"]
    dataset_sha = HfApi().dataset_info(dataset_id).sha
    dataset = load_dataset(dataset_id, split=manifest["split"], revision=dataset_sha)
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
    run(["git", "reset", "--hard", base_commit], cwd=repo)
    context, retrieval_audit = collect_context(repo, problem)

    seed = 230804
    a_raw = call_llm(prompt_a(problem, context), max_tokens=1536, seed=seed + 1)
    a_patch = extract_diff(a_raw["content"])
    b_raw = call_llm(prompt_b(problem, context), max_tokens=1536, seed=seed + 2)
    b_patch = extract_diff(b_raw["content"])
    proposal_raw = call_llm(prompt_proposal(problem, context), max_tokens=1536, seed=seed + 3)
    proposal_patch = extract_diff(proposal_raw["content"])
    critique_raw = call_llm(
        prompt_critic(problem, context, proposal_patch), max_tokens=768, seed=seed + 4
    )
    final_raw = call_llm(
        prompt_final(problem, context, proposal_patch, critique_raw["content"]),
        max_tokens=1536,
        seed=seed + 5,
    )
    c_patch = extract_diff(final_raw["content"])

    checks = {
        "A": apply_check(repo, a_patch, "A"),
        "B": apply_check(repo, b_patch, "B"),
        "C": apply_check(repo, c_patch, "C"),
    }
    patches = {"A": a_patch, "B": b_patch, "C": c_patch}
    for mode, patch in patches.items():
        path = OUT / f"predictions_{mode}.jsonl"
        path.write_text(json.dumps(prediction(instance_id, mode, patch)) + "\n", encoding="utf-8")

    metadata = {
        "format": "nexus-r23-swebench-generation/1",
        "dataset": dataset_id,
        "dataset_repository_sha": dataset_sha,
        "instance_id": instance_id,
        "repo": repo_name,
        "base_commit": base_commit,
        "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        "retrieval_audit": retrieval_audit,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "gold_fields_excluded_from_prompts": ["patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"],
        "apply_checks": checks,
        "raw": {
            "A": a_raw,
            "B": b_raw,
            "C_proposal": proposal_raw,
            "C_critic": critique_raw,
            "C_final": final_raw,
        },
    }
    (OUT / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "problem_statement.txt").write_text(problem, encoding="utf-8")
    (OUT / "retrieved_context.txt").write_text(context, encoding="utf-8")
    print(json.dumps({"instance_id": instance_id, "apply_checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
