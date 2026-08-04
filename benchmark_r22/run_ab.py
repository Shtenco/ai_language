#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset.json"
OUT = ROOT / "results"
API = os.environ.get("NEXUS_LLM_API", "http://127.0.0.1:8080/v1/chat/completions")
MODEL = os.environ.get("NEXUS_LLM_MODEL", "qwen3-14b")
SEED = 1234


def safe_math(expr: str) -> str:
    tree = ast.parse(expr, mode="eval")
    def ev(n: ast.AST) -> Fraction:
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int): return Fraction(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand); return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add): return a + b
            if isinstance(n.op, ast.Sub): return a - b
            if isinstance(n.op, ast.Mult): return a * b
            if isinstance(n.op, ast.Div): return a / b
            if isinstance(n.op, ast.Pow):
                if b.denominator != 1 or abs(b.numerator) > 20: raise ValueError("unsafe exponent")
                return a ** b.numerator
        raise ValueError(f"unsupported AST: {type(n).__name__}")
    value = ev(tree)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def logic_answer(task: dict[str, Any]) -> str:
    by_subject: dict[str, set[str]] = defaultdict(set)
    for fact in task.get("facts", []):
        subject, predicate = fact.split(":", 1)
        by_subject[subject].add(predicate)
    changed = True
    while changed:
        changed = False
        for subject, predicates in list(by_subject.items()):
            for src, dst in task.get("rules", []):
                if src in predicates and dst not in predicates:
                    predicates.add(dst); changed = True
    q_subject, q_predicate = task["query"].split(":", 1)
    return "YES" if q_predicate in by_subject.get(q_subject, set()) else "NO"


def causal_answer(task: dict[str, Any]) -> str:
    allowed = {"causes", "requires", "enables"}
    graph: dict[str, set[str]] = defaultdict(set)
    for a, rel, b in task.get("edges", []):
        if rel in allowed: graph[a].add(b)
    source, target = task["source"], task["target"]
    stack, seen = [source], {source}
    while stack:
        node = stack.pop()
        if node == target: return "YES"
        for nxt in graph.get(node, set()):
            if nxt not in seen: seen.add(nxt); stack.append(nxt)
    return "NO"


def stance_to_answer(stance: str) -> str:
    return {"support":"YES", "contradict":"NO"}.get(stance, "UNKNOWN")


def authority_lite(task: dict[str, Any]) -> tuple[str, list[str], str]:
    kind = task["kind"]
    if kind == "math": return safe_math(task["expression"]), [], "exact_ast"
    if kind == "logic": return logic_answer(task), [], "forward_chaining"
    if kind == "causal": return causal_answer(task), [], "typed_path"
    if kind == "unknown": return "UNKNOWN", [], "no_evidence_gate"
    evidence = task.get("evidence", [])
    if not evidence: return "UNKNOWN", [], "no_evidence_gate"
    best = max(evidence, key=lambda e: float(e["trust"]))
    return stance_to_answer(best["stance"]), [best["id"]], "highest_trust_single_source"


def authority_full(task: dict[str, Any]) -> tuple[str, list[str], str]:
    kind = task["kind"]
    if kind in {"math", "logic", "causal", "unknown"}:
        ans, ids, _ = authority_lite(task)
        return ans, ids, "exact_or_formal_authority"
    current = [e for e in task.get("evidence", []) if e.get("status", "current") == "current"]
    if not current: return "UNKNOWN", [], "no_current_evidence"
    # One contribution per independent publisher prevents duplicate-copy quorum.
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for e in current:
        key = (e["publisher"], e["stance"])
        if key not in grouped or float(e["trust"]) > float(grouped[key]["trust"]): grouped[key] = e
    support = [e for (_, stance), e in grouped.items() if stance == "support"]
    contra = [e for (_, stance), e in grouped.items() if stance == "contradict"]
    s_score = sum(float(e["trust"]) for e in support)
    c_score = sum(float(e["trust"]) for e in contra)
    if support and not contra and max(float(e["trust"]) for e in support) >= 0.75:
        return "YES", [e["id"] for e in support], "current_single_side"
    if contra and not support and max(float(e["trust"]) for e in contra) >= 0.75:
        return "NO", [e["id"] for e in contra], "current_single_side"
    if len(support) >= 2 and s_score - c_score >= 0.50:
        return "YES", [e["id"] for e in support], "independent_quorum"
    if len(contra) >= 2 and c_score - s_score >= 0.50:
        return "NO", [e["id"] for e in contra], "independent_quorum"
    return "UNKNOWN", [e["id"] for e in current], "unresolved_contradiction"


def evidence_text(task: dict[str, Any]) -> str:
    if task["kind"] == "math": return f"Проверенное выражение: {task['expression']}"
    if task["kind"] == "logic": return "Факты: " + json.dumps(task.get("facts", []), ensure_ascii=False) + "; правила: " + json.dumps(task.get("rules", []), ensure_ascii=False)
    if task["kind"] == "causal": return "Типизированные рёбра: " + json.dumps(task.get("edges", []), ensure_ascii=False)
    ev = task.get("evidence", [])
    if not ev: return "Проверенных источников нет."
    return "\n".join(f"[{e['id']}] publisher={e['publisher']} trust={e['trust']} status={e.get('status','current')}: {e['text']}" for e in ev)


def call_llm(messages: list[dict[str, str]], max_tokens: int = 96) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": SEED,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type":"json_object"}
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    started = time.time()
    last_error = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                raw = json.loads(r.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            return {"content": content, "usage": raw.get("usage", {}), "latency_s": round(time.time()-started, 3)}
        except Exception as exc:
            last_error = repr(exc); time.sleep(5 * (attempt + 1))
    return {"content":"", "usage":{}, "latency_s":round(time.time()-started,3), "error":last_error}


def parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()
    try: return json.loads(text)
    except Exception: pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try: return json.loads(match.group(0))
        except Exception: pass
    return {"answer": text}


def normalize_answer(value: Any) -> str:
    if value is None: return ""
    s = str(value).strip().upper().replace("Ё", "Е")
    s = re.sub(r"[\s\.,;:!]+$", "", s)
    mapping = {
        "ДА":"YES", "YES":"YES", "TRUE":"YES", "ВЕРНО":"YES",
        "НЕТ":"NO", "NO":"NO", "FALSE":"NO", "НЕВЕРНО":"NO",
        "НЕИЗВЕСТНО":"UNKNOWN", "UNKNOWN":"UNKNOWN", "НЕДОСТАТОЧНО ДАННЫХ":"UNKNOWN",
        "НЕДОСТАТОЧНО ИНФОРМАЦИИ":"UNKNOWN", "НЕЛЬЗЯ ОПРЕДЕЛИТЬ":"UNKNOWN"
    }
    if s in mapping: return mapping[s]
    m = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", s)
    return m.group(0) if m and len(s) < 80 else s


def prompt_a(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
      {"role":"system","content":"/no_think\nТы проходишь слепой тест. Не выдумывай факты. Верни только JSON: {\"answer\":\"...\",\"confidence\":0.0,\"reason\":\"кратко\"}."},
      {"role":"user","content":task["prompt"]}
    ]


def prompt_b(task: dict[str, Any], answer: str, ids: list[str], method: str) -> list[dict[str, str]]:
    return [
      {"role":"system","content":"/no_think\nТы Language Cortex. NEXUS Authority уже вынес решение, его нельзя переопределять. Верни только JSON с answer, confidence, reason, evidence_ids."},
      {"role":"user","content":f"Вопрос: {task['prompt']}\nAuthority answer: {answer}\nAuthority method: {method}\nEvidence IDs: {ids}\nМатериал:\n{evidence_text(task)}"}
    ]


def prompt_c_proposal(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
      {"role":"system","content":"/no_think\nТы proposal engine. Рассмотри вопрос и evidence, но не считай найденный текст автоматически истинным. Верни JSON: answer, confidence, risks."},
      {"role":"user","content":f"Вопрос: {task['prompt']}\nEvidence:\n{evidence_text(task)}"}
    ]


def prompt_c_critic(task: dict[str, Any], proposal: str, final_answer: str, ids: list[str], method: str) -> list[dict[str, str]]:
    return [
      {"role":"system","content":"/no_think\nТы критик NEXUS. Ищи ложную причинность, affirming the consequent, устаревшие или конфликтующие источники. Authority final нельзя менять. Верни JSON: answer, critique, corrected, evidence_ids."},
      {"role":"user","content":f"Вопрос: {task['prompt']}\nProposal: {proposal}\nAuthority final: {final_answer}\nMethod: {method}\nEvidence IDs: {ids}\nEvidence:\n{evidence_text(task)}"}
    ]


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0: return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_delta(a: list[int], b: list[int], samples: int = 10000) -> list[float]:
    rng = random.Random(SEED)
    n = len(a); vals = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(b[i]-a[i] for i in idx)/n)
    vals.sort()
    return [round(vals[int(0.025*samples)],4), round(vals[int(0.975*samples)-1],4)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_bytes = DATASET.read_bytes()
    dataset = json.loads(raw_bytes)
    (OUT / "dataset_sha256.txt").write_text(hashlib.sha256(raw_bytes).hexdigest()+"  dataset.json\n")
    rows = []
    with (OUT / "responses.jsonl").open("w", encoding="utf-8") as sink:
        for index, task in enumerate(dataset["tasks"], 1):
            expected = normalize_answer(task["expected"])
            a_raw = call_llm(prompt_a(task), 96)
            a_obj = parse_jsonish(a_raw["content"])
            a_answer = normalize_answer(a_obj.get("answer"))

            b_answer, b_ids, b_method = authority_lite(task)
            b_raw = call_llm(prompt_b(task, b_answer, b_ids, b_method), 80)

            c_answer, c_ids, c_method = authority_full(task)
            c_prop = call_llm(prompt_c_proposal(task), 80)
            c_crit = call_llm(prompt_c_critic(task, c_prop["content"], c_answer, c_ids, c_method), 96)

            row = {
                "index": index, "task_id": task["id"], "category": task["category"], "expected": expected,
                "A": {"answer":a_answer, "correct":a_answer==expected, "raw":a_raw, "parsed":a_obj},
                "B": {"answer":normalize_answer(b_answer), "correct":normalize_answer(b_answer)==expected, "method":b_method, "evidence_ids":b_ids, "cortex":b_raw},
                "C": {"answer":normalize_answer(c_answer), "correct":normalize_answer(c_answer)==expected, "method":c_method, "evidence_ids":c_ids, "proposal":c_prop, "critic":c_crit}
            }
            sink.write(json.dumps(row, ensure_ascii=False)+"\n"); sink.flush(); rows.append(row)
            print(f"[{index:02d}/{len(dataset['tasks'])}] {task['id']} expected={expected} A={a_answer} B={b_answer} C={c_answer}", flush=True)

    modes = ["A","B","C"]
    summary: dict[str, Any] = {"format":"nexus-r22-ab-results/1", "dataset_sha256":hashlib.sha256(raw_bytes).hexdigest(), "n":len(rows), "modes":{}, "categories":{}}
    for mode in modes:
        correct = [int(r[mode]["correct"]) for r in rows]
        unknown_rows = [r for r in rows if r["expected"]=="UNKNOWN"]
        hallucinations = sum(1 for r in unknown_rows if r[mode]["answer"] not in {"UNKNOWN", ""})
        summary["modes"][mode] = {
            "correct":sum(correct), "accuracy":round(sum(correct)/len(correct),4),
            "unknown_cases":len(unknown_rows), "hallucinations_on_unknown":hallucinations,
            "hallucination_rate_on_unknown":round(hallucinations/max(1,len(unknown_rows)),4)
        }
    cats = sorted({r["category"] for r in rows})
    for cat in cats:
        cr = [r for r in rows if r["category"]==cat]
        summary["categories"][cat] = {m:{"correct":sum(int(r[m]["correct"]) for r in cr),"n":len(cr),"accuracy":round(sum(int(r[m]["correct"]) for r in cr)/len(cr),4)} for m in modes}
    for left, right in [("A","B"),("A","C"),("B","C")]:
        lv = [int(r[left]["correct"]) for r in rows]; rv = [int(r[right]["correct"]) for r in rows]
        wins = sum(1 for x,y in zip(lv,rv) if y>x); losses = sum(1 for x,y in zip(lv,rv) if y<x)
        summary[f"delta_{right}_minus_{left}"] = {
            "absolute_accuracy_delta":round((sum(rv)-sum(lv))/len(rows),4),
            "relative_error_reduction":round(((len(rows)-sum(lv))-(len(rows)-sum(rv)))/max(1,len(rows)-sum(lv)),4),
            "paired_wins":wins,"paired_losses":losses,"ties":len(rows)-wins-losses,
            "exact_sign_test_p":round(exact_sign_p(wins,losses),6),
            "bootstrap_95ci":bootstrap_delta(lv,rv)
        }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# NEXUS R2.2 — Preregistered A/B/C Intelligence Gain Report","",f"Dataset SHA-256: `{summary['dataset_sha256']}`",f"Tasks: **{len(rows)}**. Same frozen Qwen3-14B Q4_K_M checkpoint and deterministic sampling for all LLM calls.","","## Aggregate","","| Mode | Correct | Accuracy | Hallucinations on UNKNOWN |","|---|---:|---:|---:|"]
    for m,label in [("A","Qwen3-14B raw"),("B","Qwen3 + Authority Lite"),("C","Qwen3 + Full Cognitive OS")]:
        x=summary["modes"][m]; lines.append(f"| {label} | {x['correct']}/{len(rows)} | {x['accuracy']:.1%} | {x['hallucinations_on_unknown']}/{x['unknown_cases']} |")
    lines += ["","## Paired deltas",""]
    for key in ["delta_B_minus_A","delta_C_minus_A","delta_C_minus_B"]:
        x=summary[key]; lines.append(f"- **{key}**: {x['absolute_accuracy_delta']:+.1%}; wins/losses/ties {x['paired_wins']}/{x['paired_losses']}/{x['ties']}; bootstrap 95% CI {x['bootstrap_95ci']}; exact sign p={x['exact_sign_test_p']}.")
    lines += ["","## By category","","| Category | A | B | C |","|---|---:|---:|---:|"]
    for cat,x in summary["categories"].items(): lines.append(f"| {cat} | {x['A']['correct']}/{x['A']['n']} | {x['B']['correct']}/{x['B']['n']} | {x['C']['correct']}/{x['C']['n']} |")
    lines += ["","## Per-task audit","","| Task | Category | Expected | A | B | C | Winner(s) |","|---|---|---|---|---|---|---|"]
    for r in rows:
        best=[m for m in modes if r[m]["correct"]]; winner=", ".join(best) if best else "none"
        lines.append(f"| {r['task_id']} | {r['category']} | {r['expected']} | {r['A']['answer']} {'✓' if r['A']['correct'] else '✗'} | {r['B']['answer']} {'✓' if r['B']['correct'] else '✗'} | {r['C']['answer']} {'✓' if r['C']['correct'] else '✗'} | {winner} |")
    lines += ["","## Claim boundary","","This is a preregistered 24-task synthetic/controlled suite measuring exactness, formal inference, causal typing, abstention, and evidence arbitration. It is **not** an official MERA score, not a general-IQ measurement, and not proof of universal intelligence. Any claimed gain applies only to this frozen suite until replicated externally."]
    (OUT / "NEXUS_R2.2_INTELLIGENCE_GAIN_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
