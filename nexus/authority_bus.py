#!/usr/bin/env python3
"""Deterministic NEXUS Authority bus.

Exact tasks are solved outside the probabilistic language cortex. The bus
returns structured state, proof/provenance and a canonical Russian rendering.
If no exact task grammar matches, route() returns None and the surface-language
cortex remains sovereign over ordinary text continuation.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import deque

RULE_RE = re.compile(r'если\s+истинно\s+«([^»]+)»\s*,?\s*то\s+истинно\s+«([^»]+)»', re.I)
FACT_RE = re.compile(r'Факт:\s*истинно\s*«([^»]+)»', re.I)
CLAIM_RE = re.compile(r'следует\s+ли\s*,?\s*что\s*«([^»]+)»', re.I)
CYBER_RE = re.compile(
    r'Цель\s+регулятора:\s*(-?\d+(?:[.,]\d+)?)\.\s*'
    r'Текущее\s+значение:\s*(-?\d+(?:[.,]\d+)?)\.\s*'
    r'Допуск:\s*(\d+(?:[.,]\d+)?)\.', re.I)


def _num(s: str) -> float:
    return float(s.replace(',', '.'))


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return ('%.6f' % x).rstrip('0').rstrip('.')


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def solve_logic(text: str):
    rules = RULE_RE.findall(text)
    fm = FACT_RE.search(text)
    cm = CLAIM_RE.search(text)
    # A fact/claim pair is already a complete exact problem even with zero
    # implication rules: the claim follows iff it is the stated fact.
    if not fm or not cm:
        return None
    fact, claim = fm.group(1), cm.group(1)
    adj = {}
    for a, b in rules:
        adj.setdefault(a, []).append(b)

    parent = {fact: None}
    q = deque([fact])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in parent:
                parent[v] = u
                q.append(v)
    yes = claim in parent
    path = []
    if yes:
        cur = claim
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
    depth = max(0, len(path) - 1) if yes else None
    if yes:
        if depth == 0:
            answer = f'Да. «{claim}» дано непосредственно как истинный факт.'
        else:
            answer = f'Да. Следует «{claim}». Доказательство: ' + ' → '.join(f'«{x}»' for x in path) + '.'
    else:
        answer = f'Нет. Из факта «{fact}» по заданным правилам нельзя вывести «{claim}».'
    return {
        'handled': True,
        'kind': 'logic',
        'answer_yes': yes,
        'fact': fact,
        'claim': claim,
        'proof_path': path,
        'proof_depth': depth,
        'rules': [{'if': a, 'then': b} for a, b in rules],
        'canonical_response': answer,
        'provenance': {'solver': 'exact_forward_chaining_bfs', 'input_sha256': _digest(text)},
    }


def solve_cyber(text: str):
    m = CYBER_RE.search(text)
    if not m:
        return None
    target, current, tolerance = map(_num, m.groups())
    error = target - current
    if abs(error) <= tolerance:
        mode, action = 'стабильно', 'удерживать воздействие'
    elif error > 0:
        mode, action = 'коррекция', 'увеличить воздействие'
    else:
        mode, action = 'коррекция', 'уменьшить воздействие'
    answer = (
        f'Рассогласование равно {_fmt(error)}. Режим: {mode}. '
        f'Нужно {action} и затем снова измерить результат.'
    )
    return {
        'handled': True,
        'kind': 'cyber',
        'target': target,
        'current': current,
        'tolerance': tolerance,
        'error': error,
        'mode': mode,
        'action': action,
        'canonical_response': answer,
        'provenance': {'solver': 'exact_feedback_error_controller', 'input_sha256': _digest(text)},
    }


def route(text: str):
    # Exact grammars are intentionally narrow: uncertain/general language falls
    # through instead of pretending that a symbolic parser understood it.
    for solver in (solve_logic, solve_cyber):
        out = solver(text)
        if out is not None:
            return out
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('text', nargs='?')
    ap.add_argument('--json-file')
    a = ap.parse_args()
    text = a.text
    if a.json_file:
        obj = json.load(open(a.json_file, encoding='utf-8'))
        text = obj['text']
    if not text:
        raise SystemExit('provide text or --json-file')
    print(json.dumps(route(text), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
