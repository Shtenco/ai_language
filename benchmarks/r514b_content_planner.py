#!/usr/bin/env python3
import collections, json, re, sys
from pathlib import Path
import torch
import torch.nn.functional as F
import r514_hierarchical_planner as p
import r57_concept_graph_language as base

WORD = re.compile(r'[А-Яа-яЁё][А-Яа-яЁё-]{3,}')
CORPUS_FREQ = collections.Counter()
FREQ_BLACK = set()

def init_freq():
    global CORPUS_FREQ, FREQ_BLACK
    path = sys.argv[sys.argv.index('--train-docs') + 1]
    for d in base.load_jsonl(path):
        CORPUS_FREQ.update(w.lower() for w in WORD.findall(d.get('text', '')))
    FREQ_BLACK = {w for w, _ in CORPUS_FREQ.most_common(256)} | {x.lower() for x in base.STOP}
    print('R514B words', len(CORPUS_FREQ), 'black', len(FREQ_BLACK), flush=True)

def content_vocab(tok):
    ids, pieces = [], {}
    for i in range(base.VOCAB):
        raw = tok.dec([i]).decode('utf-8', 'ignore')
        sp_piece = tok.sp.id_to_piece(i)
        s = raw.strip(); low = s.lower()
        if not sp_piece.startswith('▁'): continue
        if len(s) < 4 or not WORD.fullmatch(s): continue
        if low in FREQ_BLACK or CORPUS_FREQ[low] < 4: continue
        ids.append(i); pieces[i] = s
    if len(ids) < 300:
        raise RuntimeError('content candidate set too small: %d' % len(ids))
    print('R514B candidates', len(ids), [pieces[i] for i in ids[:50]], flush=True)
    return ids, pieces

def plan_loss(scores, examples, cand_pos, rng):
    losses = []
    C = scores.shape[1]
    for i, ex in enumerate(examples):
        if ex['meta'].get('kind') != 'main': continue
        gold = [cand_pos[t] for t in p.gold_anchors(ex, cand_pos) if t in cand_pos]
        if not gold: continue
        g = torch.tensor(sorted(set(gold)), dtype=torch.long)
        pos = scores[i, g]
        masked = scores[i].clone(); masked[g] = -1e9
        k = min(max(16, 2 * len(gold)), C - len(gold))
        neg = masked.topk(k).values
        losses.append(F.softplus(-pos).mean() + F.softplus(neg).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0

def main():
    init_freq()
    p.content_vocab = content_vocab
    p.plan_loss = plan_loss
    p.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    f = out / '00_R514_RESULTS.json'
    r = json.loads(f.read_text(encoding='utf-8'))
    r['format'] = 'nexus-r514b-content-planner/1'
    r['protocol']['content_filter'] = 'word-boundary piece; corpus frequency>=4; top256 frequent and STOP excluded'
    r['protocol']['plan_loss'] = 'multi-positive hard-negative logistic ranking'
    f.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
