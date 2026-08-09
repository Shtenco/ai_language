#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import r52_tokenizer_tournament as tt

VOCAB = 4096
D_MODEL = 192
HEADS = 6
LAYERS = 6
FF_DIM = 570
PARAM_TARGET = 3_000_000
PREFIX_TOKENS = 32
CONTEXT_TOKENS = 48
TARGET_TOKENS = 48
SEQ_TOKENS = PREFIX_TOKENS + CONTEXT_TOKENS + TARGET_TOKENS
SCREEN_STEPS = 384
SCREEN_BATCH = 8
LONG_STEPS = 4096
LONG_BATCH = 8
MODES = ('A_PLAIN', 'B_FLAT_RAG', 'C_CONCEPT_GRAPH', 'D_LOGIC_CYBER')

PROMPTS = [
    'Вечером он вышел из дома и',
    'Наука развивается потому, что',
    'Москва — это город, в котором',
    'Человек посмотрел в окно и сказал:',
    'Искусственный интеллект может помочь человеку',
    'Когда наступила весна,',
    'Если система получила сигнал, то',
    'Хорошее доказательство должно опираться на',
]

WORD_RE = re.compile(r'[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z-]{2,}', re.UNICODE)
CYR_RE = re.compile(r'[А-Яа-яЁё]')
CAP_RE = re.compile(r'(?<![.!?]\s)(?<!^)([А-ЯЁ][а-яё]{2,})')
STOP = {
    'это','как','что','чтобы','когда','если','или','для','при','над','под','без','его','её','она','они','оно','тот','эта','эти',
    'был','была','были','быть','есть','уже','ещё','очень','также','может','могут','будет','будут','который','которая','которые',
    'после','перед','между','через','только','всего','свой','свои','свою','своего','такой','такая','такие','более','менее','тоже',
    'весь','все','всех','один','одна','одни','него','нее','них','нас','вас','вам','нам','где','куда','почему','поэтому','потому',
    'и','а','но','не','ни','на','в','во','к','ко','из','от','до','по','за','с','со','у','о','об','про','же','ли','бы'
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def load_jsonl(path):
    out = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def keywords(text, limit=10):
    words = [w.lower().strip('-') for w in WORD_RE.findall(text)]
    words = [w for w in words if len(w) >= 4 and w not in STOP]
    c = collections.Counter(words)
    last = {w: i for i, w in enumerate(words)}
    ranked = sorted(c, key=lambda w: (c[w], last[w], len(w)), reverse=True)
    return ranked[:limit]


def infer_relation_goal(text):
    t = text.lower()[-500:]
    if 'потому что' in t:
        return 'причина', 'объяснить причину'
    if 'поэтому' in t or 'следовательно' in t:
        return 'следствие', 'сформулировать следствие'
    if re.search(r'\bесли\b', t):
        return 'условие', 'продолжить условную связь'
    if re.search(r'\bчтобы\b', t):
        return 'цель', 'объяснить цель действия'
    if re.search(r'\bкогда\b', t):
        return 'время', 'продолжить временную связь'
    if 'сказал:' in t or 'сказала:' in t or t.rstrip().endswith(':'):
        return 'речь', 'дать связное продолжение высказывания'
    if ' — это ' in t or ' называется ' in t:
        return 'определение', 'раскрыть понятие'
    if '?' in t[-120:]:
        return 'вопрос', 'дать прямой ответ'
    return 'продолжение', 'сохранить тему и логическую связность'


def rope(q, k):
    dh = q.shape[-1]
    pos = torch.arange(q.shape[-2], device=q.device, dtype=q.dtype)
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=q.device, dtype=q.dtype) / dh))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[None, None, :, :]
    sin = ang.sin()[None, None, :, :]

    def rot(x):
        xe, xo = x[..., 0::2], x[..., 1::2]
        y = torch.empty_like(x)
        y[..., 0::2] = xe * cos - xo * sin
        y[..., 1::2] = xe * sin + xo * cos
        return y
    return rot(q), rot(k)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, FF_DIM)
        self.fc2 = nn.Linear(FF_DIM, D_MODEL)

    def forward(self, x):
        b, l, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, l, 3, HEADS, d // HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = rope(q, k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, l, d))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = VOCAB
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class MemoryIndex:
    def __init__(self, docs, max_chunks=4500):
        self.chunks = []
        for d in docs:
            text = d.get('text', '')
            parts = [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+', text) if len(x.strip()) >= 80]
            buf = ''
            for p in parts:
                if len(buf) + len(p) + 1 <= 520:
                    buf = (buf + ' ' + p).strip()
                else:
                    if len(buf) >= 140:
                        self.chunks.append({'text': buf, 'source': d.get('source', ''), 'id': d.get('id', '')})
                    buf = p
                if len(self.chunks) >= max_chunks:
                    break
            if len(buf) >= 140 and len(self.chunks) < max_chunks:
                self.chunks.append({'text': buf, 'source': d.get('source', ''), 'id': d.get('id', '')})
            if len(self.chunks) >= max_chunks:
                break
        self.inv = collections.defaultdict(list)
        self.df = collections.Counter()
        for i, c in enumerate(self.chunks):
            ks = set(keywords(c['text'], 18))
            for w in ks:
                self.inv[w].append(i)
                self.df[w] += 1

    def retrieve(self, text):
        ks = keywords(text, 10)
        if not ks or not self.chunks:
            return None
        cand = collections.Counter()
        n = len(self.chunks)
        for w in ks:
            idf = math.log((n + 1) / (1 + self.df.get(w, 0))) + 1.0
            for i in self.inv.get(w, ())[:250]:
                cand[i] += idf
        if not cand:
            return None
        i, score = max(cand.items(), key=lambda z: (z[1], -z[0]))
        c = dict(self.chunks[i])
        c['score'] = score
        return c


def concept_prefix(context, retrieved=None, exact_meta=None, full=False):
    ks = keywords(context, 8)
    memks = keywords(retrieved['text'], 6) if retrieved else []
    rel, goal = infer_relation_goal(context)
    entities = CAP_RE.findall(context)[:4]
    parts = [
        'Граф понятий: ' + (', '.join(ks) if ks else 'общий контекст') + '.',
        'Связь: ' + rel + '.',
    ]
    if entities:
        parts.append('Сущности: ' + ', '.join(entities) + '.')
    if memks:
        parts.append('Память: ' + ', '.join(memks) + '.')
    if full:
        parts.append('Цель: ' + goal + '.')
        parts.append('Контракт: сохранить тему, не противоречить данным, завершить мысль.')
        if exact_meta and exact_meta.get('kind') == 'logic':
            status = 'доказано' if exact_meta['answer_yes'] else 'не доказано'
            parts.append(f"Логика: статус={status}; вывод={exact_meta['claim']}; глубина={exact_meta['depth']}.")
        elif exact_meta and exact_meta.get('kind') == 'cyber':
            parts.append(
                f"Кибернетика: цель={exact_meta['target']}; факт={exact_meta['current']}; "
                f"ошибка={exact_meta['error']}; допуск={exact_meta['tolerance']}; режим={exact_meta['mode']}; "
                f"действие={exact_meta['action']}."
            )
    return ' '.join(parts)


def prefix_ids(mode, tok, context_text, memory, meta):
    retrieved = memory.retrieve(context_text)
    if mode == 'A_PLAIN':
        text = ''
    elif mode == 'B_FLAT_RAG':
        text = 'Память: ' + (retrieved['text'][:240] if retrieved else 'нет релевантного фрагмента')
    elif mode == 'C_CONCEPT_GRAPH':
        text = concept_prefix(context_text, retrieved, meta, full=False)
    elif mode == 'D_LOGIC_CYBER':
        text = concept_prefix(context_text, retrieved, meta, full=True)
    else:
        raise ValueError(mode)
    ids = tok.enc(text.encode('utf-8')) if text else []
    ids = ids[-PREFIX_TOKENS:]
    return [0] * (PREFIX_TOKENS - len(ids)) + ids


TRAIN_PROPS = [
    'сигнал получен', 'проверка начата', 'доступ разрешён', 'архив проверен', 'модуль активен',
    'канал открыт', 'данные загружены', 'отчёт построен', 'датчик сработал', 'задача подтверждена',
    'контроль выполнен', 'выход доступен', 'процесс завершён', 'шлюз готов', 'сервис отвечает'
]
EVAL_PROPS = [
    'контрольная сумма верна', 'пакет принят', 'схема валидна', 'резерв создан', 'маршрут разрешён',
    'ключ подтверждён', 'сессия открыта', 'проверка подписи завершена', 'узел синхронизирован', 'команда исполнена',
    'состояние сохранено', 'канал подтверждён', 'объект доступен', 'режим стабилен', 'цель достигнута'
]


def make_logic_example(rng, train=True):
    pool = TRAIN_PROPS if train else EVAL_PROPS
    p1, p2, p3 = rng.sample(pool, 3)
    yes = rng.random() < 0.65
    if yes:
        fact = p1
        claim = p3
        depth = 2
        target = f'Да. Из факта «{p1}» и двух последовательных правил следует «{p3}».'
    else:
        fact = rng.choice([x for x in pool if x not in (p1, p2)])
        claim = p3
        depth = 0
        target = f'Нет. Из текущего факта нельзя вывести «{p3}»: необходимая посылка «{p1}» отсутствует.'
    context = (
        f'Правило 1: если истинно «{p1}», то истинно «{p2}». '
        f'Правило 2: если истинно «{p2}», то истинно «{p3}». '
        f'Факт: истинно «{fact}». Вопрос: следует ли, что «{claim}»?'
    )
    return context, target, {'kind': 'logic', 'answer_yes': yes, 'claim': claim, 'depth': depth, 'premise': p1}


def fmt_num(x):
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return ('%.3f' % x).rstrip('0').rstrip('.')


def make_cyber_example(rng):
    target = rng.randint(20, 100)
    delta = rng.choice([-20, -15, -10, -6, -3, 0, 2, 5, 8, 12, 18])
    current = target + delta
    tolerance = rng.choice([1, 2, 3])
    error = target - current
    if abs(error) <= tolerance:
        mode, action = 'стабильно', 'удерживать воздействие'
    elif error > 0:
        mode, action = 'коррекция', 'увеличить воздействие'
    else:
        mode, action = 'коррекция', 'уменьшить воздействие'
    context = (
        f'Цель регулятора: {target}. Текущее значение: {current}. Допуск: {tolerance}. '
        'Каково рассогласование и какое действие нужно выполнить?'
    )
    target_text = (
        f'Рассогласование равно {fmt_num(error)}. Режим: {mode}. Нужно {action} и затем снова измерить результат.'
    )
    meta = {
        'kind': 'cyber', 'target': target, 'current': current, 'tolerance': tolerance,
        'error': error, 'mode': mode, 'action': action,
    }
    return context, target_text, meta


def tokenize_docs(tok, docs):
    out = []
    for d in docs:
        ids = tok.enc(d['text'].encode('utf-8'))
        if len(ids) >= CONTEXT_TOKENS + TARGET_TOKENS + 4:
            out.append({'ids': ids, 'source': d.get('source', ''), 'id': d.get('id', '')})
    return out


def make_main_example(rng, docs, tok=None):
    d = rng.choice(docs)
    ids = d['ids']
    start = rng.randrange(0, len(ids) - CONTEXT_TOKENS - TARGET_TOKENS)
    ctx = ids[start:start + CONTEXT_TOKENS]
    tgt = ids[start + CONTEXT_TOKENS:start + CONTEXT_TOKENS + TARGET_TOKENS]
    context_text = tok.dec(ctx).decode('utf-8', 'replace')
    return {'ctx': ctx, 'tgt': tgt, 'ctx_text': context_text, 'meta': {'kind': 'main'}}


def make_aux_example(kind, rng, tok, train=True):
    if kind == 'logic':
        context, target, meta = make_logic_example(rng, train=train)
    else:
        context, target, meta = make_cyber_example(rng)
    ctx = tok.enc(context.encode('utf-8'))[-CONTEXT_TOKENS:]
    tgt = tok.enc(target.encode('utf-8'))[:TARGET_TOKENS]
    return {'ctx': ctx, 'tgt': tgt, 'ctx_text': context, 'meta': meta, 'target_text': target}


def sample_example(rng, docs, tok):
    u = rng.random()
    if u < 0.86:
        return make_main_example(rng, docs, tok)
    if u < 0.94:
        return make_aux_example('logic', rng, tok, train=True)
    return make_aux_example('cyber', rng, tok, train=True)


def pack(examples, mode, tok, memory):
    xs, ys, masks = [], [], []
    target_bytes = 0
    target_tokens = 0
    for ex in examples:
        pfx = prefix_ids(mode, tok, ex['ctx_text'], memory, ex['meta'])
        ctx = ex['ctx'][-CONTEXT_TOKENS:]
        ctxslot = [0] * (CONTEXT_TOKENS - len(ctx)) + list(ctx)
        tgt = list(ex['tgt'][:TARGET_TOKENS])
        target_bytes += max(1, len(tok.dec(tgt)))
        target_tokens += len(tgt)
        tgtslot = tgt + [0] * (TARGET_TOKENS - len(tgt))
        seq = pfx + ctxslot + tgtslot
        x, y = seq[:-1], seq[1:]
        mask = [0] * len(x)
        start = PREFIX_TOKENS + CONTEXT_TOKENS - 1
        for j in range(start, min(start + len(tgt), len(mask))):
            mask[j] = 1
        xs.append(x); ys.append(y); masks.append(mask)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
        target_bytes,
        target_tokens,
    )


def lr_factor(step, total):
    warm = max(32, total // 32)
    if step < warm:
        return max(0.05, (step + 1) / warm)
    q = (step - warm) / max(1, total - warm)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, q)))


def train_mode(mode, seed, tok, docs, memory, steps, batch, lr=1.2e-3):
    set_seed(seed)
    model = LM()
    p = param_count(model)
    assert p == 2_998_620, p
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = random.Random(seed + 77881)
    losses, target_bytes = [], 0
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        ex = [sample_example(rng, docs, tok) for _ in range(batch)]
        x, y, mask, tb, _ = pack(ex, mode, tok, memory)
        for pg in opt.param_groups:
            pg['lr'] = lr * lr_factor(step, steps)
        opt.zero_grad(set_to_none=True)
        z = model(x)
        ce = F.cross_entropy(z.reshape(-1, VOCAB), y.reshape(-1), reduction='none').view_as(y)
        loss = (ce * mask).sum() / tb
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        target_bytes += tb
        if (step + 1) % 256 == 0:
            print('TRAIN', mode, seed, step + 1, 'npb', sum(losses[-64:]) / min(64, len(losses)),
                  'target_MB', target_bytes / 1048576, flush=True)
    return model, {
        'mode': mode,
        'seed': seed,
        'params': p,
        'steps': steps,
        'batch': batch,
        'train_s': time.perf_counter() - t0,
        'target_bytes': target_bytes,
        'last64_nats_per_byte': sum(losses[-64:]) / min(64, len(losses)),
    }


@torch.no_grad()
def evaluate_examples(model, mode, examples, tok, memory, batch=8):
    model.eval()
    nll = 0.0
    bts = 0
    toks = 0
    correct = 0
    for i in range(0, len(examples), batch):
        raw_ex = examples[i:i + batch]
        real = len(raw_ex)
        ex = raw_ex if real == batch else raw_ex + [raw_ex[-1]] * (batch - real)
        x, y, mask, _, _ = pack(ex, mode, tok, memory)
        x, y, mask = x[:real], y[:real], mask[:real]
        tb = sum(max(1, len(tok.dec(e['tgt'][:TARGET_TOKENS]))) for e in ex[:real])
        z = model(x)
        ce = F.cross_entropy(z.reshape(-1, VOCAB), y.reshape(-1), reduction='none').view_as(y)
        nll += float((ce * mask).sum())
        bts += tb
        toks += int(mask.sum())
        correct += int(((z.argmax(-1) == y) & mask).sum())
    return {
        'bpb': nll / max(1, bts) / math.log(2),
        'nats_per_byte': nll / max(1, bts),
        'token_top1': correct / max(1, toks),
        'target_bytes': bts,
        'target_tokens': toks,
    }


def build_main_eval(tok, raw, seed, n=160):
    ids = tok.enc(raw)
    if len(ids) < CONTEXT_TOKENS + TARGET_TOKENS + 8:
        return []
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        start = rng.randrange(0, len(ids) - CONTEXT_TOKENS - TARGET_TOKENS)
        ctx = ids[start:start + CONTEXT_TOKENS]
        tgt = ids[start + CONTEXT_TOKENS:start + CONTEXT_TOKENS + TARGET_TOKENS]
        out.append({'ctx': ctx, 'tgt': tgt, 'ctx_text': tok.dec(ctx).decode('utf-8', 'replace'), 'meta': {'kind': 'main'}})
    return out


def build_aux_eval(tok, kind, seed, n=96):
    rng = random.Random(seed)
    return [make_aux_example(kind, rng, tok, train=False) for _ in range(n)]


def generation_input(mode, tok, memory, prompt, meta=None):
    pfx = prefix_ids(mode, tok, prompt, memory, meta or {'kind': 'main'})
    ctx = tok.enc(prompt.encode('utf-8'))[-CONTEXT_TOKENS:]
    ctxslot = [0] * (CONTEXT_TOKENS - len(ctx)) + ctx
    return pfx + ctxslot


@torch.no_grad()
def generate(model, mode, tok, memory, prompt, seed, meta=None, sample=True, max_new=64, temp=0.78, topk=40):
    set_seed(seed)
    model.eval()
    ids = generation_input(mode, tok, memory, prompt, meta)
    generated = []
    for _ in range(max_new):
        x = torch.tensor([ids[-160:]], dtype=torch.long)
        log = model(x)[0, -1]
        if sample:
            q = log / max(1e-5, temp)
            v, ix = torch.topk(q, min(topk, len(q)))
            p = F.softmax(v, dim=-1)
            nxt = int(ix[torch.multinomial(p, 1)])
        else:
            nxt = int(log.argmax())
        ids.append(nxt)
        generated.append(nxt)
    return tok.dec(generated).decode('utf-8', 'replace')


def text_metrics(s):
    letters = re.findall(r'[A-Za-zА-Яа-яЁё]', s)
    cyr = CYR_RE.findall(s)
    words = [w.lower() for w in re.findall(r'[А-Яа-яЁё]+', s)]
    tri = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    repeat3 = 1.0 - len(set(tri)) / max(1, len(tri))
    dash = s.count('—') + s.count('–')
    endings = len(re.findall(r'[.!?]', s))
    return {
        'chars': len(s),
        'words': len(words),
        'cyrillic_letter_share': len(cyr) / max(1, len(letters)),
        'unique_word_ratio': len(set(words)) / max(1, len(words)),
        'repeated_trigram_rate': repeat3,
        'dash_per_100_chars': 100 * dash / max(1, len(s)),
        'sentence_endings_per_100_chars': 100 * endings / max(1, len(s)),
        'replacement_chars': s.count('�'),
    }


def generation_suite(model, mode, tok, memory, seed):
    rows = []
    for i, prompt in enumerate(PROMPTS):
        for sample in (False, True):
            cont = generate(model, mode, tok, memory, prompt, seed + i * 101 + int(sample), sample=sample)
            rows.append({'prompt': prompt, 'decode': 'sample' if sample else 'greedy', 'continuation': cont, **text_metrics(cont)})
    return rows


def aux_generation_accuracy(model, mode, tok, memory, kind, seed, n=32):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        context, target, meta = make_logic_example(rng, train=False) if kind == 'logic' else make_cyber_example(rng)
        cont = generate(model, mode, tok, memory, context, seed + i * 17, meta=meta, sample=False, max_new=32)
        low = cont.lower()
        if kind == 'logic':
            yes_ok = (('да' in low[:40]) if meta['answer_yes'] else ('нет' in low[:50] or 'недостат' in low))
            claim_ok = meta['claim'].lower() in low
            ok = yes_ok and claim_ok
        else:
            err_ok = fmt_num(meta['error']) in low
            action_word = 'увелич' if meta['error'] > meta['tolerance'] else ('уменьш' if meta['error'] < -meta['tolerance'] else 'удерж')
            action_ok = action_word in low
            ok = err_ok and action_ok
        rows.append({'context': context, 'target': target, 'generated': cont, 'ok': bool(ok), **meta})
    return {'accuracy': sum(r['ok'] for r in rows) / len(rows), 'rows': rows}


@torch.no_grad()
def runtime(model, mode, tok, memory, docs, seed, reps=20):
    rng = random.Random(seed)
    ex = [make_main_example(rng, docs, tok) for _ in range(8)]
    x, _, _, _, _ = pack(ex, mode, tok, memory)
    model.eval()
    for _ in range(3):
        model(x)
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter(); model(x); vals.append(time.perf_counter() - t0)
    vals.sort()
    med = vals[len(vals) // 2]
    return {'median_s': med, 'model_tokens_per_s': x.numel() / med, 'seq_len': x.shape[1]}


def flatten(prefix, d):
    return {prefix + k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-text', required=True)
    ap.add_argument('--train-docs', required=True)
    ap.add_argument('--memory-docs', required=True)
    ap.add_argument('--tests', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--screen-seeds', default='11,29')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw_train = Path(a.train_text).read_bytes()
    train_docs_raw = load_jsonl(a.train_docs)
    memory_docs = load_jsonl(a.memory_docs)
    tests = {Path(p).stem: Path(p).read_bytes() for p in a.tests if Path(p).exists() and Path(p).stat().st_size > 0}
    seeds = [int(x) for x in a.screen_seeds.split(',')]

    tok = tt.train_sp(raw_train, VOCAB, 'unigram', out)
    probe = 'Проверка lossless: русский текст №123 — без потерь.\n'.encode('utf-8')
    assert tok.dec(tok.enc(probe)) == probe

    model_probe = LM()
    params = param_count(model_probe)
    assert params == 2_998_620, params
    del model_probe

    train_docs = tokenize_docs(tok, train_docs_raw)
    if len(train_docs) < 50:
        raise RuntimeError(f'too few eligible training documents: {len(train_docs)}')
    memory = MemoryIndex(memory_docs)
    if len(memory.chunks) < 100:
        raise RuntimeError(f'too few graph-memory chunks: {len(memory.chunks)}')

    eval_main = {name: build_main_eval(tok, raw, 7000 + i * 31, n=128) for i, (name, raw) in enumerate(tests.items())}
    eval_logic = build_aux_eval(tok, 'logic', 88001, n=96)
    eval_cyber = build_aux_eval(tok, 'cyber', 99001, n=96)

    screen_rows = []
    for seed in seeds:
        for mode in MODES:
            print('SCREEN', mode, seed, flush=True)
            model, tr = train_mode(mode, seed, tok, train_docs, memory, SCREEN_STEPS, SCREEN_BATCH)
            row = {**tr}
            for name, ex in eval_main.items():
                row.update(flatten(name + '_', evaluate_examples(model, mode, ex, tok, memory)))
            row.update(flatten('logic_', evaluate_examples(model, mode, eval_logic, tok, memory)))
            row.update(flatten('cyber_', evaluate_examples(model, mode, eval_cyber, tok, memory)))
            row.update(flatten('runtime_', runtime(model, mode, tok, memory, train_docs, seed, reps=12)))
            gens = generation_suite(model, mode, tok, memory, seed + 333)
            samples = [g for g in gens if g['decode'] == 'sample']
            row['gen_repeat3_mean'] = sum(g['repeated_trigram_rate'] for g in samples) / len(samples)
            row['gen_dash100_mean'] = sum(g['dash_per_100_chars'] for g in samples) / len(samples)
            row['gen_cyr_mean'] = sum(g['cyrillic_letter_share'] for g in samples) / len(samples)
            row['gen_sentence100_mean'] = sum(g['sentence_endings_per_100_chars'] for g in samples) / len(samples)
            screen_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    screen_agg = []
    main_bpb_cols = [f'{name}_bpb' for name in eval_main]
    for mode in MODES:
        rr = [r for r in screen_rows if r['mode'] == mode]
        z = {'mode': mode, 'n_seeds': len(rr), 'params': params}
        for k in main_bpb_cols + ['logic_bpb','cyber_bpb','runtime_model_tokens_per_s','gen_repeat3_mean','gen_dash100_mean','gen_cyr_mean','gen_sentence100_mean','train_s']:
            vals = [float(r[k]) for r in rr]
            mu = sum(vals) / len(vals)
            z[k + '_mean'] = mu
            z[k + '_sd'] = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        z['mean_main_bpb'] = sum(z[k + '_mean'] for k in main_bpb_cols) / max(1, len(main_bpb_cols))
        screen_agg.append(z)

    final_seed = 20260810
    long_results = {}
    for mode in ('A_PLAIN', 'D_LOGIC_CYBER'):
        print('LONG', mode, final_seed, flush=True)
        model, tr = train_mode(mode, final_seed, tok, train_docs, memory, LONG_STEPS, LONG_BATCH, lr=1.0e-3)
        r = {'training': tr, 'eval': {}, 'generation': generation_suite(model, mode, tok, memory, final_seed + 501)}
        for name, ex in eval_main.items():
            r['eval'][name] = evaluate_examples(model, mode, ex, tok, memory)
        r['eval']['logic_teacher_forced'] = evaluate_examples(model, mode, eval_logic, tok, memory)
        r['eval']['cyber_teacher_forced'] = evaluate_examples(model, mode, eval_cyber, tok, memory)
        r['logic_generation'] = aux_generation_accuracy(model, mode, tok, memory, 'logic', 123451, n=32)
        r['cyber_generation'] = aux_generation_accuracy(model, mode, tok, memory, 'cyber', 223451, n=32)
        r['runtime'] = runtime(model, mode, tok, memory, train_docs, final_seed, reps=20)
        long_results[mode] = r
        torch.save({
            'state_dict': model.state_dict(), 'mode': mode, 'seed': final_seed,
            'config': {'vocab': VOCAB, 'd_model': D_MODEL, 'heads': HEADS, 'layers': LAYERS, 'ff': FF_DIM,
                       'prefix_tokens': PREFIX_TOKENS, 'context_tokens': CONTEXT_TOKENS, 'target_tokens': TARGET_TOKENS,
                       'rope': True, 'params': params},
        }, out / f'R57_{mode}_3M.pt')

    mem_snapshot = [{'source': c['source'], 'id': c['id'], 'text': c['text']} for c in memory.chunks[:4500]]
    (out / 'R57_GRAPH_MEMORY_SNAPSHOT.json').write_text(json.dumps(mem_snapshot, ensure_ascii=False), encoding='utf-8')

    result = {
        'format': 'nexus-r57-concept-graph-language/1',
        'protocol': {
            'trainable_params': params,
            'parameter_target': PARAM_TARGET,
            'vocab': VOCAB,
            'tokenizer': 'lossless SentencePiece Unigram4096 + byte fallback',
            'decoder': {'d_model': D_MODEL, 'heads': HEADS, 'layers': LAYERS, 'ff': FF_DIM, 'rope': True, 'attention': 'dense causal SDPA'},
            'fixed_training_geometry': {'prefix_tokens': PREFIX_TOKENS, 'context_tokens': CONTEXT_TOKENS, 'target_tokens': TARGET_TOKENS, 'sequence_tokens': SEQ_TOKENS},
            'screen': {'steps': SCREEN_STEPS, 'batch': SCREEN_BATCH, 'seeds': seeds},
            'long': {'steps': LONG_STEPS, 'batch': LONG_BATCH, 'seed': final_seed, 'modes': ['A_PLAIN','D_LOGIC_CYBER']},
            'task_mix': {'main_russian_continuation': 0.86, 'formal_logic': 0.08, 'cybernetic_control': 0.06},
            'modes': {
                'A_PLAIN': 'same cortex + neutral fixed prefix slot',
                'B_FLAT_RAG': 'same cortex + raw disjoint-memory retrieval',
                'C_CONCEPT_GRAPH': 'same cortex + typed concept/relation memory serialization',
                'D_LOGIC_CYBER': 'same cortex + concept graph + goal/contract + exact logic/cyber state',
            },
            'authority_boundary': 'exact logic/cyber state is external deterministic working state; neural decoder verbalizes it',
        },
        'data': {'train_docs_eligible': len(train_docs), 'memory_chunks': len(memory.chunks), 'test_splits': list(eval_main)},
        'screen_per_seed': screen_rows,
        'screen_aggregate': screen_agg,
        'long_results': long_results,
    }
    (out / '00_R57_RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    if screen_rows:
        keys = []
        for r in screen_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with (out / '01_SCREEN_PER_SEED.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(screen_rows)
    with (out / '02_SCREEN_AGGREGATE.csv').open('w', newline='', encoding='utf-8') as f:
        keys = []
        for r in screen_agg:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(screen_agg)

    lines = []
    for mode, r in long_results.items():
        lines.append(f'===== {mode} =====')
        for g in r['generation']:
            lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')}, ensure_ascii=False)}")
        lines.append(f"LOGIC_GENERATION_ACCURACY={r['logic_generation']['accuracy']:.4f}")
        lines.append(f"CYBER_GENERATION_ACCURACY={r['cyber_generation']['accuracy']:.4f}")
    (out / '03_RAW_RUSSIAN_GENERATIONS.txt').write_text('\n\n'.join(lines), encoding='utf-8')

    (out / 'README_RU.md').write_text(
        '# NEXUS R5.7 Concept-Graph Language AI\n\n'
        'Контролируемый A/B/C/D эксперимент: одна и та же 2,998,620-параметрическая 6-слойная RoPE-кора получает либо нейтральный prefix, либо flat RAG, либо граф понятий, либо граф понятий + формальную логику + кибернетическое рабочее состояние. '
        'Все режимы используют один lossless Unigram4096 tokenizer, одинаковые target-последовательности и matched seeds. После screening выполняется отдельное длинное обучение A и D с фиксированным seed 20260810. '
        'Raw greedy/sample Russian continuations, teacher-forced BPB, formal-logic generation and cybernetic-control generation сохраняются без внешнего LLM judge.\n',
        encoding='utf-8'
    )
    print('DONE', json.dumps({'params': params, 'train_docs': len(train_docs), 'memory_chunks': len(memory.chunks), 'screen_aggregate': screen_agg}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
