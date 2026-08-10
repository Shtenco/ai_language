#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

import torch
import r57_concept_graph_language as base

MODE = 'D_LOGIC_CYBER'
BATCH = 8
SEED = 20260810

# Keep exact logic/cyber organs alive, but make this overwhelmingly a surface-language run.
def deep_sample(rng, docs, tok):
    u = rng.random()
    if u < 0.94:
        return base.make_main_example(rng, docs, tok)
    if u < 0.97:
        return base.make_aux_example('logic', rng, tok, train=True)
    return base.make_aux_example('cyber', rng, tok, train=True)

base.sample_example = deep_sample

PROMPTS = [
    'Вечером он вышел из дома и',
    'Наука развивается потому, что',
    'Москва — это город, в котором',
    'Человек посмотрел в окно и сказал:',
    'Искусственный интеллект может помочь человеку',
    'Когда наступила весна,',
    'Если система получила сигнал, то',
    'Хорошее доказательство должно опираться на',
    'Современная компьютерная система обрабатывает данные и',
    'Если эксперимент не подтвердил гипотезу, исследователь должен',
    'Причинная связь отличается от корреляции тем, что',
    'Память системы хранит факты, чтобы',
    'Когда программа обнаружила противоречие, она',
    'Чтобы проверить утверждение, необходимо',
    'После ошибки система изменила своё состояние и',
    'В городе открыли новый научный центр, где',
]

HIST_RE = re.compile(r'\b(?:17|18|19)\d{2}\b|Пушкин|Булгарин|Петербург|С.-Петербург|Вяземск|Карамзин', re.I)
WORD_RE = re.compile(r'[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z-]{2,}', re.UNICODE)


def extra_generation_metrics(gens):
    text = '\n'.join(g.get('continuation', '') for g in gens)
    words = WORD_RE.findall(text)
    return {
        'historic_markers': len(HIST_RE.findall(text)),
        'historic_markers_per_1000_chars': 1000.0 * len(HIST_RE.findall(text)) / max(1, len(text)),
        'all_generation_chars': len(text),
        'all_generation_words': len(words),
        'distinct_word_ratio': len(set(w.lower() for w in words)) / max(1, len(words)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-text', required=True)
    ap.add_argument('--train-docs', required=True)
    ap.add_argument('--memory-docs', required=True)
    ap.add_argument('--tests', nargs='+', required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.train_text).read_bytes()
    tok = base.tt.train_sp(raw, base.VOCAB, 'unigram', out)
    probe = 'Глубокое разнообразное обучение NEXUS №514.\n'.encode('utf-8')
    assert tok.dec(tok.enc(probe)) == probe

    docs = base.tokenize_docs(tok, base.load_jsonl(a.train_docs))
    memory = base.MemoryIndex(base.load_jsonl(a.memory_docs))
    tests = {Path(p).stem: Path(p).read_bytes() for p in a.tests}
    assert len(docs) >= 1000 and len(memory.chunks) >= 500

    model, tr = base.train_mode(MODE, SEED, tok, docs, memory, a.steps, BATCH, lr=7e-4)

    main_eval = {
        name: base.build_main_eval(tok, b, 7000 + i * 31, n=256)
        for i, (name, b) in enumerate(tests.items())
    }
    logic = base.build_aux_eval(tok, 'logic', 88001, n=192)
    cyber = base.build_aux_eval(tok, 'cyber', 99001, n=192)
    ev = {name: base.evaluate_examples(model, MODE, x, tok, memory) for name, x in main_eval.items()}
    ev['logic_teacher_forced'] = base.evaluate_examples(model, MODE, logic, tok, memory)
    ev['cyber_teacher_forced'] = base.evaluate_examples(model, MODE, cyber, tok, memory)

    old_prompts = base.PROMPTS
    base.PROMPTS = PROMPTS
    gens = base.generation_suite(model, MODE, tok, memory, SEED + 514)
    base.PROMPTS = old_prompts
    logic_acc = base.aux_generation_accuracy(model, MODE, tok, memory, 'logic', 123451, n=64)
    cyber_acc = base.aux_generation_accuracy(model, MODE, tok, memory, 'cyber', 223451, n=64)

    result = {
        'format': 'nexus-r514-deep-diverse/1',
        'protocol': {
            'params': base.param_count(model),
            'architecture': 'R5.7 D_LOGIC_CYBER 6-layer RoPE cortex',
            'tokenizer': 'lossless Unigram4096',
            'steps': a.steps,
            'batch': BATCH,
            'seed': SEED,
            'curriculum': {'russian_surface': 0.94, 'logic': 0.03, 'cyber': 0.03},
            'corpus': 'document-disjoint Wikipedia_ru + RuHeritage + SynTagRus; graph memory separated from LM train',
            'purpose': 'test deep exposure on a diverse modern Russian corpus after R5.12 removed immediate AR collapse but overfit RuHeritage style',
        },
        'training': tr,
        'eval': ev,
        'generation': gens,
        'generation_summary': extra_generation_metrics(gens),
        'logic_generation': logic_acc,
        'cyber_generation': cyber_acc,
    }

    (out / '00_R514_RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = []
    for g in gens:
        metrics = {k: g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')}
        lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps(metrics, ensure_ascii=False)}")
    lines.append('GENERATION_SUMMARY ' + json.dumps(result['generation_summary'], ensure_ascii=False))
    lines.append(f"LOGIC={logic_acc['accuracy']:.6f} CYBER={cyber_acc['accuracy']:.6f}")
    (out / '01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines), encoding='utf-8')
    torch.save({'state_dict': model.state_dict(), 'protocol': result['protocol']}, out / 'R514_DEEP_DIVERSE_3M.pt')
    (out / 'README_RU.md').write_text(
        '# NEXUS R5.14 Deep-Diverse Russian\n\n'
        'Тот же 3M D_LOGIC_CYBER cortex обучается существенно глубже на документно-разделённом смешанном русском корпусе: Wikipedia + RuHeritage + SynTagRus. '
        'Критерий успеха: не только BPB, но исчезновение историко-архивного lexical lock и связный raw rollout на современных нейтральных промптах.\n',
        encoding='utf-8'
    )
    print(json.dumps({
        'steps': a.steps,
        'train': tr,
        'bpb': {k: v['bpb'] for k, v in ev.items()},
        'generation_summary': result['generation_summary'],
        'logic': logic_acc['accuracy'],
        'cyber': cyber_acc['accuracy'],
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
