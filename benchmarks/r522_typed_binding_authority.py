#!/usr/bin/env python3
import argparse, hashlib, json, random, re
from pathlib import Path
import r57_concept_graph_language as base

SEED=20260810
LOGIC_N=10000
CYBER_N=10000


def choose_variant(key, n):
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:4],'big') % n


def logic_state(meta):
    # In production these typed values are nodes/edges in the proof graph, not text guessed by the LM.
    return {
        'kind':'logic',
        'status':'proved' if meta['answer_yes'] else 'not_proved',
        'claim':meta['claim'],
        'premise':meta['premise'],
        'depth':int(meta['depth']),
    }


def render_logic(st):
    claim=st['claim'];prem=st['premise']
    if st['status']=='proved':
        variants=[
            f'Да. Из факта «{prem}» и двух последовательных правил следует «{claim}».',
            f'Да. Утверждение «{claim}» выводится из факта «{prem}» по цепочке из двух правил.',
            f'Да. Основание «{prem}» достаточно: после двух шагов вывода получаем «{claim}».',
        ]
    else:
        variants=[
            f'Нет. Из текущего факта нельзя вывести «{claim}»: необходимая посылка «{prem}» отсутствует.',
            f'Нет. Для вывода «{claim}» не хватает посылки «{prem}».',
            f'Нет. Утверждение «{claim}» не доказано, потому что отсутствует исходная посылка «{prem}».',
        ]
    return variants[choose_variant(claim+'\x1f'+prem, len(variants))]


def cyber_state(meta):
    return {
        'kind':'cyber',
        'target':meta['target'],
        'current':meta['current'],
        'error':meta['error'],
        'tolerance':meta['tolerance'],
        'mode':meta['mode'],
        'action':meta['action'],
    }


def render_cyber(st):
    return (
        f"Цель {st['target']}, текущее значение {st['current']}, ошибка {st['error']}. "
        f"При допуске {st['tolerance']} режим: {st['mode']}. Действие: {st['action']}."
    )


def verify_logic(text, st):
    return (
        ('Да.' in text) == (st['status']=='proved') and
        st['claim'] in text and st['premise'] in text and
        ('не доказано' not in text.lower() if st['status']=='proved' else True)
    )


def verify_cyber(text, st):
    fields=[str(st[k]) for k in ('target','current','error','tolerance')]
    return all(x in text for x in fields) and st['mode'] in text and st['action'] in text


def run_logic(n):
    rng=random.Random(SEED+22);ok=0;rows=[]
    for i in range(n):
        ctx,target,meta=base.make_logic_example(rng,train=False)
        st=logic_state(meta);text=render_logic(st);good=verify_logic(text,st);ok+=int(good)
        if i<32:rows.append({'context':ctx,'legacy_target':target,'typed_state':st,'rendered':text,'verified':good})
    return {'n':n,'verified':ok,'accuracy':ok/n,'rows':rows}


def run_cyber(n):
    rng=random.Random(SEED+23);ok=0;rows=[]
    for i in range(n):
        ctx,target,meta=base.make_cyber_example(rng)
        st=cyber_state(meta);text=render_cyber(st);good=verify_cyber(text,st);ok+=int(good)
        if i<32:rows.append({'context':ctx,'legacy_target':target,'typed_state':st,'rendered':text,'verified':good})
    return {'n':n,'verified':ok,'accuracy':ok/n,'rows':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='nexus_r522_results');ap.add_argument('--logic-n',type=int,default=LOGIC_N);ap.add_argument('--cyber-n',type=int,default=CYBER_N);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    logic=run_logic(a.logic_n);cyber=run_cyber(a.cyber_n)
    result={
        'format':'nexus-r522-typed-binding-authority/1',
        'principle':'probabilistic language organ may choose wording; exact identity, polarity, numbers and proof bindings remain typed graph references under Authority',
        'logic':logic,'cyber':cyber,
        'contract':{
            'LM_owns':['surface wording','discourse connective','style'],
            'Authority_owns':['claim identity','premise identity','proof status','numeric state','mode','action','provenance'],
            'failure_policy':'if a generated surface form disagrees with typed state, reject or deterministically re-render; never average the conflict in logits'
        }
    }
    (out/'00_R522_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for group in (logic['rows'],cyber['rows']):
        for r in group:lines.append(json.dumps(r,ensure_ascii=False))
    (out/'01_TYPED_BINDING_EXAMPLES.jsonl').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'README_RU.md').write_text(
        '# NEXUS R5.22 Typed Binding / Authority\n\n'
        'R5.12 показал: языковой organ уже выучивает логический каркас ответа, но подменяет конкретные имена переменных. '
        'R5.22 запрещает вероятностному декодеру владеть идентичностью факта. Claim, premise, числа, polarity и proof status остаются типизированными значениями графа; language organ отвечает только за формулировку. '
        'Benchmark проверяет 10k unseen logic и 10k cyber states с post-render verifier.\n',encoding='utf-8')
    print(json.dumps({'logic_accuracy':logic['accuracy'],'cyber_accuracy':cyber['accuracy'],'total':a.logic_n+a.cyber_n},ensure_ascii=False))
    if logic['accuracy']!=1.0 or cyber['accuracy']!=1.0:raise SystemExit('Authority binding invariant failed')

if __name__=='__main__':main()
