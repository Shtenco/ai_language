#!/usr/bin/env python3
import argparse
import json
import random
import re
import shutil
from pathlib import Path

import torch
import r57_concept_graph_language as base57
import r515_encoder_decoder as base515

SENT_PAIRS = []
SPLIT_RE = re.compile(r'(?<=[.!?…])\s+|\n+')
CYR = re.compile(r'[А-Яа-яЁё]')
WORD = re.compile(r'[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z-]{1,}')
BAD = re.compile(r'https?://|www\.|ISBN|==|\[править\]|\{\{|\}\}|\|\||<ref|</ref', re.I)

PROMPTS = [
    'Наука развивается потому, что',
    'Причинная связь отличается от корреляции тем, что',
    'Если эксперимент не подтвердил гипотезу, исследователь должен',
    'Память помогает рассуждению, потому что',
    'Когда программа обнаружила противоречие, она',
    'Чтобы проверить утверждение, необходимо',
    'Современная компьютерная система обрабатывает данные и',
    'После ошибки система изменила своё состояние и',
    'Хорошее объяснение начинается с того, что',
    'Если два факта противоречат друг другу, нужно',
    'Исследователь сравнил результаты и обнаружил, что',
    'Человек принимает решение после того, как',
]


def clean_sentence(s):
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) < 45 or len(s) > 420 or BAD.search(s):
        return None
    words = WORD.findall(s)
    if len(words) < 7 or len(words) > 55:
        return None
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return None
    cyr = sum(1 for c in letters if CYR.match(c)) / len(letters)
    if cyr < 0.78:
        return None
    digits = sum(c.isdigit() for c in s)
    if digits > 8 or digits / max(1, len(s)) > 0.06:
        return None
    return s


def build_sentence_pairs(tok, raw_docs, limit=80000):
    pairs = []
    for d in raw_docs:
        sents = []
        for x in SPLIT_RE.split(d.get('text','')):
            y = clean_sentence(x)
            if y:
                sents.append(y)
        for i in range(1, len(sents)):
            ctx_text = ' '.join(sents[max(0, i-2):i])
            tgt_text = sents[i]
            ctx = tok.enc(ctx_text.encode('utf-8'))[-base515.CONTEXT_TOKENS:]
            tgt = tok.enc(tgt_text.encode('utf-8'))
            if not (8 <= len(tgt) <= base515.TARGET_TOKENS):
                continue
            if len(ctx) < 8:
                continue
            pairs.append({'ctx':ctx,'tgt':tgt,'ctx_text':ctx_text,'meta':{'kind':'main','sentence_aligned':True},'target_text':tgt_text})
            if len(pairs) >= limit:
                return pairs
    return pairs


def sentence_sample(rng, docs, tok):
    u = rng.random()
    if u < 0.70 and SENT_PAIRS:
        return SENT_PAIRS[rng.randrange(len(SENT_PAIRS))]
    if u < 0.94:
        return base57.make_main_example(rng, docs, tok)
    if u < 0.97:
        return base57.make_aux_example('logic', rng, tok, train=True)
    return base57.make_aux_example('cyber', rng, tok, train=True)


def eval_sentence_bytes(tok, data, limit=192):
    text = data.decode('utf-8','replace')
    pseudo = [{'text':text}]
    return build_sentence_pairs(tok, pseudo, limit=limit)


def main():
    global SENT_PAIRS
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True)
    ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--warm-checkpoint',required=True);ap.add_argument('--steps',type=int,default=8192);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shutil.copy2(a.tokenizer_model,out/'sp_unigram_4096.model')
    tok=base515.FixedSP(a.tokenizer_model)
    raw_docs=base57.load_jsonl(a.train_docs);docs=base57.tokenize_docs(tok,raw_docs);memory=base57.MemoryIndex(base57.load_jsonl(a.memory_docs))
    SENT_PAIRS=build_sentence_pairs(tok,raw_docs,limit=80000)
    assert len(SENT_PAIRS)>=5000,len(SENT_PAIRS)
    base515.sample_example=sentence_sample
    model=base515.NEXUS515();copied=base515.warm_start(model,a.warm_checkpoint);params=base515.param_count(model);assert params==5404367
    print(json.dumps({'params':params,'warm_tensors':len(copied),'train_docs':len(docs),'sentence_pairs':len(SENT_PAIRS),'memory_chunks':len(memory.chunks)},ensure_ascii=False),flush=True)
    tr=base515.train(model,tok,docs,memory,a.steps,base515.BATCH,base515.SEED)
    tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};ev={}
    for i,(name,b) in enumerate(tests.items()):
        random_eval=base57.build_main_eval(tok,b,7000+i*31,n=256)
        sent_eval=eval_sentence_bytes(tok,b,192)
        ev[name]={
            'random_true':base515.eval_examples(model,random_eval,tok,memory,'true'),
            'random_null':base515.eval_examples(model,random_eval,tok,memory,'null'),
            'sentence_true':base515.eval_examples(model,sent_eval,tok,memory,'true') if sent_eval else None,
            'sentence_examples':len(sent_eval),
        }
    logic=base57.build_aux_eval(tok,'logic',88001,n=192);cyber=base57.build_aux_eval(tok,'cyber',99001,n=192)
    ev['logic_teacher_forced']=base515.eval_examples(model,logic,tok,memory,'true');ev['cyber_teacher_forced']=base515.eval_examples(model,cyber,tok,memory,'true')
    gens=[]
    for i,p in enumerate(PROMPTS):
        for decode in ('greedy','sample'):
            txt=base515.generate(model,tok,memory,p,decode,base515.SEED+1700+i,max_new=48,graph_mode='true')
            gens.append({'prompt':p,'decode':decode,'continuation':txt,**base515.gen_metrics(txt)})
    la=base515.exact_aux_accuracy(model,tok,memory,'logic',123451,64);ca=base515.exact_aux_accuracy(model,tok,memory,'cyber',223451,64)
    result={'format':'nexus-r517-sentence-curriculum/1','protocol':{'params':params,'base':'R5.15 semantic encoder-decoder','tokenizer':'frozen R5.12 Unigram4096','warm_start':'R5.12 32K','steps':a.steps,'batch':base515.BATCH,'curriculum':{'sentence_aligned':0.70,'random_surface':0.24,'logic':0.03,'cyber':0.03},'sentence_filter':'7-55 words, 78%+ Cyrillic letters, low digits, no obvious markup/bibliography noise'},'training':tr,'sentence_pairs':len(SENT_PAIRS),'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca,'warm_tensors':len(copied)}
    (out/'00_R517_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','replacement_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={la['accuracy']:.6f} CYBER={ca['accuracy']:.6f}")
    (out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8')
    torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R517_SENTENCE_CURRICULUM.pt')
    print(json.dumps({'train':tr,'sentence_pairs':len(SENT_PAIRS),'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
