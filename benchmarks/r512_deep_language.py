#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
import torch
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER'
STEPS=32768
BATCH=8
SEED=20260810

# Deep language curriculum: keep exact organs alive, but put most target exposure into Russian surface language.
def deep_sample(rng,docs,tok):
    u=rng.random()
    if u<0.94:return base.make_main_example(rng,docs,tok)
    if u<0.97:return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)

base.sample_example=deep_sample

PROMPTS=base.PROMPTS+[
    'Россия занимает большую территорию, поэтому',
    'Исследователь проверил данные и пришёл к выводу, что',
    'Память помогает рассуждению, потому что',
    'Причина отличается от простой корреляции тем, что',
    'Для достижения цели система должна',
    'После ошибки программа изменила своё состояние и',
    'В книге автор рассказывает о том, как',
    'Утром город проснулся, и на улицах',
]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-text',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);raw=Path(a.train_text).read_bytes();tok=base.tt.train_sp(raw,base.VOCAB,'unigram',out);probe='Глубокое обучение русского NEXUS №123.\n'.encode();assert tok.dec(tok.enc(probe))==probe
    docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};assert len(docs)>=50 and len(memory.chunks)>=100
    model,tr=base.train_mode(MODE,SEED,tok,docs,memory,STEPS,BATCH,lr=7e-4)
    main_eval={name:base.build_main_eval(tok,b,7000+i*31,n=256) for i,(name,b) in enumerate(tests.items())};logic=base.build_aux_eval(tok,'logic',88001,n=192);cyber=base.build_aux_eval(tok,'cyber',99001,n=192)
    ev={name:base.evaluate_examples(model,MODE,x,tok,memory) for name,x in main_eval.items()};ev['logic_teacher_forced']=base.evaluate_examples(model,MODE,logic,tok,memory);ev['cyber_teacher_forced']=base.evaluate_examples(model,MODE,cyber,tok,memory)
    old=base.PROMPTS;base.PROMPTS=PROMPTS;gens=base.generation_suite(model,MODE,tok,memory,SEED+501);base.PROMPTS=old
    la=base.aux_generation_accuracy(model,MODE,tok,memory,'logic',123451,n=64);ca=base.aux_generation_accuracy(model,MODE,tok,memory,'cyber',223451,n=64)
    result={'format':'nexus-r512-deep-language/1','protocol':{'params':base.param_count(model),'architecture':'R5.7 D_LOGIC_CYBER 6-layer RoPE cortex','tokenizer':'lossless Unigram4096','steps':STEPS,'batch':BATCH,'seed':SEED,'curriculum':{'russian_surface':.94,'logic':.03,'cyber':.03},'purpose':'test whether prior free-generation collapse was primarily underexposure; 4096-step R5.7 saw only 7.27MB target bytes'},'training':tr,'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca}
    (out/'00_R512_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={la['accuracy']:.6f} CYBER={ca['accuracy']:.6f}");(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R512_DEEP_LANGUAGE_3M.pt')
    (out/'README_RU.md').write_text('# NEXUS R5.12 Deep Language Exposure\n\nТот же проверенный R5.7 D cortex обучается 32 768 шагов при 94% языкового curriculum. Цель — отделить архитектурный предел от банального недообучения: предыдущие 4096 шагов дали лишь 7.27 MB target-байтов. Главный критерий — сырой greedy/sample русский после freeze, а не только BPB.\n',encoding='utf-8');print(json.dumps({'train':tr,'bpb':{k:v['bpb'] for k,v in ev.items()},'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
