#!/usr/bin/env python3
import argparse,json,random,shutil
from pathlib import Path
import torch
import r57_concept_graph_language as base
import r515_encoder_decoder as encdec

MODE='D_LOGIC_CYBER';SEED=20260810;BATCH=8

def deep_sample(rng,docs,tok):
    u=rng.random()
    if u<.94:return base.make_main_example(rng,docs,tok)
    if u<.97:return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)
base.sample_example=deep_sample

PROMPTS=base.PROMPTS+[
 'Россия занимает большую территорию, поэтому',
 'Исследователь проверил данные и пришёл к выводу, что',
 'Память помогает рассуждению, потому что',
 'Причина отличается от простой корреляции тем, что',
 'Для достижения цели система должна',
 'После ошибки программа изменила своё состояние и',
 'Современная компьютерная система обрабатывает данные и',
 'Если эксперимент не подтвердил гипотезу, исследователь должен',
]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--warm-checkpoint',required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shutil.copy2(a.tokenizer_model,out/'sp_unigram_4096.model');tok=encdec.FixedSP(a.tokenizer_model)
    probe='Продолжение обучения современного русского NEXUS №521.'.encode();assert tok.dec(tok.enc(probe))==probe
    docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};assert len(docs)>=1000 and len(memory.chunks)>=100
    ck=torch.load(a.warm_checkpoint,map_location='cpu');LMClass=base.LM
    def warm_factory():
        m=LMClass();m.load_state_dict(ck['state_dict'],strict=True);return m
    base.LM=warm_factory
    try:model,tr=base.train_mode(MODE,SEED,tok,docs,memory,a.steps,BATCH,lr=1.5e-4)
    finally:base.LM=LMClass
    assert base.param_count(model)==2_998_620
    main_eval={name:base.build_main_eval(tok,b,7000+i*31,n=256) for i,(name,b) in enumerate(tests.items())};logic=base.build_aux_eval(tok,'logic',88001,n=192);cyber=base.build_aux_eval(tok,'cyber',99001,n=192)
    ev={name:base.evaluate_examples(model,MODE,x,tok,memory) for name,x in main_eval.items()};ev['logic_teacher_forced']=base.evaluate_examples(model,MODE,logic,tok,memory);ev['cyber_teacher_forced']=base.evaluate_examples(model,MODE,cyber,tok,memory)
    old=base.PROMPTS;base.PROMPTS=PROMPTS;gens=base.generation_suite(model,MODE,tok,memory,SEED+521);base.PROMPTS=old
    la=base.aux_generation_accuracy(model,MODE,tok,memory,'logic',123451,n=64);ca=base.aux_generation_accuracy(model,MODE,tok,memory,'cyber',223451,n=64)
    result={'format':'nexus-r521-modern-continuation/1','protocol':{'params':base.param_count(model),'architecture':'unchanged R5.12/R5.7 D_LOGIC_CYBER 6-layer RoPE','tokenizer':'frozen R5.12 Unigram4096','warm_checkpoint':'R5.12 32K RuHeritage-heavy','extra_steps':a.steps,'batch':BATCH,'lr':1.5e-4,'curriculum':{'surface':.94,'logic':.03,'cyber':.03},'purpose':'pure data/exposure control: no new layers or random adapters'},'training':tr,'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca}
    (out/'00_R521_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={la['accuracy']:.6f} CYBER={ca['accuracy']:.6f}");(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R521_MODERN_CONTINUATION_3M.pt');print(json.dumps({'train':tr,'bpb':{k:v['bpb'] for k,v in ev.items()},'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
