#!/usr/bin/env python3
import argparse,json,math,random,shutil,time
from pathlib import Path
import torch
import torch.nn.functional as F
import r57_concept_graph_language as base
import r515_encoder_decoder as encdec

MODE='D_LOGIC_CYBER';SEED=20260810;BATCH=8

def sample(rng,docs,tok):
    u=rng.random()
    if u<.94:return base.make_main_example(rng,docs,tok)
    if u<.97:return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)


def load_model(path):
    ck=torch.load(path,map_location='cpu');m=base.LM();m.load_state_dict(ck['state_dict'],strict=True);return m


def train(student,teacher,tok,docs,memory,steps,lam):
    rng=random.Random(SEED+523);opt=torch.optim.AdamW(student.parameters(),lr=1.5e-4,weight_decay=.01,betas=(.9,.95));teacher.eval();student.train();hist=[];kls=[];tb_total=0;t0=time.perf_counter()
    for st in range(steps):
        ex=[sample(rng,docs,tok) for _ in range(BATCH)];x,y,mask,tb,_=base.pack(ex,MODE,tok,memory)
        fac=base.lr_factor(st,steps)
        for pg in opt.param_groups:pg['lr']=1.5e-4*fac
        with torch.no_grad():tz=teacher(x);tp=F.softmax(tz/1.0,dim=-1)
        z=student(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);ce_loss=(ce*mask).sum()/tb
        logp=F.log_softmax(z,dim=-1);kl_tok=F.kl_div(logp,tp,reduction='none').sum(-1);kl=(kl_tok*mask).sum()/mask.sum().clamp_min(1)
        loss=ce_loss+lam*kl
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.0);opt.step();hist.append(float(ce_loss));kls.append(float(kl));tb_total+=tb
        if (st+1)%256==0:print(json.dumps({'step':st+1,'ce_npb':sum(hist[-64:])/len(hist[-64:]),'kl_token':sum(kls[-64:])/len(kls[-64:]),'lambda':lam,'target_MB':tb_total/1048576},ensure_ascii=False),flush=True)
    return {'steps':steps,'lambda':lam,'target_bytes':tb_total,'train_s':time.perf_counter()-t0,'last64_npb':sum(hist[-64:])/len(hist[-64:]),'last64_kl':sum(kls[-64:])/len(kls[-64:])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--warm-checkpoint',required=True);ap.add_argument('--steps',type=int,default=2048);ap.add_argument('--lambda-kl',type=float,required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shutil.copy2(a.tokenizer_model,out/'sp_unigram_4096.model');tok=encdec.FixedSP(a.tokenizer_model);docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));tests={Path(p).stem:Path(p).read_bytes() for p in a.tests}
    teacher=load_model(a.warm_checkpoint);student=load_model(a.warm_checkpoint);assert base.param_count(student)==2_998_620
    tr=train(student,teacher,tok,docs,memory,a.steps,a.lambda_kl)
    main_eval={name:base.build_main_eval(tok,b,7000+i*31,n=256) for i,(name,b) in enumerate(tests.items())};logic=base.build_aux_eval(tok,'logic',88001,n=192);cyber=base.build_aux_eval(tok,'cyber',99001,n=192)
    ev={name:base.evaluate_examples(student,MODE,x,tok,memory) for name,x in main_eval.items()};ev['logic_teacher_forced']=base.evaluate_examples(student,MODE,logic,tok,memory);ev['cyber_teacher_forced']=base.evaluate_examples(student,MODE,cyber,tok,memory)
    old=base.PROMPTS;base.PROMPTS=base.PROMPTS+['Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','После ошибки программа изменила своё состояние и','Современная система получает данные и'];gens=base.generation_suite(student,MODE,tok,memory,SEED+523);base.PROMPTS=old
    la=base.aux_generation_accuracy(student,MODE,tok,memory,'logic',123451,n=64);ca=base.aux_generation_accuracy(student,MODE,tok,memory,'cyber',223451,n=64)
    result={'format':'nexus-r523-kl-homeostat/1','protocol':{'params':2_998_620,'warm':'R5.12 32K','steps':a.steps,'lambda_kl':a.lambda_kl,'lr':1.5e-4,'principle':'modern CE adaptation + frozen warm-model KL negative feedback; no new trainable layers'},'training':tr,'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca}
    (out/'00_R523_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={la['accuracy']:.6f} CYBER={ca['accuracy']:.6f}");(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':student.state_dict(),'protocol':result['protocol']},out/'R523_KL_HOMEOSTAT_3M.pt');print(json.dumps({'train':tr,'bpb':{k:v['bpb'] for k,v in ev.items()},'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
