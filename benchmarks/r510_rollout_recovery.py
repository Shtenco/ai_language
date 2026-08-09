#!/usr/bin/env python3
import argparse, json, math, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER'

def set_seed(s): random.seed(s); torch.manual_seed(s)
def lr_factor(step,total):
    warm=max(32,total//32)
    if step<warm:return max(.05,(step+1)/warm)
    q=(step-warm)/max(1,total-warm);return .5*(1+math.cos(math.pi*min(1,q)))

def corrupt_history(x,z_clean,mask,p):
    bad=x.clone();B,L=x.shape;pred=z_clean.argmax(-1)
    start=base.PREFIX_TOKENS+base.CONTEXT_TOKENS
    for j in range(start,L):
        eligible=mask[:,j-1] & (x[:,j]!=0)
        if not bool(eligible.any()):continue
        choose=(torch.rand(B)<p) & eligible
        bad[choose,j]=pred[choose,j-1]
    return bad

def unlikelihood(z,x,y,mask):
    probs=F.softmax(z,dim=-1);prev=probs.gather(-1,x[...,None]).squeeze(-1).clamp(max=1-1e-6);valid=mask & (y!=x)
    if not bool(valid.any()):return z.sum()*0
    return (-torch.log1p(-prev[valid])).mean()

def train_recovery(seed,tok,docs,memory,steps,batch,lr=1e-3):
    set_seed(seed);model=base.LM();pc=base.param_count(model);assert pc==2_998_620,pc
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(seed+77881)
    clean_hist=[];recover_hist=[];ul_hist=[];seen=0;t0=time.perf_counter();model.train()
    for step in range(steps):
        ex=[base.sample_example(rng,docs,tok) for _ in range(batch)];x,y,mask,tb,_=base.pack(ex,MODE,tok,memory)
        for pg in opt.param_groups:pg['lr']=lr*lr_factor(step,steps)
        opt.zero_grad(set_to_none=True);z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);clean=(ce*mask).sum()/tb
        p=0.04+0.24*(step/max(1,steps-1));bad=corrupt_history(x,z.detach(),mask,p);zb=model(bad);ceb=F.cross_entropy(zb.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);recover=(ceb*mask).sum()/tb;ul=unlikelihood(zb,bad,y,mask)
        loss=.45*clean+.55*recover+.04*ul;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        clean_hist.append(float(clean));recover_hist.append(float(recover));ul_hist.append(float(ul));seen+=tb
        if (step+1)%256==0:print('TRAIN',step+1,'clean',sum(clean_hist[-64:])/64,'recover',sum(recover_hist[-64:])/64,'ul',sum(ul_hist[-64:])/64,'corrupt_p',p,'MB',seen/1048576,flush=True)
    return model,{'mode':'R5.10_RECOVERY','seed':seed,'params':pc,'steps':steps,'batch':batch,'train_s':time.perf_counter()-t0,'target_bytes':seen,'last64_clean_npb':sum(clean_hist[-64:])/min(64,len(clean_hist)),'last64_recovery_npb':sum(recover_hist[-64:])/min(64,len(recover_hist)),'last64_unlikelihood':sum(ul_hist[-64:])/min(64,len(ul_hist)),'final_corruption_p':.28}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-text',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=4096);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--seed',type=int,default=20260810);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);raw=Path(a.train_text).read_bytes();tok=base.tt.train_sp(raw,base.VOCAB,'unigram',out);probe='Кибернетическое восстановление русского текста №123.\n'.encode();assert tok.dec(tok.enc(probe))==probe
    docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};assert len(docs)>=50 and len(memory.chunks)>=100
    model,tr=train_recovery(a.seed,tok,docs,memory,a.steps,a.batch);main_eval={name:base.build_main_eval(tok,b,7000+i*31,n=128) for i,(name,b) in enumerate(tests.items())};logic=base.build_aux_eval(tok,'logic',88001,n=96);cyber=base.build_aux_eval(tok,'cyber',99001,n=96)
    ev={name:base.evaluate_examples(model,MODE,x,tok,memory) for name,x in main_eval.items()};ev['logic_teacher_forced']=base.evaluate_examples(model,MODE,logic,tok,memory);ev['cyber_teacher_forced']=base.evaluate_examples(model,MODE,cyber,tok,memory)
    gens=base.generation_suite(model,MODE,tok,memory,a.seed+501);la=base.aux_generation_accuracy(model,MODE,tok,memory,'logic',123451,n=32);ca=base.aux_generation_accuracy(model,MODE,tok,memory,'cyber',223451,n=32);rt=base.runtime(model,MODE,tok,memory,docs,a.seed,reps=20)
    result={'format':'nexus-r510-cybernetic-rollout-recovery/1','protocol':{'params':base.param_count(model),'base':'R5.7 D_LOGIC_CYBER','tokenizer':'lossless Unigram4096','architecture':'same 6-layer RoPE 2,998,620-param cortex','training':'clean teacher forcing + own-greedy history corruption ramp 4%->28% + immediate-loop unlikelihood','steps':a.steps,'batch':a.batch,'seed':a.seed},'training':tr,'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca,'runtime':rt}
    (out/'00_R510_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC_GENERATION_ACCURACY={la['accuracy']:.6f}");lines.append(f"CYBER_GENERATION_ACCURACY={ca['accuracy']:.6f}");(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R510_RECOVERY_3M.pt')
    (out/'README_RU.md').write_text('# NEXUS R5.10 Cybernetic Rollout Recovery\n\nТот же R5.7 D cortex учится не только на правильной teacher-forced истории, но и восстанавливать правильное продолжение после собственных greedy-ошибок в предыдущих target-токенах. Вероятность self-corruption растёт от 4% до 28%; отдельный unlikelihood-term подавляет мгновенные self-loops, когда реальный текст требует другого следующего токена.\n',encoding='utf-8');print(json.dumps({'training':tr,'eval':{k:v['bpb'] for k,v in ev.items()},'logic_acc':la['accuracy'],'cyber_acc':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
