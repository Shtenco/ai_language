#!/usr/bin/env python3
import argparse,json,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
import r524_capacity_sweep as cap

SEED=20260810
CONFIG='11M'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=8192);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--lr',type=float,default=0.00025);a=ap.parse_args()
    torch.set_num_threads(a.threads);random.seed(SEED);torch.manual_seed(SEED)
    tok=cap.Tok(a.tokenizer_model);assert tok.sp.unk_id()==0 and tok.sp.pad_id()==-1
    docs=cap.load_docs(a.train_docs,tok);assert len(docs)>=100
    cfg=cap.CONFIGS[CONFIG];model=cap.LM(cfg);ck=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(ck['state_dict']);assert cap.nparams(model)==cfg['expected']
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(SEED+526);hist=[];tokens=rawbytes=0;t0=time.perf_counter();model.train()
    for step in range(a.steps):
        q=[cap.sample(rng,docs) for _ in range(a.batch)];x,y,m=cap.pack(q);z=model(x);ce=F.cross_entropy(z.reshape(-1,cap.VOCAB),y.reshape(-1),reduction='none').view_as(y);loss=(ce*m).sum()/int(m.sum());opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);fac=cap.lrf(step,a.steps)
        for pg in opt.param_groups:pg['lr']=a.lr*fac
        opt.step();hist.append(float(loss.detach()));tokens+=int(m.sum());rawbytes+=sum(len(tok.dec(t).encode('utf-8')) for _,t in q)
        if (step+1)%512==0:print('TRAIN',step+1,'npt',sum(hist[-128:])/128,'MB',rawbytes/1048576,'tok/s',tokens/max(1,time.perf_counter()-t0),flush=True)
    tests={Path(p).stem:tok.enc(Path(p).read_text(encoding='utf-8')) for p in a.tests};ev={name:{str(c):cap.eval_bpb(model,tok,ids,c,SEED+i*101+c) for c in (8,16,32,48)} for i,(name,ids) in enumerate(tests.items())};gens=[cap.generate(model,tok,p,d,SEED+52600+i) for i,p in enumerate(cap.PROMPTS) for d in ('greedy','sample')]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r={'format':'nexus-r526-11m-diverse-clean/1','protocol':{'warmstart':'R5.24P 11M 1024-step clean direct-UD','params':cap.nparams(model),'d_model':cfg['d'],'heads':cfg['h'],'layers':cfg['l'],'ff':cfg['f'],'tokenizer':'fixed Unigram4096','surface_prefix':'NONE','graph_prefix':'NONE','left_padding':'NONE','context_train':'uniform 4..48','target_tokens':48,'continuation_steps':a.steps,'batch':a.batch,'lr':a.lr,'corpus':'document-disjoint diverse Russian'},'training':{'seconds':time.perf_counter()-t0,'target_tokens':tokens,'target_bytes':rawbytes,'last128_npt':sum(hist[-128:])/128},'eval':ev,'generation':gens};(out/'00_R526_RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nuniq={g['unique_word_ratio']:.3f} rep3={g['repeated_trigram_rate']:.3f}" for g in gens),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':r['protocol']},out/'R526_11M_DIVERSE_CLEAN.pt');print(json.dumps({'protocol':r['protocol'],'training':r['training'],'eval':ev},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
