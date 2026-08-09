#!/usr/bin/env python3
import argparse,csv,json,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
import r52_tokenizer_tournament as tt
import r53_source_matched_lm as r53

SPECS={1024:(188,4,659),2048:(138,6,983),4096:(114,6,900)}

def train_custom(V,seed,tok,raw):
 d,h,f=SPECS[V];r53.set_seed(seed);m=r53.LM(V,d,h,f,'r4');params=r53.pc(m);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=.01,betas=(.9,.95));ss=r53.starts_for(raw,seed+10000,r53.STEPS*r53.BATCH);losses=[];nb=0;t0=time.perf_counter();m.train()
 for j in range(r53.STEPS):
  x,y,mask,tbytes,sbytes=r53.encode_batch(tok,raw,ss[j*r53.BATCH:(j+1)*r53.BATCH]);opt.zero_grad(set_to_none=True);z=m(x);ce=F.cross_entropy(z.reshape(-1,V),y.reshape(-1),reduction='none').view_as(y);loss=(ce*mask).sum()/tbytes;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss));nb+=sbytes
 return m,{'params':params,'train_s':time.perf_counter()-t0,'train_source_bytes':nb,'train_loss_nats_per_byte_last32':sum(losses[-32:])/32}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='3,11,23,47,73');ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);train=Path(a.train).read_bytes();tests={Path(p).stem:Path(p).read_bytes() for p in a.tests}
 toks={v:tt.train_sp(train,v,'unigram',out) for v in SPECS};probe='Hello, мир! Проверка 123.\n'.encode();
 for v,t in toks.items():assert t.dec(t.enc(probe))==probe,v
 rows=[]
 for s in map(int,a.seeds.split(',')):
  for V,tok in toks.items():
   print('RUN',s,'UNIGRAM',V,flush=True);m,tm=train_custom(V,s,tok,train);row={'seed':s,'vocab':V,'model':f'R4_UNIGRAM{V}',**tm}
   for sp,raw in tests.items():
    e=r53.evaluate(m,tok,raw,s);row.update({f'{sp}_{k}':v for k,v in e.items()})
   rt=r53.runtime(m,tok,tests[next(iter(tests))],s);row.update({'rt_'+k:v for k,v in rt.items()});rows.append(row);print(json.dumps({'seed':s,'vocab':V,'params':row['params'],**{sp:row[f'{sp}_bpb'] for sp in tests},'byte_s':row['rt_source_byte_s']}),flush=True)
 bpb=[f'{sp}_bpb' for sp in tests]
 for x in rows:x['mean_eval_bpb']=sum(x[k] for k in bpb)/len(bpb)
 agg=[]
 for V in SPECS:
  rr=[x for x in rows if x['vocab']==V];z={'vocab':V,'model':f'R4_UNIGRAM{V}','n':len(rr),'params':rr[0]['params']}
  for k in bpb+['mean_eval_bpb','rt_source_byte_s','train_s','train_source_bytes']:
   vals=[x[k] for x in rr];mu=sum(vals)/len(vals);z[k+'_mean']=mu;z[k+'_sd']=(sum((q-mu)**2 for q in vals)/len(vals))**0.5
  agg.append(z)
 agg.sort(key=lambda z:z['mean_eval_bpb_mean']);(out/'00_SWEEP_RESULTS.json').write_text(json.dumps({'protocol':{'source_context_bytes':128,'steps':256,'batch':8,'topology':'R4','specs':{str(k):{'d':v[0],'heads':v[1],'ff':v[2]} for k,v in SPECS.items()}},'aggregate':agg,'per_seed':rows},indent=2))
 for fn,arr in [('01_PER_SEED.csv',rows),('02_AGGREGATE.csv',agg)]:
  keys=list(arr[0]);
  with open(out/fn,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(arr)
 (out/'README_RU.md').write_text('# NEXUS R5.5 — UNIGRAM VOCAB SWEEP\n\nR4, matched 128 source-byte context/exposure. Vocab 1024/2048/4096; parameter budgets are 999,974 / 999,982 / 999,984 (spread 10 params).\n\n'+'\n'.join(f"- {z['model']}: mean BPB {z['mean_eval_bpb_mean']:.4f}; source B/s {z['rt_source_byte_s_mean']:.0f}" for z in agg))
 print('DONE',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__':main()
