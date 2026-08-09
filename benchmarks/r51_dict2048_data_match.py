#!/usr/bin/env python3
import argparse,csv,json,math,random
from pathlib import Path
import torch
import r51_dict2048_real_lm as b

STEPS_EACH=54  # 108*1024=110592 dict tokens; ~264k source bytes at train compression ~2.39 B/token

def matched_plan(n,s):
 g=random.Random(s);o=[]
 for L,B in ((64,16),(128,8)):
  for _ in range(STEPS_EACH):o.append((L,[g.randrange(n-L-1) for _ in range(B)]))
 return o

def sampled_bytes(data,lens,s):
 total=0;nt=0
 for L,st in matched_plan(len(data),s+10000):
  _,y=b.batch(data,L,st);total+=int(lens[y].sum());nt+=y.numel()
 return nt,total

def main():
 p=argparse.ArgumentParser()
 for q in ('train','valid','test','shift','out'):p.add_argument('--'+q,required=True)
 p.add_argument('--seeds',default='11,29,47');p.add_argument('--threads',type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads)
 raw={k:Path(getattr(a,k)).read_bytes() for k in ('train','valid','test','shift')};dt=b.DictTok.train(raw['train']);enc={k:dt.enc(v) for k,v in raw.items()}
 for k in raw:assert dt.dec(enc[k].tolist())==raw[k]
 old=b.plan;b.plan=matched_plan
 rows=[]
 try:
  for s in map(int,a.seeds.split(',')):
   nt,nb=sampled_bytes(enc['train'],dt.lens,s)
   for name in ('T0_DICT2048','R4_DICT2048'):
    print('RUN DATA_MATCH',s,name,'tokens',nt,'source_bytes',nb,flush=True);m,tm=b.train(name,s,enc['train']);r={'seed':s,'model':name+'_DATA_MATCH','params':b.pc(m),'train_tokens':nt,'train_source_bytes':nb,'train_source_bytes_ratio_vs_byte262144':nb/262144,**tm}
    for sp in ('valid','test','shift'):
     for L in (64,128):
      for k,v in b.ev(m,enc[sp],dt.lens,L,s+500).items():r[f'{sp}_{L}_{k}']=v
    for k,v in b.speed(m,enc['test'],dt.lens).items():r['rt128_'+k]=v
    rows.append(r)
 finally:b.plan=old
 agg=[]
 for name in ('T0_DICT2048_DATA_MATCH','R4_DICT2048_DATA_MATCH'):
  rr=[r for r in rows if r['model']==name];z={'model':name,'n':len(rr)}
  for k in ('train_tokens','train_source_bytes','train_source_bytes_ratio_vs_byte262144','test_128_bpb','valid_128_bpb','shift_128_bpb','test_128_token_ppl','test_128_token_top1','test_128_bytes_per_token','rt128_tok_s','rt128_byte_s','train_s'):
   v=[r[k] for r in rr];mu=sum(v)/len(v);z[k+'_mean']=mu;z[k+'_sd']=(sum((x-mu)**2 for x in v)/len(v))**.5
  agg.append(z)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'01_DATA_MATCH_RESULTS.json').write_text(json.dumps({'aggregate':agg,'per_seed':rows},indent=2))
 for fn,arr in [('02_DATA_MATCH_PER_SEED.csv',rows),('03_DATA_MATCH_AGGREGATE.csv',agg)]:
  keys=[]
  for d in arr:
   for k in d:
    if k not in keys:keys.append(k)
  with open(out/fn,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(arr)
 (out/'README.md').write_text('# DICT2048 raw-byte-matched control\n\n~262k original source bytes, not 262k dictionary tokens.\n\n'+'\n'.join(f"- {z['model']}: train source bytes {z['train_source_bytes_mean']:.0f}; test BPB {z['test_128_bpb_mean']:.4f}; shift BPB {z['shift_128_bpb_mean']:.4f}" for z in agg))
 print('DONE',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__':main()
