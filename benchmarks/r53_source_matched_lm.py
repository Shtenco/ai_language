#!/usr/bin/env python3
import argparse,csv,json,math,random,time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import r52_tokenizer_tournament as tt

CTX_BYTES=128; STEPS=256; BATCH=8; EVAL_BATCHES=24; MAX_TOK=128
SPECS={
 'BYTE256':(256,195,5,789),
 'LEXDICT2048':(2048,135,5,1029),
 'BPE2048':(2048,135,5,1029),
 'UNIGRAM2048':(2048,135,5,1029),
}

def set_seed(s):random.seed(s);torch.manual_seed(s)
def pc(m):return sum(p.numel() for p in m.parameters())
def smask(L,w=16,k=4):
 m=torch.zeros(L,L,dtype=torch.bool)
 for i in range(L):
  m[i,max(0,i-w+1):i+1]=1;x=(i+1)*0x9E3779B1
  for q in range(k):x=(1664525*x+1013904223+q*97)&0xffffffff;m[i,x%(i+1)]=1
  m[i,0]=1
 return m
class Block(nn.Module):
 def __init__(self,d,f):
  super().__init__();self.a=nn.LayerNorm(d);self.q=nn.Linear(d,3*d);self.p=nn.Linear(d,d);self.b=nn.LayerNorm(d);self.f1=nn.Linear(d,f);self.f2=nn.Linear(f,d)
class LM(nn.Module):
 def __init__(self,V,d,h,f,top):
  super().__init__();self.V=V;self.d=d;self.hn=h;self.f=f;self.top=top;self.e=nn.Embedding(V,d);self.pos=nn.Embedding(MAX_TOK,d);self.bs=nn.ModuleList([Block(d,f) for _ in range(2)]);self.l=nn.LayerNorm(d);self.head=nn.Linear(d,V,bias=False);self.head.weight=self.e.weight
 def forward(self,x):
  B,L=x.shape;d=self.d;H=self.hn;hd=d//H;y=self.e(x)+self.pos(torch.arange(L,device=x.device))[None];ca=torch.tril(torch.ones(L,L,dtype=torch.bool,device=x.device));ma=ca[None,None] if self.top=='dense' else smask(L).to(x.device)[None,None]
  for b in self.bs:
   z=b.q(b.a(y)).view(B,L,3,H,hd).permute(2,0,3,1,4);q,k,v=z;a=F.scaled_dot_product_attention(q,k,v,attn_mask=ma);y=y+b.p(a.transpose(1,2).contiguous().view(B,L,d));z=F.gelu(b.f1(b.b(y)))
   if self.top=='r4':g=torch.arange(self.f,device=x.device)*4//self.f;z=z*(g[None,None,:]!=(x%4)[:,:,None])
   y=y+b.f2(z)
  return self.head(self.l(y))
class ByteTok:
 name='BYTE256'
 def enc(self,b):return list(b)
 def dec(self,ids):return bytes(ids)

def safe_span(raw,start,n=CTX_BYTES):
 start=min(start,len(raw)-2)
 while start<len(raw) and (raw[start]&0xC0)==0x80:start+=1
 end=min(len(raw),start+n)
 while end>start and end<len(raw) and (raw[end]&0xC0)==0x80:end-=1
 x=raw[start:end]
 try:x.decode('utf-8')
 except UnicodeDecodeError:return safe_span(raw,start+1,n)
 return x

def starts_for(raw,seed,n):
 g=random.Random(seed);return [g.randrange(0,max(1,len(raw)-CTX_BYTES-8)) for _ in range(n)]
def encode_batch(tok,raw,starts):
 seq=[];spans=[];firstlens=[]
 for st in starts:
  b=safe_span(raw,st);ids=tok.enc(b)
  if len(ids)<2:continue
  if len(ids)>MAX_TOK:ids=ids[:MAX_TOK];b=tok.dec(ids)
  seq.append(ids);spans.append(b);firstlens.append(len(tok.dec([ids[0]])))
 L=max(len(x) for x in seq);xx=torch.zeros(len(seq),L-1,dtype=torch.long);yy=torch.zeros_like(xx);mask=torch.zeros_like(xx,dtype=torch.bool)
 for i,ids in enumerate(seq):
  n=len(ids)-1;xx[i,:n]=torch.tensor(ids[:-1]);yy[i,:n]=torch.tensor(ids[1:]);mask[i,:n]=1
 target_bytes=sum(max(1,len(b)-fl) for b,fl in zip(spans,firstlens));source_bytes=sum(len(b) for b in spans)
 return xx,yy,mask,target_bytes,source_bytes

def build_toks(train,out):
 word=tt.ordered_candidates(tt.candidate_counts(train,'word'));lex=tt.ordered_candidates(tt.candidate_counts(train,'lex'))
 toks={'BYTE256':ByteTok(),'LEXDICT2048':tt.build_dict(lex,2048,'lex')}
 toks['BPE2048']=tt.train_sp(train,2048,'bpe',out);toks['UNIGRAM2048']=tt.train_sp(train,2048,'unigram',out)
 for n,t in toks.items():
  probe='Hello, мир! Проверка 123.\n'.encode();assert t.dec(t.enc(probe))==probe,(n,t.dec(t.enc(probe)))
 return toks

def train_one(name,top,seed,tok,raw):
 V,d,h,f=SPECS[name];set_seed(seed);m=LM(V,d,h,f,top);assert pc(m)==999978,(name,pc(m));opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=.01,betas=(.9,.95));ss=starts_for(raw,seed+10000,STEPS*BATCH);losses=[];nb=0;t0=time.perf_counter();m.train()
 for j in range(STEPS):
  x,y,mask,tbytes,sbytes=encode_batch(tok,raw,ss[j*BATCH:(j+1)*BATCH]);opt.zero_grad(set_to_none=True);z=m(x);ce=F.cross_entropy(z.reshape(-1,V),y.reshape(-1),reduction='none').view_as(y);loss=(ce*mask).sum()/tbytes;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss));nb+=sbytes
 return m,{'train_s':time.perf_counter()-t0,'train_source_bytes':nb,'train_loss_nats_per_byte_last32':sum(losses[-32:])/32}
@torch.no_grad()
def evaluate(m,tok,raw,seed):
 ss=starts_for(raw,seed+50000,EVAL_BATCHES*BATCH);N=0.;TB=0;SB=0;correct=0;nt=0;m.eval()
 for j in range(EVAL_BATCHES):
  x,y,mask,tbytes,sbytes=encode_batch(tok,raw,ss[j*BATCH:(j+1)*BATCH]);z=m(x);ce=F.cross_entropy(z.reshape(-1,m.V),y.reshape(-1),reduction='none').view_as(y);N+=float((ce*mask).sum());TB+=tbytes;SB+=sbytes;correct+=int(((z.argmax(-1)==y)&mask).sum());nt+=int(mask.sum())
 return {'bpb':N/TB/math.log(2),'nats_per_byte':N/TB,'token_top1':correct/nt,'target_bytes':TB,'source_bytes':SB,'target_tokens':nt}
@torch.no_grad()
def runtime(m,tok,raw,seed,reps=20):
 ss=starts_for(raw,seed+70000,BATCH);x,y,mask,tb,sb=encode_batch(tok,raw,ss);m.eval();[m(x) for _ in range(4)];vv=[]
 for _ in range(reps):t=time.perf_counter();m(x);vv.append(time.perf_counter()-t)
 vv.sort();med=vv[len(vv)//2];return {'median_s':med,'source_byte_s':sb/med,'target_token_s':int(mask.sum())/med,'padded_seq_len':x.shape[1]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='11,29,47');ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);train=Path(a.train).read_bytes();tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};toks=build_toks(train,out);rows=[]
 for s in map(int,a.seeds.split(',')):
  for name,tok in toks.items():
   for top in ('dense','r4'):
    print('RUN',s,name,top,flush=True);m,tm=train_one(name,top,s,tok,train);r={'seed':s,'tokenizer':name,'topology':top,'model':f'{top.upper()}_{name}','params':pc(m),**tm}
    for sp,raw in tests.items():
     e=evaluate(m,tok,raw,s);r.update({f'{sp}_{k}':v for k,v in e.items()})
    rt=runtime(m,tok,tests[next(iter(tests))],s);r.update({'rt_'+k:v for k,v in rt.items()});rows.append(r);print(json.dumps({'model':r['model'],'seed':s,'train_bytes':r['train_source_bytes'],**{sp:r[f'{sp}_bpb'] for sp in tests},'byte_s':r['rt_source_byte_s']},indent=2),flush=True)
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with open(out/'01_PER_SEED.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 agg=[];metrics=[f'{sp}_bpb' for sp in tests]+['rt_source_byte_s','rt_target_token_s','train_s','train_source_bytes']
 for model in sorted({r['model'] for r in rows}):
  rr=[r for r in rows if r['model']==model];z={'model':model,'n':len(rr),'params':rr[0]['params']}
  for k in metrics:
   v=[r[k] for r in rr];mu=sum(v)/len(v);z[k+'_mean']=mu;z[k+'_sd']=(sum((x-mu)**2 for x in v)/len(v))**.5
  z['mean_eval_bpb']=sum(z[f'{sp}_bpb_mean'] for sp in tests)/len(tests);agg.append(z)
 agg.sort(key=lambda z:z['mean_eval_bpb']);(out/'00_RESULTS.json').write_text(json.dumps({'protocol':{'ctx_source_bytes':CTX_BYTES,'steps':STEPS,'batch':BATCH,'params':999978,'tokenizers':['BYTE256','LEXDICT2048','BPE2048','UNIGRAM2048'],'note':'tokenizers trained on train only; LM exposure and source-byte context matched'},'aggregate':agg,'per_seed':rows},indent=2))
 with open(out/'02_AGGREGATE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
 (out/'README_RU.md').write_text('# NEXUS R5.3 SOURCE-MATCHED TOKENIZER LM\n\nAll models: exactly 999,978 trainable params. Same random 128-source-byte windows, same optimizer steps, same seeds. Primary metric: BPB. Tokenizers are fitted on train only.\n\n'+'\n'.join(f"- {z['model']}: mean eval BPB {z['mean_eval_bpb']:.4f}; source throughput {z['rt_source_byte_s_mean']:.0f} B/s" for z in agg))
 print('DONE',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__':main()
