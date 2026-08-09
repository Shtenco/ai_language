#!/usr/bin/env python3
import argparse,csv,hashlib,json,math,random,re,time
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_CTX=128; TRAIN_TOKENS=262144
SPECS={
'T0_BYTE256':(256,195,5,789,'dense','byte'),
'R4_BYTE256':(256,195,5,789,'r4','byte'),
'T0_DICT2048':(2048,135,5,1029,'dense','dict'),
'R4_DICT2048':(2048,135,5,1029,'r4','dict'),
}

def seed(s): random.seed(s); torch.manual_seed(s)
class ByteTok:
 vocab_size=256; lens=torch.ones(256,dtype=torch.long)
 def enc(self,b): return torch.tensor(list(b),dtype=torch.long)
 def dec(self,x): return bytes(map(int,x))
class DictTok:
 def __init__(self,toks):
  self.toks=toks; self.vocab_size=len(toks); self.lens=torch.tensor([len(x) for x in toks]); self.trie={}
  for i,t in enumerate(toks[256:],256):
   n=self.trie
   for b in t:n=n.setdefault(b,{})
   n[-1]=i
 @classmethod
 def train(cls,b,v=2048):
  text=b.decode('utf-8'); c=Counter()
  # our old dictionary idea: frequent lexical chunks + reusable affixes, scored by byte saving
  for s in re.findall(r" ?\w+| ?[^\w\s]+|\s+",text,re.UNICODE):
   q=s.encode();
   if 2<=len(q)<=64:c[q]+=1
   if 6<=len(q)<=64:
    for n in (3,4,5,6,8,12):
     if n<len(q):c[q[:n]]+=1;c[q[-n:]]+=1
  ranked=sorted(c.items(),key=lambda z:(-(z[1]-1)*(len(z[0])-1),-z[1],-len(z[0]),z[0]))
  seen={bytes([i]) for i in range(256)}; extra=[]
  for q,n in ranked:
   if n>=2 and q not in seen: extra.append(q);seen.add(q)
   if len(extra)==v-256:break
  assert len(extra)==v-256
  return cls([bytes([i]) for i in range(256)]+extra)
 def enc(self,b):
  o=[];i=0
  while i<len(b):
   n=self.trie;j=i;best=None;bj=i
   while j<len(b) and b[j] in n:
    n=n[b[j]];j+=1
    if -1 in n:best=n[-1];bj=j
   if best is None:o.append(b[i]);i+=1
   else:o.append(best);i=bj
  return torch.tensor(o,dtype=torch.long)
 def dec(self,x):return b''.join(self.toks[int(i)] for i in x)
 def meta(self):
  blob=b'\0'.join(self.toks)
  return {'vocab':2048,'base_bytes':256,'dictionary':1792,'sha256':hashlib.sha256(blob).hexdigest(),'preview':[{'id':i,'text':t.decode('utf-8','replace'),'hex':t.hex()} for i,t in list(enumerate(self.toks[256:306],256))]}

def smask(L,w=16,k=4):
 m=torch.zeros(L,L,dtype=torch.bool)
 for i in range(L):
  m[i,max(0,i-w+1):i+1]=1;x=(i+1)*0x9E3779B1
  for s in range(k):x=(1664525*x+1013904223+s*97)&0xffffffff;m[i,x%(i+1)]=1
  m[i,0]=1
 return m
class Block(nn.Module):
 def __init__(self,d,f):
  super().__init__();self.a=nn.LayerNorm(d);self.q=nn.Linear(d,3*d);self.p=nn.Linear(d,d);self.b=nn.LayerNorm(d);self.f1=nn.Linear(d,f);self.f2=nn.Linear(f,d)
class LM(nn.Module):
 def __init__(self,s):
  super().__init__();V,D,H,Fd,top,_=s;self.s=s;self.e=nn.Embedding(V,D);self.pos=nn.Embedding(MAX_CTX,D);self.bs=nn.ModuleList([Block(D,Fd) for _ in range(2)]);self.l=nn.LayerNorm(D);self.h=nn.Linear(D,V,bias=False);self.h.weight=self.e.weight
 def mask(self,x):
  B,L=x.shape;ca=torch.tril(torch.ones(L,L,dtype=torch.bool,device=x.device))
  return ca[None,None] if self.s[4]=='dense' else smask(L).to(x.device)[None,None]
 def forward(self,x):
  V,D,H,Fd,top,_=self.s;B,L=x.shape;hd=D//H;y=self.e(x)+self.pos(torch.arange(L))[None];ma=self.mask(x)
  for b in self.bs:
   z=b.q(b.a(y)).view(B,L,3,H,hd).permute(2,0,3,1,4);q,k,v=z
   a=F.scaled_dot_product_attention(q,k,v,attn_mask=ma);y=y+b.p(a.transpose(1,2).contiguous().view(B,L,D))
   z=F.gelu(b.f1(b.b(y)))
   if top!='dense':
    g=torch.arange(Fd)*4//Fd;z=z*(g[None,None,:]!=(x%4)[:,:,None])
   y=y+b.f2(z)
  return self.h(self.l(y))
def pc(m):return sum(p.numel() for p in m.parameters())
def plan(n,s):
 g=random.Random(s);o=[]
 for L,B in ((64,16),(128,8)):
  for _ in range(128):o.append((L,[g.randrange(n-L-1) for _ in range(B)]))
 return o
def batch(d,L,st):return torch.stack([d[i:i+L] for i in st]),torch.stack([d[i+1:i+L+1] for i in st])
def train(name,s,d):
 seed(s);m=LM(SPECS[name]);assert pc(m)==999978;op=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01,betas=(.9,.95));ls=[];t=time.perf_counter()
 for L,st in plan(len(d),s+10000):
  x,y=batch(d,L,st);op.zero_grad(set_to_none=True);z=m(x);loss=F.cross_entropy(z.reshape(-1,SPECS[name][0]),y.reshape(-1));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);op.step();ls.append(float(loss))
 return m,{'train_s':time.perf_counter()-t,'train_loss_last32':sum(ls[-32:])/32}
@torch.no_grad()
def ev(m,d,lens,L,s):
 g=random.Random(s+L);N=0;bits=0;tok=0;correct=0
 for _ in range(24):
  x,y=batch(d,L,[g.randrange(len(d)-L-1) for _ in range(8)]);z=m(x);n=float(F.cross_entropy(z.reshape(-1,m.s[0]),y.reshape(-1),reduction='sum'));N+=n;tok+=y.numel();bits+=int(lens[y].sum());correct+=int((z.argmax(-1)==y).sum())
 return {'token_ppl':math.exp(min(N/tok,20)),'token_top1':correct/tok,'bpb':N/bits/math.log(2),'bytes_per_token':bits/tok}
@torch.no_grad()
def speed(m,d,lens,L=128,B=16):
 x,_=batch(d,L,[i*17%(len(d)-L-1) for i in range(B)]);m.eval();[m(x) for _ in range(4)];a=[]
 for _ in range(20):t=time.perf_counter();m(x);a.append(time.perf_counter()-t)
 a.sort();med=a[len(a)//2];return {'tok_s':B*L/med,'byte_s':int(lens[x].sum())/med}
def main():
 p=argparse.ArgumentParser();
 for q in ('train','valid','test','shift','out'):p.add_argument('--'+q,required=True)
 p.add_argument('--seeds',default='11,29,47');p.add_argument('--threads',type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads)
 raw={k:Path(getattr(a,k)).read_bytes() for k in ('train','valid','test','shift')};bt=ByteTok();dt=DictTok.train(raw['train']);enc={'byte':{},'dict':{}};stats={}
 for n,t in [('byte',bt),('dict',dt)]:
  stats[n]={}
  for sp,b in raw.items():
   x=t.enc(b);assert t.dec(x.tolist())==b;enc[n][sp]=x;stats[n][sp]={'raw_bytes':len(b),'tokens':len(x),'bytes_per_token':len(b)/len(x)}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'00_TOKENIZER.json').write_text(json.dumps({'dict':dt.meta(),'stats':stats,'note':'dictionary trained on train only; lossless byte fallback; cross-tokenizer metric is BPB, not token PPL'},ensure_ascii=False,indent=2))
 rows=[]
 for s in map(int,a.seeds.split(',')):
  for name,spec in SPECS.items():
   tn=spec[5];print('RUN',s,name,flush=True);m,tm=train(name,s,enc[tn]['train']);r={'seed':s,'model':name,'params':pc(m),'tokenizer':tn,'topology':spec[4],**tm}
   for sp in ('valid','test','shift'):
    for L in (64,128):
     for k,v in ev(m,enc[tn][sp],bt.lens if tn=='byte' else dt.lens,L,s+500).items():r[f'{sp}_{L}_{k}']=v
   for k,v in speed(m,enc[tn]['test'],bt.lens if tn=='byte' else dt.lens).items():r['rt128_'+k]=v
   rows.append(r);torch.save(m.state_dict(),out/f'{name}_seed{s}.pt');print(json.dumps({'model':name,'seed':s,'test_bpb':r['test_128_bpb'],'shift_bpb':r['shift_128_bpb'],'byte_s':r['rt128_byte_s']}),flush=True)
 agg=[]
 for name in SPECS:
  rr=[r for r in rows if r['model']==name];z={'model':name,'n':len(rr)}
  for k in ('test_128_bpb','valid_128_bpb','shift_128_bpb','test_128_token_ppl','test_128_token_top1','test_128_bytes_per_token','rt128_tok_s','rt128_byte_s','train_s'):
   v=[r[k] for r in rr];mu=sum(v)/len(v);z[k+'_mean']=mu;z[k+'_sd']=(sum((x-mu)**2 for x in v)/len(v))**.5
  agg.append(z)
 base=next(x for x in agg if x['model']=='R4_BYTE256')
 for z in agg:z['bpb_ratio_vs_R4_BYTE']=z['test_128_bpb_mean']/base['test_128_bpb_mean'];z['byte_s_ratio_vs_R4_BYTE']=z['rt128_byte_s_mean']/base['rt128_byte_s_mean']
 (out/'01_RESULTS.json').write_text(json.dumps({'aggregate':agg,'per_seed':rows},indent=2));
 for fn,arr in [('02_PER_SEED.csv',rows),('03_AGGREGATE.csv',agg)]:
  keys=[]
  for d in arr:
   for k in d:
    if k not in keys:keys.append(k)
  with open(out/fn,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(arr)
 (out/'README_RU.md').write_text('# NEXUS R5.1 DICT2048\n\nMatched 999,978-param 2x2 tokenizer × topology test. Primary cross-tokenizer metric: bits per original byte (BPB).\n\n'+ '\n'.join(f"- {z['model']}: test128 BPB {z['test_128_bpb_mean']:.4f}; shift BPB {z['shift_128_bpb_mean']:.4f}; {z['rt128_byte_s_mean']:.0f} byte/s" for z in agg))
 print('DONE',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__':main()
