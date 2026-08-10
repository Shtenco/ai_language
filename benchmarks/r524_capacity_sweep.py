#!/usr/bin/env python3
import argparse,json,math,random,re,time
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB,TARGET,SEED=4096,48,20260810
DATA_SEED=SEED+524
CONFIGS={
 '3M':dict(d=192,h=6,l=6,f=570,expected=2998620),
 '8M':dict(d=304,h=8,l=8,f=768,expected=7966688),
 '11M':dict(d=336,h=8,l=10,f=840,expected=11576208),
}
PROMPTS=['Вечером он вышел из дома и','Наука развивается потому, что','Москва — это город, в котором','Человек посмотрел в окно и сказал:','Искусственный интеллект может помочь человеку','Когда наступила весна,','Если система получила сигнал, то','Хорошее доказательство должно опираться на','Россия занимает большую территорию, поэтому','Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','Причина отличается от простой корреляции тем, что','Для достижения цели система должна','После ошибки программа изменила своё состояние и','В книге автор рассказывает о том, как','Утром город проснулся, и на улицах']

class Tok:
 def __init__(self,path): self.sp=spm.SentencePieceProcessor(model_file=str(path))
 def enc(self,s): return self.sp.encode(s,out_type=int)
 def dec(self,ids): return self.sp.decode([int(x) for x in ids])

def rope(q,k):
 dh=q.shape[-1];pos=torch.arange(q.shape[-2],device=q.device,dtype=q.dtype);inv=1/(10000**(torch.arange(0,dh,2,device=q.device,dtype=q.dtype)/dh));a=torch.outer(pos,inv);c=a.cos()[None,None];s=a.sin()[None,None]
 def r(x):
  e,o=x[...,0::2],x[...,1::2];y=torch.empty_like(x);y[...,0::2]=e*c-o*s;y[...,1::2]=e*s+o*c;return y
 return r(q),r(k)

class Block(nn.Module):
 def __init__(self,d,h,f):
  super().__init__();self.h=h;self.n1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.p=nn.Linear(d,d);self.n2=nn.LayerNorm(d);self.f1=nn.Linear(d,f);self.f2=nn.Linear(f,d)
 def forward(self,x):
  B,L,D=x.shape;z=self.qkv(self.n1(x)).view(B,L,3,self.h,D//self.h).permute(2,0,3,1,4);q,k,v=z[0],z[1],z[2];q,k=rope(q,k);a=F.scaled_dot_product_attention(q,k,v,is_causal=True);x=x+self.p(a.transpose(1,2).contiguous().view(B,L,D));return x+self.f2(F.gelu(self.f1(self.n2(x))))

class LM(nn.Module):
 def __init__(self,c):
  super().__init__();d=c['d'];self.e=nn.Embedding(VOCAB,d);self.b=nn.ModuleList([Block(d,c['h'],c['f']) for _ in range(c['l'])]);self.n=nn.LayerNorm(d);self.o=nn.Linear(d,VOCAB,bias=False);self.o.weight=self.e.weight
 def forward(self,x):
  x=self.e(x)
  for b in self.b:x=b(x)
  return self.o(self.n(x))

def nparams(m): return sum(p.numel() for p in m.parameters())
def load_docs(path,tok):
 out=[]
 for line in Path(path).read_text(encoding='utf-8').splitlines():
  if line.strip():
   ids=tok.enc(json.loads(line)['text'])
   if len(ids)>=96:out.append(ids)
 return out

def sample(rng,docs):
 d=rng.choice(docs);L=rng.randint(4,48);s=rng.randrange(0,len(d)-TARGET-L+1);return d[s:s+L],d[s+L:s+L+TARGET]
def pack(batch):
 seq=[c+t for c,t in batch];starts=[len(c)-1 for c,_ in batch];mx=max(map(len,seq));xs=[];ys=[];ms=[]
 for s,st in zip(seq,starts):
  q=s+[0]*(mx-len(s));x,y=q[:-1],q[1:];m=[0]*len(x)
  for j in range(st,min(st+TARGET,len(m))):m[j]=1
  xs.append(x);ys.append(y);ms.append(m)
 return torch.tensor(xs),torch.tensor(ys),torch.tensor(ms,dtype=torch.bool)
def lrf(s,n):
 w=max(32,int(.05*n))
 if s<w:return (s+1)/w
 p=(s-w)/max(1,n-w);return .5*(1+math.cos(math.pi*min(1,p)))

def eval_bpb(model,tok,ids,ctx,seed,n=256):
 rng=random.Random(seed);q=[]
 for _ in range(n):
  s=rng.randrange(0,len(ids)-ctx-TARGET);q.append((ids[s:s+ctx],ids[s+ctx:s+ctx+TARGET]))
 nll=bts=cor=toks=0
 model.eval()
 with torch.no_grad():
  for o in range(0,n,16):
   zq=q[o:o+16];x,y,m=pack(zq);z=model(x);ce=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y);nll+=float((ce*m).sum());cor+=int(((z.argmax(-1)==y)&m).sum());toks+=int(m.sum());bts+=sum(len(tok.dec(t).encode()) for _,t in zq)
 return {'bpb':nll/max(1,bts)/math.log(2),'token_top1':cor/max(1,toks)}

def metrics(text):
 w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))];return {'words':len(w),'unique_word_ratio':len(set(x.lower() for x in w))/max(1,len(w)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0,'cyrillic_share':sum(bool(re.search(r'[А-Яа-яЁё]',x)) for x in w)/max(1,len(w))}
@torch.no_grad()
def generate(model,tok,prompt,kind,seed,new=64):
 ids=tok.enc(prompt);g=torch.Generator().manual_seed(seed);out=[];model.eval()
 for _ in range(new):
  z=model(torch.tensor([ids]))[0,-1]
  if kind=='greedy':t=int(z.argmax())
  else:
   p=F.softmax(z/.82,dim=-1);v,ix=torch.sort(p,descending=True);keep=torch.cumsum(v,0)<=.92;keep[0]=True;v,ix=v[keep],ix[keep];v=v/v.sum();t=int(ix[torch.multinomial(v,1,generator=g)])
  ids.append(t);out.append(t)
 text=tok.dec(out);return {'prompt':prompt,'decode':kind,'continuation':text,**metrics(text)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',choices=CONFIGS,required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=8192);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--lr',type=float,default=5e-4);a=ap.parse_args();torch.set_num_threads(a.threads);random.seed(SEED);torch.manual_seed(SEED)
 tok=Tok(a.tokenizer_model);assert tok.sp.unk_id()==0 and tok.sp.pad_id()==-1;docs=load_docs(a.train_docs,tok);assert len(docs)>=100;cfg=CONFIGS[a.config];model=LM(cfg);assert nparams(model)==cfg['expected'];opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(DATA_SEED);hist=[];tokens=rawbytes=0;t0=time.perf_counter();model.train()
 for step in range(a.steps):
  q=[sample(rng,docs) for _ in range(a.batch)];x,y,m=pack(q);z=model(x);ce=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y);loss=(ce*m).sum()/int(m.sum());opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);fac=lrf(step,a.steps)
  for pg in opt.param_groups:pg['lr']=a.lr*fac
  opt.step();hist.append(float(loss.detach()));tokens+=int(m.sum());rawbytes+=sum(len(tok.dec(t).encode()) for _,t in q)
  if (step+1)%512==0:print('TRAIN',a.config,step+1,'npt',sum(hist[-128:])/128,'MB',rawbytes/1048576,'tok/s',tokens/max(1,time.perf_counter()-t0),flush=True)
 tests={Path(p).stem:tok.enc(Path(p).read_text(encoding='utf-8')) for p in a.tests};ev={name:{str(c):eval_bpb(model,tok,ids,c,SEED+i*101+c) for c in (8,16,32,48)} for i,(name,ids) in enumerate(tests.items())};gens=[generate(model,tok,p,d,SEED+7000+i) for i,p in enumerate(PROMPTS) for d in ('greedy','sample')];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r={'format':'nexus-r524-capacity-sweep/3','protocol':{'config':a.config,'params':nparams(model),'d_model':cfg['d'],'heads':cfg['h'],'layers':cfg['l'],'ff':cfg['f'],'tokenizer':'fixed lossless Unigram4096','surface_prefix':'NONE','graph_prefix':'NONE','left_padding':'NONE','batch_padding':'right of complete context+target only','eligible_docs':'token length >=96','context_train':'uniform 4..48','target_tokens':48,'steps':a.steps,'batch':a.batch,'seed':SEED,'data_seed':DATA_SEED,'data_order':'IDENTICAL across 3M/8M/11M','lr':a.lr},'training':{'seconds':time.perf_counter()-t0,'target_tokens':tokens,'target_bytes':rawbytes,'last128_npt':sum(hist[-128:])/128},'eval':ev,'generation':gens};(out/'00_R524_RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nuniq={g['unique_word_ratio']:.3f} rep3={g['repeated_trigram_rate']:.3f}" for g in gens),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':r['protocol']},out/f'R524_{a.config}.pt');print(json.dumps({'protocol':r['protocol'],'training':r['training'],'eval':ev},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
