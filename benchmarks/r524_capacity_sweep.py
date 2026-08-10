#!/usr/bin/env python3
import argparse,json,math,random,re,time
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB=4096;TARGET=48;SEED=20260810
CONFIGS={
 '3M':dict(d=192,h=6,l=6,f=570,expected=2998620),
 '8M':dict(d=304,h=8,l=8,f=768,expected=7966688),
 '11M':dict(d=336,h=8,l=10,f=840,expected=11576208),
}
PROMPTS=['Вечером он вышел из дома и','Наука развивается потому, что','Москва — это город, в котором','Человек посмотрел в окно и сказал:','Искусственный интеллект может помочь человеку','Когда наступила весна,','Если система получила сигнал, то','Хорошее доказательство должно опираться на','Россия занимает большую территорию, поэтому','Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','Причина отличается от простой корреляции тем, что','Для достижения цели система должна','После ошибки программа изменила своё состояние и','В книге автор рассказывает о том, как','Утром город проснулся, и на улицах']

class Tok:
 def __init__(self,path):self.sp=spm.SentencePieceProcessor(model_file=str(path));self.path=Path(path)
 def enc(self,s):return self.sp.encode(s,out_type=int)
 def dec(self,ids):return self.sp.decode([int(x) for x in ids])

def rope(q,k):
 dh=q.shape[-1];pos=torch.arange(q.shape[-2],device=q.device,dtype=q.dtype);inv=1.0/(10000**(torch.arange(0,dh,2,device=q.device,dtype=q.dtype)/dh));ang=torch.outer(pos,inv);c=ang.cos()[None,None,:,:];s=ang.sin()[None,None,:,:]
 def r(x):
  e,o=x[...,0::2],x[...,1::2];y=torch.empty_like(x);y[...,0::2]=e*c-o*s;y[...,1::2]=e*s+o*c;return y
 return r(q),r(k)

class Block(nn.Module):
 def __init__(self,d,h,f):
  super().__init__();self.d=d;self.h=h;self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.ln2=nn.LayerNorm(d);self.fc1=nn.Linear(d,f);self.fc2=nn.Linear(f,d)
 def forward(self,x):
  b,l,d=x.shape;z=self.ln1(x);qkv=self.qkv(z).view(b,l,3,self.h,d//self.h).permute(2,0,3,1,4);q,k,v=qkv[0],qkv[1],qkv[2];q,k=rope(q,k);a=F.scaled_dot_product_attention(q,k,v,is_causal=True);x=x+self.proj(a.transpose(1,2).contiguous().view(b,l,d));return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))

class LM(nn.Module):
 def __init__(self,c):
  super().__init__();d=c['d'];self.emb=nn.Embedding(VOCAB,d);self.blocks=nn.ModuleList([Block(d,c['h'],c['f']) for _ in range(c['l'])]);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,VOCAB,bias=False);self.head.weight=self.emb.weight
 def forward(self,ids):
  x=self.emb(ids)
  for b in self.blocks:x=b(x)
  return self.head(self.norm(x))

def pc(m):return sum(p.numel() for p in m.parameters())
def load_docs(path,tok):
 docs=[]
 for line in Path(path).read_text(encoding='utf-8').splitlines():
  if not line.strip():continue
  x=json.loads(line);ids=tok.enc(x['text'])
  # Every eligible document must support the full matched context range 4..48 plus 48 targets.
  if len(ids)>=TARGET+48:docs.append(ids)
 return docs

def sample(rng,docs):
 d=rng.choice(docs);L=rng.randint(4,48);start=rng.randrange(0,len(d)-TARGET-L+1);return d[start:start+L],d[start+L:start+L+TARGET]
def pack(batch):
 seq=[c+t for c,t in batch];starts=[len(c)-1 for c,t in batch];mx=max(map(len,seq));xs=[];ys=[];ms=[]
 for s,st in zip(seq,starts):
  q=s+[0]*(mx-len(s));x=q[:-1];y=q[1:];m=[0]*len(x)
  for j in range(st,min(st+TARGET,len(m))):m[j]=1
  xs.append(x);ys.append(y);ms.append(m)
 return torch.tensor(xs),torch.tensor(ys),torch.tensor(ms,dtype=torch.bool)
def lr_factor(step,steps):
 warm=max(32,int(.05*steps))
 if step<warm:return (step+1)/warm
 p=(step-warm)/max(1,steps-warm);return .5*(1+math.cos(math.pi*min(1,p)))

def eval_bpb(model,ids,ctx,n=256,seed=0,batch=16):
 rng=random.Random(seed);examples=[]
 for _ in range(n):
  st=rng.randrange(0,len(ids)-ctx-TARGET);examples.append((ids[st:st+ctx],ids[st+ctx:st+ctx+TARGET]))
 nll=0.;bts=0;corr=0;toks=0;model.eval()
 with torch.no_grad():
  for o in range(0,n,batch):
   q=examples[o:o+batch];x,y,m=pack(q);z=model(x);ce=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y);nll+=float((ce*m).sum());corr+=int(((z.argmax(-1)==y)&m).sum());toks+=int(m.sum());bts+=sum(max(1,len(tok_global.dec(t).encode('utf-8'))) for _,t in q)
 return {'bpb':nll/max(1,bts)/math.log(2),'token_top1':corr/max(1,toks)}

def met(text):
 w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))];return {'words':len(w),'unique_word_ratio':len(set(x.lower() for x in w))/max(1,len(w)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0.0,'cyrillic_share':sum(bool(re.search(r'[А-Яа-яЁё]',x)) for x in w)/max(1,len(w))}
@torch.no_grad()
def generate(model,tok,prompt,decode,seed,new=64):
 ids=tok.enc(prompt);g=torch.Generator().manual_seed(seed);out=[];model.eval()
 for _ in range(new):
  z=model(torch.tensor([ids]))[0,-1]
  if decode=='greedy':t=int(z.argmax())
  else:
   p=F.softmax(z/.82,dim=-1);v,ix=torch.sort(p,descending=True);cs=torch.cumsum(v,0);keep=cs<=.92;keep[0]=True;v=v[keep];ix=ix[keep];v=v/v.sum();t=int(ix[torch.multinomial(v,1,generator=g)])
  ids.append(t);out.append(t)
 text=tok.dec(out);return {'prompt':prompt,'decode':decode,'continuation':text,**met(text)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',choices=CONFIGS,required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=8192);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--lr',type=float,default=5e-4);a=ap.parse_args();torch.set_num_threads(a.threads);random.seed(SEED);torch.manual_seed(SEED)
 global tok_global;tok=Tok(a.tokenizer_model);tok_global=tok;assert tok.sp.unk_id()==0 and tok.sp.pad_id()==-1
 docs=load_docs(a.train_docs,tok);assert len(docs)>=100;cfg=CONFIGS[a.config];model=LM(cfg);assert pc(model)==cfg['expected'],(pc(model),cfg);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(SEED+pc(model));hist=[];tokens=0;rawbytes=0;t0=time.perf_counter();model.train()
 for step in range(a.steps):
  q=[sample(rng,docs) for _ in range(a.batch)];x,y,m=pack(q);opt.zero_grad(set_to_none=True);z=model(x);ce=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y);loss=(ce*m).sum()/max(1,int(m.sum()));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);fac=lr_factor(step,a.steps)
  for pg in opt.param_groups:pg['lr']=a.lr*fac
  opt.step();hist.append(float(loss.detach()));tokens+=int(m.sum());rawbytes+=sum(len(tok.dec(t).encode('utf-8')) for _,t in q)
  if (step+1)%512==0:print('TRAIN',a.config,step+1,'npt',sum(hist[-128:])/128,'MB',rawbytes/1048576,'tok/s',tokens/max(1,time.perf_counter()-t0),flush=True)
 tests={Path(p).stem:tok.enc(Path(p).read_text(encoding='utf-8')) for p in a.tests};ev={name:{str(c):eval_bpb(model,ids,c,256,SEED+i*101+c) for c in (8,16,32,48)} for i,(name,ids) in enumerate(tests.items())};gens=[]
 for i,p in enumerate(PROMPTS):
  for d in ('greedy','sample'):gens.append(generate(model,tok,p,d,SEED+7000+i))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);r={'format':'nexus-r524-capacity-sweep/2','protocol':{'config':a.config,'params':pc(model),'d_model':cfg['d'],'heads':cfg['h'],'layers':cfg['l'],'ff':cfg['f'],'tokenizer':'fixed lossless Unigram4096 from R5.12','surface_prefix':'NONE','graph_prefix':'NONE','left_padding':'NONE','batch_padding':'right of complete context+target only','eligible_docs':'at least 96 tokenizer tokens so every doc supports ctx4..48 + target48','context_train':'uniform 4..48','target_tokens':TARGET,'steps':a.steps,'batch':a.batch,'seed':SEED,'lr':a.lr},'training':{'seconds':time.perf_counter()-t0,'target_tokens':tokens,'target_bytes':rawbytes,'last128_npt':sum(hist[-128:])/max(1,len(hist[-128:]))},'eval':ev,'generation':gens};(out/'00_R524_RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nuniq={g['unique_word_ratio']:.3f} rep3={g['repeated_trigram_rate']:.3f}" for g in gens),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':r['protocol']},out/f'R524_{a.config}.pt');print(json.dumps({'protocol':r['protocol'],'training':r['training'],'eval':ev},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
