#!/usr/bin/env python3
import argparse,csv,json,math,random,re,time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import r52_tokenizer_tournament as tt

TARGET=3_000_000
MAX_TOK=192
CTX_BYTES=256
SWEEP_STEPS=512
SWEEP_BATCH=8
FINAL_STEPS=4096
FINAL_BATCH=16
EVAL_BATCHES=32
# Fixed total-budget sweep. Parameter spread is only 30 parameters.
SPECS={
 4096:(256,8,1338),   # 2,999,924
 8192:(216,8,937),    # 2,999,954
 16384:(138,6,1006),  # 2,999,924
}
PROMPTS=[
 'Вечером он вышел из дома и',
 'Наука развивается потому, что',
 'Москва — это город, в котором',
 'Человек посмотрел в окно и сказал:',
 'Искусственный интеллект может помочь человеку',
 'Когда наступила весна,',
]

def set_seed(s):random.seed(s);torch.manual_seed(s)
def pc(m):return sum(p.numel() for p in m.parameters())

def smask(L,w=24,k=6):
 m=torch.zeros(L,L,dtype=torch.bool)
 for i in range(L):
  m[i,max(0,i-w+1):i+1]=1;x=(i+1)*0x9E3779B1
  for q in range(k):
   x=(1664525*x+1013904223+q*97)&0xffffffff;m[i,x%(i+1)]=1
  m[i,0]=1
 return m

class Block(nn.Module):
 def __init__(self,d,f):
  super().__init__();self.a=nn.LayerNorm(d);self.q=nn.Linear(d,3*d);self.p=nn.Linear(d,d);self.b=nn.LayerNorm(d);self.f1=nn.Linear(d,f);self.f2=nn.Linear(f,d)

class LM(nn.Module):
 def __init__(self,V,d,h,f):
  super().__init__();assert d%h==0
  self.V=V;self.d=d;self.hn=h;self.f=f
  self.e=nn.Embedding(V,d);self.pos=nn.Embedding(MAX_TOK,d);self.bs=nn.ModuleList([Block(d,f) for _ in range(2)]);self.l=nn.LayerNorm(d);self.head=nn.Linear(d,V,bias=False);self.head.weight=self.e.weight
 def forward(self,x):
  B,L=x.shape;d=self.d;H=self.hn;hd=d//H
  y=self.e(x)+self.pos(torch.arange(L,device=x.device))[None]
  ma=smask(L).to(x.device)[None,None]
  for b in self.bs:
   z=b.q(b.a(y)).view(B,L,3,H,hd).permute(2,0,3,1,4);q,k,v=z
   a=F.scaled_dot_product_attention(q,k,v,attn_mask=ma)
   y=y+b.p(a.transpose(1,2).contiguous().view(B,L,d))
   z=F.gelu(b.f1(b.b(y)))
   # R4 conditional route. As in earlier experiments this is logical gating,
   # not yet a physically sparse FFN kernel.
   g=torch.arange(self.f,device=x.device)*4//self.f
   z=z*(g[None,None,:]!=(x%4)[:,:,None])
   y=y+b.f2(z)
  return self.head(self.l(y))

def safe_span(raw,start,n=CTX_BYTES):
 start=min(start,max(0,len(raw)-2))
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
 seq=[];spans=[];firstlens=[];trunc=0
 for st in starts:
  b=safe_span(raw,st);ids=tok.enc(b)
  if len(ids)<2:continue
  if len(ids)>MAX_TOK:
   ids=ids[:MAX_TOK];b=tok.dec(ids);trunc+=1
  seq.append(ids);spans.append(b);firstlens.append(len(tok.dec([ids[0]])))
 L=max(len(x) for x in seq);xx=torch.zeros(len(seq),L-1,dtype=torch.long);yy=torch.zeros_like(xx);mask=torch.zeros_like(xx,dtype=torch.bool)
 for i,ids in enumerate(seq):
  n=len(ids)-1;xx[i,:n]=torch.tensor(ids[:-1]);yy[i,:n]=torch.tensor(ids[1:]);mask[i,:n]=1
 target_bytes=sum(max(1,len(b)-fl) for b,fl in zip(spans,firstlens));source_bytes=sum(len(b) for b in spans)
 return xx,yy,mask,target_bytes,source_bytes,trunc

def lr_factor(step,total,warmup):
 if step<warmup:return max(.05,(step+1)/warmup)
 q=(step-warmup)/max(1,total-warmup)
 return .5*(1+math.cos(math.pi*min(1,q)))

def train_model(V,seed,tok,raw,steps,batch,base_lr=1.5e-3):
 d,h,f=SPECS[V];set_seed(seed);m=LM(V,d,h,f);params=pc(m)
 opt=torch.optim.AdamW(m.parameters(),lr=base_lr,weight_decay=.01,betas=(.9,.95))
 ss=starts_for(raw,seed+10000,steps*batch);losses=[];nb=0;nt=0;t0=time.perf_counter();m.train();warm=max(32,steps//32)
 for j in range(steps):
  x,y,mask,tbytes,sbytes,tr=encode_batch(tok,raw,ss[j*batch:(j+1)*batch]);nt+=tr
  for pg in opt.param_groups:pg['lr']=base_lr*lr_factor(j,steps,warm)
  opt.zero_grad(set_to_none=True);z=m(x);ce=F.cross_entropy(z.reshape(-1,V),y.reshape(-1),reduction='none').view_as(y);loss=(ce*mask).sum()/tbytes
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss));nb+=sbytes
  if (j+1)%256==0:print('TRAIN',V,seed,j+1,'loss_npb',sum(losses[-64:])/min(64,len(losses)),'source_MB',nb/1048576,flush=True)
 return m,{'params':params,'train_s':time.perf_counter()-t0,'train_source_bytes':nb,'truncated_spans':nt,'loss_nats_per_byte_last64':sum(losses[-64:])/min(64,len(losses))}

@torch.no_grad()
def evaluate(m,tok,raw,seed,batches=EVAL_BATCHES,batch=8):
 ss=starts_for(raw,seed+50000,batches*batch);N=0.;TB=0;SB=0;correct=0;tokn=0;trunc=0;m.eval()
 for j in range(batches):
  x,y,mask,tbytes,sbytes,tr=encode_batch(tok,raw,ss[j*batch:(j+1)*batch]);trunc+=tr;z=m(x);ce=F.cross_entropy(z.reshape(-1,m.V),y.reshape(-1),reduction='none').view_as(y);N+=float((ce*mask).sum());TB+=tbytes;SB+=sbytes;correct+=int(((z.argmax(-1)==y)&mask).sum());tokn+=int(mask.sum())
 return {'bpb':N/TB/math.log(2),'nats_per_byte':N/TB,'token_top1':correct/tokn,'target_bytes':TB,'source_bytes':SB,'target_tokens':tokn,'truncated_spans':trunc}

@torch.no_grad()
def runtime(m,tok,raw,seed,reps=20):
 ss=starts_for(raw,seed+70000,8);x,y,mask,tb,sb,tr=encode_batch(tok,raw,ss);m.eval();[m(x) for _ in range(3)];vv=[]
 for _ in range(reps):
  t=time.perf_counter();m(x);vv.append(time.perf_counter()-t)
 vv.sort();med=vv[len(vv)//2]
 return {'median_s':med,'source_byte_s':sb/med,'target_token_s':int(mask.sum())/med,'padded_seq_len':x.shape[1]}

@torch.no_grad()
def generate(m,tok,prompt,seed,mode='sample',max_new=160,temp=.75,topk=40):
 set_seed(seed);ids=tok.enc(prompt.encode('utf-8'));m.eval()
 for _ in range(max_new):
  x=torch.tensor([ids[-MAX_TOK:]],dtype=torch.long);log=m(x)[0,-1]
  if mode=='greedy':nxt=int(log.argmax())
  else:
   q=log/max(temp,1e-5);k=min(topk,len(q));v,ix=torch.topk(q,k);p=F.softmax(v,dim=-1);nxt=int(ix[torch.multinomial(p,1)])
  ids.append(nxt)
 raw=tok.dec(ids);text=raw.decode('utf-8','replace')
 cont=text[len(prompt):] if text.startswith(prompt) else text
 return text,cont

def text_metrics(s):
 letters=re.findall(r'[A-Za-zА-Яа-яЁё]',s);cy=[x for x in letters if re.match(r'[А-Яа-яЁё]',x)]
 words=re.findall(r'[А-Яа-яЁё]+',s.lower());tri=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
 return {'chars':len(s),'words':len(words),'cyrillic_letter_share':len(cy)/max(1,len(letters)),'unique_word_ratio':len(set(words))/max(1,len(words)),'unique_word_trigram_ratio':len(set(tri))/max(1,len(tri)),'replacement_chars':s.count('�')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='11,29,47');ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);train=Path(a.train).read_bytes();tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};seeds=list(map(int,a.seeds.split(',')))
 toks={v:tt.train_sp(train,v,'unigram',out) for v in SPECS}
 probe='Привет, мир! Lossless проверка №123.\n'.encode('utf-8')
 for v,t in toks.items():assert t.dec(t.enc(probe))==probe,v
 param_table={v:pc(LM(v,*SPECS[v])) for v in SPECS};assert max(param_table.values())-min(param_table.values())<=30,param_table
 rows=[]
 for s in seeds:
  for V,tok in toks.items():
   print('SWEEP RUN',s,V,flush=True);m,tm=train_model(V,s,tok,train,SWEEP_STEPS,SWEEP_BATCH);row={'seed':s,'vocab':V,'model':f'R4_3M_UNIGRAM{V}',**tm}
   for sp,raw in tests.items():row.update({f'{sp}_{k}':v for k,v in evaluate(m,tok,raw,s).items()})
   row.update({'rt_'+k:v for k,v in runtime(m,tok,tests[next(iter(tests))],s).items()});rows.append(row)
   print(json.dumps({'seed':s,'vocab':V,'params':row['params'],**{sp:row[f'{sp}_bpb'] for sp in tests},'B_s':row['rt_source_byte_s']},ensure_ascii=False),flush=True)
 bpb_cols=[f'{sp}_bpb' for sp in tests];ru_cols=[k for k in bpb_cols if ('ru_' in k.lower() or 'synt' in k.lower() or 'gsd' in k.lower())]
 for r in rows:r['mean_eval_bpb']=sum(r[k] for k in bpb_cols)/len(bpb_cols);r['mean_ru_bpb']=sum(r[k] for k in ru_cols)/max(1,len(ru_cols))
 agg=[]
 for V in SPECS:
  rr=[x for x in rows if x['vocab']==V];z={'vocab':V,'model':f'R4_3M_UNIGRAM{V}','n':len(rr),'params':rr[0]['params']}
  for k in bpb_cols+['mean_eval_bpb','mean_ru_bpb','rt_source_byte_s','train_s','train_source_bytes','truncated_spans']:
   vals=[x[k] for x in rr];mu=sum(vals)/len(vals);z[k+'_mean']=mu;z[k+'_sd']=(sum((q-mu)**2 for q in vals)/len(vals))**.5
  agg.append(z)
 agg.sort(key=lambda z:z['mean_ru_bpb_mean']);winner=int(agg[0]['vocab']);print('WINNER',winner,flush=True)

 # Fixed-seed final Russian model: no best-seed cherry-picking.
 final_seed=20260809;final_tok=toks[winner];final,ft=train_model(winner,final_seed,final_tok,train,FINAL_STEPS,FINAL_BATCH,base_lr=1.2e-3)
 final_eval={sp:evaluate(final,final_tok,raw,final_seed,batches=64,batch=8) for sp,raw in tests.items()}
 final_rt=runtime(final,final_tok,tests[next(iter(tests))],final_seed,reps=30)
 samples=[]
 for i,p in enumerate(PROMPTS):
  for mode in ('greedy','sample'):
   text,cont=generate(final,final_tok,p,final_seed+i*17+(0 if mode=='greedy' else 1),mode=mode)
   samples.append({'prompt':p,'mode':mode,'text':text,'continuation':cont,**text_metrics(cont)})

 result={'protocol':{'target_params':TARGET,'max_tokens':MAX_TOK,'source_context_bytes':CTX_BYTES,'sweep_steps':SWEEP_STEPS,'sweep_batch':SWEEP_BATCH,'final_steps':FINAL_STEPS,'final_batch':FINAL_BATCH,'topology':'R4','tokenizer':'lossless SentencePiece Unigram + byte fallback','specs':{str(v):{'d':SPECS[v][0],'heads':SPECS[v][1],'ff':SPECS[v][2],'params':param_table[v]} for v in SPECS}},'sweep_aggregate':agg,'sweep_per_seed':rows,'winner_vocab':winner,'final_training':ft,'final_eval':final_eval,'final_runtime':final_rt,'generation_samples':samples}
 (out/'00_R56_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 keys=list(rows[0]);
 with open(out/'01_SWEEP_PER_SEED.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 with open(out/'02_SWEEP_AGGREGATE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 (out/'03_RUSSIAN_GENERATIONS.txt').write_text('\n\n'.join(f"[{x['mode']}] {x['prompt']}\n{x['text']}\nMETRICS: {json.dumps({k:x[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','unique_word_trigram_ratio','replacement_chars')},ensure_ascii=False)}" for x in samples),encoding='utf-8')
 torch.save({'state_dict':final.state_dict(),'vocab':winner,'spec':SPECS[winner],'max_tok':MAX_TOK,'seed':final_seed},out/'R56_3M_RUSSIAN_FINAL.pt')
 (out/'README_RU.md').write_text('# NEXUS R5.6 — 3M Russian Adequacy\n\n3M-parameter R4 sweep over lossless Unigram 4096/8192/16384, then fixed-seed long training of the Russian-BPB winner and free autoregressive Russian generation. Primary adequacy evidence is held-out BPB plus raw free-running samples; no external LLM judge and no anti-repeat postprocessing are used.\n\nWinner vocab: **'+str(winner)+'**\n\nFinal eval:\n'+ '\n'.join(f"- {k}: BPB={v['bpb']:.4f}, top1={v['token_top1']:.4f}" for k,v in final_eval.items()),encoding='utf-8')
 print('FINAL',json.dumps({'winner':winner,'params':ft['params'],'train_MB':ft['train_source_bytes']/1048576,'eval_bpb':{k:v['bpb'] for k,v in final_eval.items()},'runtime_B_s':final_rt['source_byte_s']},ensure_ascii=False,indent=2),flush=True)
 for x in samples:print('\nGEN',x['mode'],x['prompt'],'\n',x['text'],flush=True)

if __name__=='__main__':main()
