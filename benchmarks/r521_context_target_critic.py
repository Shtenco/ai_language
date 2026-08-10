#!/usr/bin/env python3
import argparse,json,random,re
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER';SEED=20260810;D=base.D_MODEL;FEAT=D*6
PROMPTS=base.PROMPTS+['Россия занимает большую территорию, поэтому','Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','Причина отличается от простой корреляции тем, что','Для достижения цели система должна','После ошибки программа изменила своё состояние и','В книге автор рассказывает о том, как','Утром город проснулся, и на улицах']

class Tok:
    def __init__(self,path):self.sp=spm.SentencePieceProcessor(model_file=str(path));self.model_path=Path(path);self.name='UNIGRAM4096-WARM'
    def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids):return self.sp.decode([int(x) for x in ids]).encode('utf-8')
    def vocab_bytes(self):return self.model_path.stat().st_size

class Critic(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(FEAT,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def hidden(model,ids):
    x=model.emb(ids)
    for b in model.blocks:x=b(x)
    return model.norm(x)

@torch.no_grad()
def features_batch(model,seqs,prefix_len=base.PREFIX_TOKENS,ctx_len=base.CONTEXT_TOKENS,tgt_len=base.TARGET_TOKENS):
    h=hidden(model,torch.tensor(seqs,dtype=torch.long));c=h[:,prefix_len:prefix_len+ctx_len,:];t=h[:,prefix_len+ctx_len:prefix_len+ctx_len+tgt_len,:]
    cm=c.mean(1);cl=c[:,-1,:];tm=t.mean(1);tl=t[:,-1,:]
    return torch.cat([cm,cl,tm,tl,cl*tm,(cl-tm).abs()],dim=-1).cpu()

def seq(ex,tok,memory,target):return list(base.prefix_ids(MODE,tok,ex['ctx_text'],memory,ex['meta']))+list(ex['ctx'])+list(target)
def repeat4(t):return (list(t[:4])*20)[:base.TARGET_TOKENS]
def shuffle4(t,rng):
    b=[list(t[i:i+4]) for i in range(0,base.TARGET_TOKENS,4)];rng.shuffle(b);return [x for q in b for x in q][:base.TARGET_TOKENS]

def dataset(model,examples,tok,memory,rng,batch=24):
    fs=[];ys=[];ts=[]
    for o in range(0,len(examples),batch):
        ex=examples[o:o+batch];ss=[];meta=[]
        for i,e in enumerate(ex):
            true=list(e['tgt'][:base.TARGET_TOKENS]);wrong=list(ex[(i+1)%len(ex)]['tgt'][:base.TARGET_TOKENS]) if len(ex)>1 else true[::-1]
            for typ,t,y in [('true',true,1.),('wrong_context',wrong,0.),('repeat4',repeat4(true),0.),('block_shuffle',shuffle4(true,rng),0.)]:ss.append(seq(e,tok,memory,t));meta.append((typ,y))
        fs.append(features_batch(model,ss));ys.extend(y for _,y in meta);ts.extend(t for t,_ in meta)
    return torch.cat(fs),torch.tensor(ys,dtype=torch.float32),ts

def pair_auc(pos,neg):return float(((pos[:,None]>neg[None,:]).float()+.5*(pos[:,None]==neg[None,:]).float()).mean())
def eval_scores(sc,types):
    out={};pos=sc[torch.tensor([t=='true' for t in types])]
    for typ in ('wrong_context','repeat4','block_shuffle'):
        neg=sc[torch.tensor([t==typ for t in types])];out[typ]={'auc_true_over_negative':pair_auc(pos,neg),'true_mean':float(pos.mean()),'negative_mean':float(neg.mean())}
    return out

def train(c,x,y,steps=2000):
    opt=torch.optim.AdamW(c.parameters(),lr=1e-3,weight_decay=.01);g=torch.Generator().manual_seed(SEED+521);hist=[]
    for _ in range(steps):
        ix=torch.randint(0,len(y),(128,),generator=g);z=c(x[ix]);loss=F.binary_cross_entropy_with_logits(z,y[ix]);opt.zero_grad();loss.backward();opt.step();hist.append(float(loss))
    return sum(hist[-100:])/100

@torch.no_grad()
def sample(model,tok,memory,prompt,seed,temp=.82,topp=.92):
    ctx=tok.enc(prompt.encode())[-base.CONTEXT_TOKENS:];ids=list(base.prefix_ids(MODE,tok,prompt,memory,{'kind':'main'}))+ctx;out=[];g=torch.Generator().manual_seed(seed)
    for _ in range(base.TARGET_TOKENS):
        p=F.softmax(model(torch.tensor([ids]))[0,-1]/temp,dim=-1);v,ix=torch.sort(p,descending=True);cs=torch.cumsum(v,0);keep=cs<=topp;keep[0]=True;v=v[keep];ix=ix[keep];v=v/v.sum();t=int(ix[torch.multinomial(v,1,generator=g)]);ids.append(t);out.append(t)
    return out

@torch.no_grad()
def feature_one(model,tok,memory,prompt,target):
    ctx=tok.enc(prompt.encode())[-base.CONTEXT_TOKENS:];p=list(base.prefix_ids(MODE,tok,prompt,memory,{'kind':'main'}));ids=p+ctx+list(target);h=hidden(model,torch.tensor([ids]));pl=len(p);clen=len(ctx);c=h[:,pl:pl+clen,:];t=h[:,pl+clen:,:];cm=c.mean(1);cl=c[:,-1,:];tm=t.mean(1);tl=t[:,-1,:];return torch.cat([cm,cl,tm,tl,cl*tm,(cl-tm).abs()],dim=-1).cpu()

def met(text):
    w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))]
    return {'words':len(w),'unique_word_ratio':len(set(x.lower() for x in w))/max(1,len(w)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0.0}

def rerank(model,c,tok,memory,n=12):
    rows=[]
    for i,prompt in enumerate(PROMPTS):
        q=[]
        for j in range(n):
            ids=sample(model,tok,memory,prompt,SEED+21000+i*100+j);score=float(c(feature_one(model,tok,memory,prompt,ids))[0]);text=tok.dec(ids).decode('utf-8','replace');q.append({'score':score,'text':text,**met(text)})
        rows.append({'prompt':prompt,'first':q[0],'best':max(q,key=lambda x:x['score']),'score_span':max(x['score'] for x in q)-min(x['score'] for x in q)})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);random.seed(SEED);torch.manual_seed(SEED);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tok=Tok(a.tokenizer_model);memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));model=base.LM();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['state_dict']);model.eval();assert base.param_count(model)==2998620
    for p in model.parameters():p.requires_grad_(False)
    rng=random.Random(SEED+521);train_ex=[base.make_main_example(rng,docs,tok) for _ in range(1024)];x,y,types=dataset(model,train_ex,tok,memory,rng);c=Critic();loss=train(c,x,y);c.eval();train_eval=eval_scores(c(x).detach(),types)
    held={}
    for i,path in enumerate(a.tests):
        ex=base.build_main_eval(tok,Path(path).read_bytes(),25210+i*53,n=192);xx,yy,tt=dataset(model,ex,tok,memory,random.Random(SEED+900+i));held[Path(path).stem]=eval_scores(c(xx).detach(),tt)
    rows=rerank(model,c,tok,memory,12);r={'format':'nexus-r521-context-target-critic/1','protocol':{'base':'R5.12 32K frozen','base_params':2998620,'critic_params':sum(p.numel() for p in c.parameters()),'feature':'context mean+last, target mean+last, elementwise product, abs difference','critic':'MLP 1152->64->1','train_examples':1024,'candidate_rerank':12,'clean_variable_generation':True},'training':{'last100_bce':loss,'pairwise':train_eval},'heldout':held,'rerank':rows};(out/'00_R521_RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_RERANK.txt').write_text('\n\n'.join(f"{z['prompt']}\nFIRST {z['first']['score']:.4f}: {z['first']['text']}\nBEST {z['best']['score']:.4f}: {z['best']['text']}" for z in rows),encoding='utf-8');torch.save({'state_dict':c.state_dict(),'protocol':r['protocol']},out/'R521_CRITIC.pt');print(json.dumps({'training':r['training'],'heldout':held},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
