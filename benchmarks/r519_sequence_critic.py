#!/usr/bin/env python3
import argparse,json,math,random,re,time
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER';SEED=20260810;FEAT=base.D_MODEL*3
PROMPTS=base.PROMPTS+['Россия занимает большую территорию, поэтому','Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','Причина отличается от простой корреляции тем, что','Для достижения цели система должна','После ошибки программа изменила своё состояние и','В книге автор рассказывает о том, как','Утром город проснулся, и на улицах']

class Tok:
    def __init__(self,path):self.model_path=Path(path);self.sp=spm.SentencePieceProcessor(model_file=str(path));self.name='UNIGRAM4096-WARM'
    def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids):return self.sp.decode([int(x) for x in ids]).encode('utf-8')
    def vocab_bytes(self):return self.model_path.stat().st_size

class Critic(nn.Module):
    def __init__(self):super().__init__();self.head=nn.Linear(FEAT,1)
    def forward(self,x):return self.head(x).squeeze(-1)

def hidden(model,ids):
    x=model.emb(ids)
    for b in model.blocks:x=b(x)
    return model.norm(x)

@torch.no_grad()
def feature_batch(model,seqs,target_len=base.TARGET_TOKENS):
    x=torch.tensor(seqs,dtype=torch.long);h=hidden(model,x)[:,-target_len:,:]
    return torch.cat([h.mean(1),h[:,-1,:],h.std(1,unbiased=False)],dim=-1).cpu()

def main_seq(ex,tok,memory,target=None):
    p=list(base.prefix_ids(MODE,tok,ex['ctx_text'],memory,ex['meta']));ctx=list(ex['ctx'][-base.CONTEXT_TOKENS:]);t=list(ex['tgt'] if target is None else target)[:base.TARGET_TOKENS]
    if len(ctx)!=base.CONTEXT_TOKENS or len(t)!=base.TARGET_TOKENS:raise RuntimeError('critic cache expects exact 48+48 main examples')
    return p+ctx+t

def neg_repeat(t):
    q=list(t[:4]);return (q*((base.TARGET_TOKENS+3)//4))[:base.TARGET_TOKENS]

def neg_blocks(t,rng):
    blocks=[list(t[i:i+4]) for i in range(0,base.TARGET_TOKENS,4)];rng.shuffle(blocks);return [x for b in blocks for x in b][:base.TARGET_TOKENS]

def build_feature_dataset(model,examples,tok,memory,rng,batch=24):
    feats=[];labs=[];types=[]
    for o in range(0,len(examples),batch):
        ex=examples[o:o+batch];seq=[];meta=[]
        for i,e in enumerate(ex):
            true=list(e['tgt'][:base.TARGET_TOKENS]);wrong=list(ex[(i+1)%len(ex)]['tgt'][:base.TARGET_TOKENS]) if len(ex)>1 else true[::-1]
            variants=[('true',true,1.0),('wrong_context',wrong,0.0),('repeat4',neg_repeat(true),0.0),('block_shuffle',neg_blocks(true,rng),0.0)]
            for typ,t,l in variants:seq.append(main_seq(e,tok,memory,t));meta.append((typ,l))
        f=feature_batch(model,seq)
        feats.append(f);labs.extend([x[1] for x in meta]);types.extend([x[0] for x in meta])
    return torch.cat(feats),torch.tensor(labs,dtype=torch.float32),types

def auc(scores,labels):
    # Mann-Whitney AUC, deterministic and dependency-free.
    pos=scores[labels>0.5];neg=scores[labels<0.5]
    if not len(pos) or not len(neg):return 0.5
    return float(((pos[:,None]>neg[None,:]).float()+.5*(pos[:,None]==neg[None,:]).float()).mean())

def train_critic(critic,x,y,steps=1500):
    opt=torch.optim.AdamW(critic.parameters(),lr=2e-3,weight_decay=.01);g=torch.Generator().manual_seed(SEED+19);n=len(y);hist=[]
    for s in range(steps):
        idx=torch.randint(0,n,(128,),generator=g);z=critic(x[idx]);loss=F.binary_cross_entropy_with_logits(z,y[idx]);opt.zero_grad();loss.backward();opt.step();hist.append(float(loss))
    return sum(hist[-100:])/100

@torch.no_grad()
def sample_candidate(model,tok,memory,prompt,seed,temp=.82,topp=.92):
    ctx=tok.enc(prompt.encode('utf-8'))[-base.CONTEXT_TOKENS:];ids=list(base.prefix_ids(MODE,tok,prompt,memory,{'kind':'main'}))+ctx;g=torch.Generator().manual_seed(seed);out=[]
    for _ in range(base.TARGET_TOKENS):
        z=model(torch.tensor([ids]))[0,-1]/temp;p=F.softmax(z,dim=-1);v,ix=torch.sort(p,descending=True);cs=torch.cumsum(v,0);keep=cs<=topp;keep[0]=True;v=v[keep];ix=ix[keep];v=v/v.sum();t=int(ix[torch.multinomial(v,1,generator=g)]);ids.append(t);out.append(t)
    return out

@torch.no_grad()
def candidate_feature(model,tok,memory,prompt,target):
    ctx=tok.enc(prompt.encode('utf-8'))[-base.CONTEXT_TOKENS:];p=list(base.prefix_ids(MODE,tok,prompt,memory,{'kind':'main'}));seq=p+ctx+list(target)
    h=hidden(model,torch.tensor([seq]))[:,-len(target):,:]
    return torch.cat([h.mean(1),h[:,-1,:],h.std(1,unbiased=False)],dim=-1)[0].cpu()

def text_metrics(text):
    w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))]
    return {'words':len(w),'unique_word_ratio':len(set(x.lower() for x in w))/max(1,len(w)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0.0,'cyrillic_share':sum(bool(re.search(r'[А-Яа-яЁё]',x)) for x in w)/max(1,len(w))}

def rerank_prompts(model,critic,tok,memory,ncand=12):
    rows=[]
    for i,prompt in enumerate(PROMPTS):
        cands=[]
        for j in range(ncand):
            ids=sample_candidate(model,tok,memory,prompt,SEED+9000+i*100+j);f=candidate_feature(model,tok,memory,prompt,ids);score=float(critic(f[None,:])[0]);text=tok.dec(ids).decode('utf-8','replace');cands.append({'critic':score,'text':text,**text_metrics(text)})
        best=max(cands,key=lambda x:x['critic']);first=cands[0]
        rows.append({'prompt':prompt,'first':first,'best':best,'score_span':max(x['critic'] for x in cands)-min(x['critic'] for x in cands)})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);random.seed(SEED);torch.manual_seed(SEED);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tok=Tok(a.tokenizer_model);memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));model=base.LM();ck=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(ck['state_dict']);model.eval();assert base.param_count(model)==2998620
    for p in model.parameters():p.requires_grad_(False)
    rng=random.Random(SEED+519);train_ex=[base.make_main_example(rng,docs,tok) for _ in range(768)];x,y,types=build_feature_dataset(model,train_ex,tok,memory,rng)
    critic=Critic();loss=train_critic(critic,x,y);critic.eval()
    with torch.no_grad():train_auc=auc(critic(x),y)
    evals={}
    for i,path in enumerate(a.tests):
        raw=Path(path).read_bytes();ex=base.build_main_eval(tok,raw,15190+i*71,n=192);ex=[e for e in ex if len(e['ctx'])==48 and len(e['tgt'])==48];xx,yy,tt=build_feature_dataset(model,ex,tok,memory,random.Random(SEED+100+i));sc=critic(xx);by={}
        for typ in sorted(set(tt)):
            ix=torch.tensor([j for j,t in enumerate(tt) if t==typ]);by[typ]={'mean_score':float(sc[ix].mean()),'n':len(ix)}
        evals[Path(path).stem]={'auc':auc(sc,yy),'by_type':by}
    rows=rerank_prompts(model,critic,tok,memory,12)
    result={'format':'nexus-r519-sequence-critic/1','protocol':{'base':'R5.12 32K frozen D cortex','base_params':2998620,'critic_params':sum(p.numel() for p in critic.parameters()),'feature':'mean+last+std of target hidden states','train_examples':768,'negative_types':['wrong_context','repeat4','block_shuffle'],'candidate_rerank':12,'generation_input':'clean variable length; no left UNK padding'},'training':{'last100_bce':loss,'train_auc':train_auc},'heldout':evals,'rerank':rows}
    (out/'00_R519_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for r in rows:lines.append(f"PROMPT: {r['prompt']}\nFIRST score={r['first']['critic']:.4f}: {r['first']['text']}\nBEST score={r['best']['critic']:.4f}: {r['best']['text']}\nFIRST uniq/rep={r['first']['unique_word_ratio']:.3f}/{r['first']['repeated_trigram_rate']:.3f} BEST={r['best']['unique_word_ratio']:.3f}/{r['best']['repeated_trigram_rate']:.3f}")
    (out/'01_RERANKED_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'critic_state_dict':critic.state_dict(),'protocol':result['protocol']},out/'R519_CRITIC.pt');print(json.dumps({'training':result['training'],'heldout':evals},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
