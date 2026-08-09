#!/usr/bin/env python3
import argparse, json, math, random, re, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import r57_concept_graph_language as base

MASK_ID=4096
PAD_ID=4097
VOCAB=4098
D=192
HEADS=6
LAYERS=6
FF=570
PREFIX=32
CTX=48
TGT=48
SEQ=128
MODE='D_LOGIC_CYBER'

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.ln1=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D);self.proj=nn.Linear(D,D);self.ln2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
    def forward(self,x):
        b,l,d=x.shape;h=self.ln1(x);qkv=self.qkv(h).view(b,l,3,HEADS,d//HEADS).permute(2,0,3,1,4);q,k,v=qkv
        dh=q.shape[-1];pos=torch.arange(l,device=x.device,dtype=x.dtype);inv=1.0/(10000**(torch.arange(0,dh,2,device=x.device,dtype=x.dtype)/dh));ang=torch.outer(pos,inv);co=ang.cos()[None,None];si=ang.sin()[None,None]
        def rot(z):
            e,o=z[...,0::2],z[...,1::2];y=torch.empty_like(z);y[...,0::2]=e*co-o*si;y[...,1::2]=e*si+o*co;return y
        q,k=rot(q),rot(k);a=F.scaled_dot_product_attention(q,k,v,is_causal=False);x=x+self.proj(a.transpose(1,2).contiguous().view(b,l,d));x=x+self.fc2(F.gelu(self.fc1(self.ln2(x))));return x

class MaskedLM(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(VOCAB,D);self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,VOCAB,bias=False);self.head.weight=self.emb.weight
    def forward(self,ids):
        x=self.emb(ids)
        for b in self.blocks:x=b(x)
        return self.head(self.norm(x))

def pcount(m):return sum(p.numel() for p in m.parameters())

def prefix(tok,memory,ctx_text,meta):
    p=base.prefix_ids(MODE,tok,ctx_text,memory,meta)
    # old helper uses token 0 as neutral pad; reserve explicit PAD in this model.
    return [PAD_ID if x==0 else x for x in p]

def pack(examples,tok,memory,rng,train=True):
    xs=[];ys=[];lossm=[];validm=[];bytes_total=0
    for ex in examples:
        p=prefix(tok,memory,ex['ctx_text'],ex['meta'])
        c=list(ex['ctx'][-CTX:]);c=[PAD_ID]*(CTX-len(c))+c
        t=list(ex['tgt'][:TGT]);valid=[1]*len(t)+[0]*(TGT-len(t));target=t+[PAD_ID]*(TGT-len(t))
        if train:
            # half the time the whole future is unknown; otherwise partial denoising teaches iterative refinement.
            ratio=1.0 if rng.random()<0.5 else rng.uniform(0.45,0.85)
            inp=target.copy();chosen=[]
            for j in range(TGT):
                if valid[j] and rng.random()<ratio:chosen.append(j)
            if not chosen and t:chosen=[rng.randrange(len(t))]
            for j in chosen:inp[j]=MASK_ID
            lm=[1 if j in chosen else 0 for j in range(TGT)]
        else:
            inp=[MASK_ID if valid[j] else PAD_ID for j in range(TGT)];lm=valid.copy()
        xs.append(p+c+inp);ys.append([PAD_ID]*(PREFIX+CTX)+target);lossm.append([0]*(PREFIX+CTX)+lm);validm.append([0]*(PREFIX+CTX)+valid);bytes_total+=max(1,len(tok.dec(t)))
    return torch.tensor(xs),torch.tensor(ys),torch.tensor(lossm,dtype=torch.bool),torch.tensor(validm,dtype=torch.bool),bytes_total

def sample_ex(rng,docs,tok):
    u=rng.random()
    if u<.92:return base.make_main_example(rng,docs,tok)
    if u<.96:return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)

def lr_factor(step,total):
    warm=max(32,total//32)
    if step<warm:return max(.05,(step+1)/warm)
    q=(step-warm)/max(1,total-warm);return .5*(1+math.cos(math.pi*min(1,q)))

def train(seed,tok,docs,memory,steps,batch,lr=1e-3):
    random.seed(seed);torch.manual_seed(seed);m=MaskedLM();pc=pcount(m);assert pc==2_999_004,pc;opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(seed+99117);hist=[];masked=0;t0=time.perf_counter();m.train()
    for step in range(steps):
        ex=[sample_ex(rng,docs,tok) for _ in range(batch)];x,y,lm,vm,_=pack(ex,tok,memory,rng,True)
        for pg in opt.param_groups:pg['lr']=lr*lr_factor(step,steps)
        opt.zero_grad(set_to_none=True);z=m(x);ce=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y);loss=(ce*lm).sum()/lm.sum().clamp_min(1);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();hist.append(float(loss));masked+=int(lm.sum())
        if (step+1)%256==0:print('TRAIN',step+1,'masked_ce',sum(hist[-64:])/min(64,len(hist)),'masked_tokens',masked,flush=True)
    return m,{'params':pc,'steps':steps,'batch':batch,'train_s':time.perf_counter()-t0,'masked_tokens':masked,'last64_masked_ce':sum(hist[-64:])/min(64,len(hist))}

@torch.no_grad()
def fullmask_eval(m,examples,tok,memory,batch=8):
    m.eval();nll=0.;bts=0;toks=0;correct=0;rng=random.Random(1)
    for i in range(0,len(examples),batch):
        ex=examples[i:i+batch];x,y,lm,vm,tb=pack(ex,tok,memory,rng,False);z=m(x);ce=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y);nll+=float((ce*lm).sum());bts+=tb;toks+=int(lm.sum());correct+=int(((z.argmax(-1)==y)&lm).sum())
    return {'fullmask_bpb':nll/max(1,bts)/math.log(2),'masked_token_ce':nll/max(1,toks),'masked_top1':correct/max(1,toks),'target_bytes':bts,'target_tokens':toks}

@torch.no_grad()
def iterative(m,tok,memory,prompt,meta=None,seed=1,refine=3):
    random.seed(seed);torch.manual_seed(seed);meta=meta or {'kind':'main'};ctx=tok.enc(prompt.encode('utf-8'))[-CTX:];ctx=[PAD_ID]*(CTX-len(ctx))+ctx;p=prefix(tok,memory,prompt,meta);ids=torch.tensor([p+ctx+[MASK_ID]*TGT]);conf=torch.zeros(TGT);m.eval()
    # six global fill rounds: every new decision sees both left and right already-filled anchors.
    for _ in range(6):
        z=m(ids)[0,PREFIX+CTX:];probs=F.softmax(z,-1);mx,pred=probs.max(-1);mask=(ids[0,PREFIX+CTX:]==MASK_ID);idx=torch.where(mask)[0]
        if len(idx)==0:break
        k=min(8,len(idx));sel=idx[torch.topk(mx[idx],k).indices];ids[0,PREFIX+CTX+sel]=pred[sel];conf[sel]=mx[sel]
    # confidence-driven correction loop.
    for _ in range(refine):
        z=m(ids)[0,PREFIX+CTX:];pr=F.softmax(z,-1);cur=ids[0,PREFIX+CTX:];curp=pr.gather(-1,cur[:,None]).squeeze(-1);bad=torch.topk(-curp,8).indices;ids[0,PREFIX+CTX+bad]=MASK_ID;z=m(ids)[0,PREFIX+CTX:];pr=F.softmax(z,-1);ids[0,PREFIX+CTX+bad]=pr[bad].argmax(-1)
    out=[int(x) for x in ids[0,PREFIX+CTX:] if int(x)<4096]
    return tok.dec(out).decode('utf-8','replace')

def text_metrics(s):return base.text_metrics(s)

def generation_suite(m,tok,memory,seed):
    rows=[]
    for i,p in enumerate(base.PROMPTS):
        c=iterative(m,tok,memory,p,seed=seed+i*31);rows.append({'prompt':p,'continuation':c,**text_metrics(c)})
    return rows

def exact_acc(m,tok,memory,kind,seed,n=32):
    rng=random.Random(seed);rows=[]
    for i in range(n):
        if kind=='logic':ctx,target,meta=base.make_logic_example(rng,train=False)
        else:ctx,target,meta=base.make_cyber_example(rng)
        g=iterative(m,tok,memory,ctx,meta,seed+i);lo=g.lower()
        if kind=='logic':ok=(('да' in lo[:50]) if meta['answer_yes'] else ('нет' in lo[:70] or 'недостат' in lo)) and meta['claim'].lower() in lo
        else:
            aw='увелич' if meta['error']>meta['tolerance'] else ('уменьш' if meta['error']<-meta['tolerance'] else 'удерж');ok=base.fmt_num(meta['error']) in lo and aw in lo
        rows.append({'context':ctx,'target':target,'generated':g,'ok':bool(ok)})
    return {'accuracy':sum(r['ok'] for r in rows)/len(rows),'rows':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-text',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=4096);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--seed',type=int,default=20260810);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);raw=Path(a.train_text).read_bytes();tok=base.tt.train_sp(raw,base.VOCAB,'unigram',out);probe='Итеративное восстановление русского текста №123.\n'.encode();assert tok.dec(tok.enc(probe))==probe;docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};m,tr=train(a.seed,tok,docs,memory,a.steps,a.batch);ev={};
    for i,(name,b) in enumerate(tests.items()):ev[name]=fullmask_eval(m,base.build_main_eval(tok,b,7000+i*31,n=128),tok,memory)
    ev['logic']=fullmask_eval(m,base.build_aux_eval(tok,'logic',88001,n=96),tok,memory);ev['cyber']=fullmask_eval(m,base.build_aux_eval(tok,'cyber',99001,n=96),tok,memory);gens=generation_suite(m,tok,memory,a.seed+501);la=exact_acc(m,tok,memory,'logic',123451,32);ca=exact_acc(m,tok,memory,'cyber',223451,32);result={'format':'nexus-r513-masked-iterative/1','protocol':{'params':pcount(m),'tokenizer':'lossless Unigram4096 + MASK/PAD only inside neural model','architecture':'6-layer bidirectional RoPE masked graph-state realizer','steps':a.steps,'batch':a.batch,'seed':a.seed,'training':'50% full-target masking; otherwise 45-85% target masking','inference':'six confidence-ranked fill rounds + three low-confidence remask/refine rounds'},'training':tr,'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca};(out/'00_R513_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"{g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={la['accuracy']:.6f} CYBER={ca['accuracy']:.6f}");(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':m.state_dict(),'protocol':result['protocol']},out/'R513_MASKED_ITERATIVE.pt');print(json.dumps({'training':tr,'eval':ev,'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
