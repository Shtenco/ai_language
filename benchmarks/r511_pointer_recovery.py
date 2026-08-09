#!/usr/bin/env python3
import argparse,json,math,random,time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import r57_concept_graph_language as base
MODE='D_LOGIC_CYBER'
SRC=base.PREFIX_TOKENS+base.CONTEXT_TOKENS

def set_seed(s):random.seed(s);torch.manual_seed(s)
def lr_factor(step,total):
    warm=max(32,total//32)
    if step<warm:return max(.05,(step+1)/warm)
    q=(step-warm)/max(1,total-warm);return .5*(1+math.cos(math.pi*min(1,q)))

class PointerLM(base.LM):
    def __init__(self):
        super().__init__();self.gen_gate=nn.Linear(base.D_MODEL,1)
    def hidden_logits(self,ids):
        x=self.emb(ids)
        for b in self.blocks:x=b(x)
        h=self.norm(x);return h,self.head(h)
    def mixture(self,ids):
        h,logits=self.hidden_logits(ids);pv=F.softmax(logits,dim=-1);source=ids[:,:min(SRC,ids.shape[1])];se=self.emb(source);score=torch.einsum('bld,bsd->bls',h,se)/math.sqrt(base.D_MODEL);valid=(source!=0)[:,None,:];score=score.masked_fill(~valid,-1e4);att=F.softmax(score,dim=-1);pc=torch.zeros_like(pv);idx=source[:,None,:].expand(-1,ids.shape[1],-1);pc.scatter_add_(2,idx,att);g=torch.sigmoid(self.gen_gate(h));mix=(g*pv+(1-g)*pc).clamp_min(1e-9);return h,logits,mix,g
    def forward(self,ids):return self.hidden_logits(ids)[1]

def nll_bytes(mix,y,mask,tb):
    p=mix.gather(-1,y[...,None]).squeeze(-1);return (-torch.log(p)*mask).sum()/tb

def corrupt_history(x,mix,mask,p):
    bad=x.clone();pred=mix.argmax(-1);B,L=x.shape;start=base.PREFIX_TOKENS+base.CONTEXT_TOKENS
    for j in range(start,L):
        elig=mask[:,j-1]&(x[:,j]!=0);choose=(torch.rand(B)<p)&elig;bad[choose,j]=pred[choose,j-1]
    return bad

def unlikelihood(mix,x,y,mask):
    prev=mix.gather(-1,x[...,None]).squeeze(-1).clamp(max=1-1e-6);valid=mask&(y!=x);return (-torch.log1p(-prev[valid])).mean() if bool(valid.any()) else mix.sum()*0

def train(seed,tok,docs,memory,steps,batch,lr=1e-3):
    set_seed(seed);m=PointerLM();pc=base.param_count(m);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(seed+77881);ch=[];rh=[];uh=[];gh=[];seen=0;t0=time.perf_counter();m.train()
    for step in range(steps):
        ex=[base.sample_example(rng,docs,tok) for _ in range(batch)];x,y,mask,tb,_=base.pack(ex,MODE,tok,memory)
        for pg in opt.param_groups:pg['lr']=lr*lr_factor(step,steps)
        opt.zero_grad(set_to_none=True);_,_,mix,g=m.mixture(x);clean=nll_bytes(mix,y,mask,tb);p=.04+.24*(step/max(1,steps-1));bad=corrupt_history(x,mix.detach(),mask,p);_,_,mixb,gb=m.mixture(bad);recover=nll_bytes(mixb,y,mask,tb);ul=unlikelihood(mixb,bad,y,mask);loss=.42*clean+.58*recover+.035*ul;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();ch.append(float(clean));rh.append(float(recover));uh.append(float(ul));gh.append(float(g[mask].mean()) if bool(mask.any()) else 1.0);seen+=tb
        if (step+1)%256==0:print('TRAIN',step+1,'clean',sum(ch[-64:])/64,'recover',sum(rh[-64:])/64,'ul',sum(uh[-64:])/64,'gen_gate',sum(gh[-64:])/64,'p',p,'MB',seen/1048576,flush=True)
    return m,{'mode':'R5.11_POINTER_RECOVERY','seed':seed,'params':pc,'steps':steps,'batch':batch,'train_s':time.perf_counter()-t0,'target_bytes':seen,'last64_clean_npb':sum(ch[-64:])/min(64,len(ch)),'last64_recovery_npb':sum(rh[-64:])/min(64,len(rh)),'last64_ul':sum(uh[-64:])/min(64,len(uh)),'last64_gen_gate':sum(gh[-64:])/min(64,len(gh))}

@torch.no_grad()
def evaluate(m,examples,tok,memory,batch=8):
    m.eval();N=0.;B=0;T=0;C=0;G=[]
    for i in range(0,len(examples),batch):
        raw=examples[i:i+batch];real=len(raw);ex=raw if real==batch else raw+[raw[-1]]*(batch-real);x,y,mask,_,_=base.pack(ex,MODE,tok,memory);x,y,mask=x[:real],y[:real],mask[:real];tb=sum(max(1,len(tok.dec(e['tgt'][:base.TARGET_TOKENS]))) for e in ex[:real]);_,_,mix,g=m.mixture(x);p=mix.gather(-1,y[...,None]).squeeze(-1);N+=float((-torch.log(p)*mask).sum());B+=tb;T+=int(mask.sum());C+=int(((mix.argmax(-1)==y)&mask).sum());G.append(float(g[mask].mean()))
    return {'bpb':N/B/math.log(2),'nats_per_byte':N/B,'token_top1':C/max(1,T),'mean_generate_gate':sum(G)/len(G)}

@torch.no_grad()
def generate(m,tok,memory,prompt,seed,meta=None,sample=True,max_new=64):
    set_seed(seed);ids=base.generation_input(MODE,tok,memory,prompt,meta);generated=[];m.eval()
    for _ in range(max_new):
        x=torch.tensor([ids[-160:]],dtype=torch.long);_,_,mix,g=m.mixture(x);p=mix[0,-1]
        if sample:
            v,ix=torch.topk(torch.log(p)/.78,min(40,len(p)));nxt=int(ix[torch.multinomial(F.softmax(v,-1),1)])
        else:nxt=int(p.argmax())
        ids.append(nxt);generated.append(nxt)
    return tok.dec(generated).decode('utf-8','replace')

def generation_suite(m,tok,memory,seed):
    rows=[]
    for i,p in enumerate(base.PROMPTS):
        for sample in (False,True):
            c=generate(m,tok,memory,p,seed+i*101+int(sample),sample=sample);rows.append({'prompt':p,'decode':'sample' if sample else 'greedy','continuation':c,**base.text_metrics(c)})
    return rows

def exact_acc(m,tok,memory,kind,seed,n=32):
    rng=random.Random(seed);rows=[]
    for i in range(n):
        if kind=='logic':ctx,target,meta=base.make_logic_example(rng,train=False)
        else:ctx,target,meta=base.make_cyber_example(rng)
        g=generate(m,tok,memory,ctx,seed+i*17,meta=meta,sample=False,max_new=32);lo=g.lower()
        if kind=='logic':ok=(('да' in lo[:40]) if meta['answer_yes'] else ('нет' in lo[:50] or 'недостат' in lo)) and meta['claim'].lower() in lo
        else:
            aw='увелич' if meta['error']>meta['tolerance'] else ('уменьш' if meta['error']<-meta['tolerance'] else 'удерж');ok=base.fmt_num(meta['error']) in lo and aw in lo
        rows.append({'context':ctx,'target':target,'generated':g,'ok':bool(ok)})
    return {'accuracy':sum(x['ok'] for x in rows)/len(rows),'rows':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-text',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1024);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--seed',type=int,default=20260810);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);raw=Path(a.train_text).read_bytes();tok=base.tt.train_sp(raw,base.VOCAB,'unigram',out);probe='Графовый pointer-контур русского текста №123.\n'.encode();assert tok.dec(tok.enc(probe))==probe;docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};m,tr=train(a.seed,tok,docs,memory,a.steps,a.batch);me={name:base.build_main_eval(tok,b,7000+i*31,n=128) for i,(name,b) in enumerate(tests.items())};logic=base.build_aux_eval(tok,'logic',88001,n=96);cyber=base.build_aux_eval(tok,'cyber',99001,n=96);ev={name:evaluate(m,x,tok,memory) for name,x in me.items()};ev['logic']=evaluate(m,logic,tok,memory);ev['cyber']=evaluate(m,cyber,tok,memory);gens=generation_suite(m,tok,memory,a.seed+501);la=exact_acc(m,tok,memory,'logic',123451,32);ca=exact_acc(m,tok,memory,'cyber',223451,32);result={'format':'nexus-r511-pointer-recovery/1','protocol':{'params':base.param_count(m),'base':'R5.7 D + R5.10 recovery','copy_source_tokens':SRC,'pointer':'dot(final_hidden, source_embedding) + learned scalar generation gate','steps':a.steps,'batch':a.batch,'seed':a.seed},'training':tr,'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca};(out/'00_R511_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\n{json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','sentence_endings_per_100_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={la['accuracy']:.6f} CYBER={ca['accuracy']:.6f}");(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':m.state_dict(),'protocol':result['protocol']},out/'R511_POINTER_RECOVERY.pt');print(json.dumps({'training':tr,'eval':{k:v['bpb'] for k,v in ev.items()},'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
