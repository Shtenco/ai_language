#!/usr/bin/env python3
import argparse,collections,json,math,random,re,time
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER';SLOTS=6;SEED=20260810
WORD=re.compile(r'[А-Яа-яЁё][А-Яа-яЁё-]{3,}')

class Tok:
    def __init__(self,path):self.model_path=Path(path);self.sp=spm.SentencePieceProcessor(model_file=str(path));self.name='UNIGRAM4096-WARM'
    def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids):return self.sp.decode([int(x) for x in ids]).encode('utf-8')
    def vocab_bytes(self):return self.model_path.stat().st_size

class OrderedLM(base.LM):
    def __init__(self):
        super().__init__();self.plan_queries=nn.Parameter(torch.empty(SLOTS,base.D_MODEL));nn.init.normal_(self.plan_queries,std=.02)

def hidden(model,ids):
    x=model.emb(ids)
    for b in model.blocks:x=b(x)
    return model.norm(x)

def candidate_vocab(tok,raw_docs):
    freq=collections.Counter()
    for d in raw_docs:freq.update(w.lower() for w in WORD.findall(d.get('text','')))
    black={w for w,_ in freq.most_common(256)}|{x.lower() for x in base.STOP}
    ids=[];pieces={}
    for i in range(base.VOCAB):
        piece=tok.sp.id_to_piece(i);s=tok.dec([i]).decode('utf-8','ignore').strip();low=s.lower()
        if not piece.startswith('▁') or not WORD.fullmatch(s) or len(s)<4:continue
        if low in black or freq[low]<4:continue
        ids.append(i);pieces[i]=s
    if len(ids)<300:raise RuntimeError(f'candidate vocab too small {len(ids)}')
    return ids,pieces

def pfx(tok,ex,memory):return base.prefix_ids(MODE,tok,ex['ctx_text'],memory,ex['meta'])

def planner_scores(model,examples,tok,memory,cand_t):
    seq=[]
    for ex in examples:
        q=list(pfx(tok,ex,memory));q[:SLOTS]=[0]*SLOTS
        ctx=list(ex['ctx'][-base.CONTEXT_TOKENS:]);ctx=[0]*(base.CONTEXT_TOKENS-len(ctx))+ctx
        seq.append(q+ctx)
    h=hidden(model,torch.tensor(seq,dtype=torch.long))[:,-1,:]
    q=F.normalize(h[:,None,:]+model.plan_queries[None,:,:],dim=-1)
    e=F.normalize(model.emb.weight[cand_t],dim=-1)
    return torch.einsum('bsd,cd->bsc',q,e)*12.0

def gold_plan(ex,cand_set):
    out=[];seen=set()
    for t in ex['tgt']:
        t=int(t)
        if t in cand_set and t not in seen:seen.add(t);out.append(t)
        if len(out)>=SLOTS:break
    return out

def predicted(scores,cand_t):
    ans=[]
    for row in scores:
        used=set();cur=[]
        for s in range(SLOTS):
            for j in row[s].argsort(descending=True).tolist():
                tid=int(cand_t[j])
                if tid not in used:used.add(tid);cur.append(tid);break
        ans.append(cur)
    return ans

def make_prefix(tok,ex,memory,plan,apply=True):
    q=list(pfx(tok,ex,memory))
    if apply:
        z=list(plan[:SLOTS])+[0]*max(0,SLOTS-len(plan));q[:SLOTS]=z[:SLOTS]
    return q

def pack(examples,plans,tok,memory,apply_main=True):
    xs=[];ys=[];ms=[];tb=0
    for ex,plan in zip(examples,plans):
        apply=apply_main and ex['meta'].get('kind')=='main'
        q=make_prefix(tok,ex,memory,plan,apply)
        ctx=list(ex['ctx'][-base.CONTEXT_TOKENS:]);ctx=[0]*(base.CONTEXT_TOKENS-len(ctx))+ctx
        tgt=list(ex['tgt'][:base.TARGET_TOKENS]);tb+=max(1,len(tok.dec(tgt)))
        seq=q+ctx+tgt+[0]*(base.TARGET_TOKENS-len(tgt));x,y=seq[:-1],seq[1:];m=[0]*len(x);st=base.PREFIX_TOKENS+base.CONTEXT_TOKENS-1
        for j in range(st,min(st+len(tgt),len(m))):m[j]=1
        xs.append(x);ys.append(y);ms.append(m)
    return torch.tensor(xs),torch.tensor(ys),torch.tensor(ms,dtype=torch.bool),tb

def plan_ce(scores,examples,cand_pos):
    ls=[]
    for i,e in enumerate(examples):
        if e['meta'].get('kind')!='main':continue
        g=gold_plan(e,cand_pos)
        for s,t in enumerate(g):ls.append(F.cross_entropy(scores[i,s:s+1],torch.tensor([cand_pos[t]])))
    return torch.stack(ls).mean() if ls else scores.sum()*0

def recall(examples,plans,cset):
    vals=[]
    for e,p in zip(examples,plans):
        if e['meta'].get('kind')!='main':continue
        g=gold_plan(e,cset)
        if g:vals.append(len(set(g)&set(p))/len(set(g)))
    return sum(vals)/max(1,len(vals))

def sample_ex(rng,docs,tok):
    u=rng.random()
    if u<.94:return base.make_main_example(rng,docs,tok)
    if u<.97:return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)

def evaluate(model,examples,which,tok,memory,cand_t,cset,batch=16):
    nll=0.;bts=0;corr=0;toks=0;recs=[];model.eval()
    with torch.no_grad():
        for o in range(0,len(examples),batch):
            ex=examples[o:o+batch];sc=planner_scores(model,ex,tok,memory,cand_t);pr=predicted(sc,cand_t);recs.append(recall(ex,pr,cset))
            if which=='true':plans=pr
            elif which=='shuffled':plans=pr[1:]+pr[:1] if len(pr)>1 else pr
            elif which=='null':plans=[[] for _ in ex]
            elif which=='oracle':plans=[gold_plan(e,cset) for e in ex]
            else:raise ValueError(which)
            x,y,m,tb=pack(ex,plans,tok,memory,apply_main=True);z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y)
            nll+=float((ce*m).sum());bts+=tb;toks+=int(m.sum());corr+=int(((z.argmax(-1)==y)&m).sum())
    return {'bpb':nll/max(1,bts)/math.log(2),'token_top1':corr/max(1,toks),'plan_recall_at6':sum(recs)/max(1,len(recs))}

def gen_metrics(text):
    w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))]
    return {'words':len(w),'unique_word_ratio':len(set(x.lower() for x in w))/max(1,len(w)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0.0}

def generate(model,prompt,tok,memory,cand_t,pieces,seed):
    ctx=tok.enc(prompt.encode('utf-8'))[-base.CONTEXT_TOKENS:];ex={'ctx':ctx,'tgt':[],'ctx_text':prompt,'meta':{'kind':'main'}}
    with torch.no_grad():sc=planner_scores(model,[ex],tok,memory,cand_t);plan=predicted(sc,cand_t)[0]
    seq=make_prefix(tok,ex,memory,plan,True)+[0]*(base.CONTEXT_TOKENS-len(ctx))+list(ctx);out=[];model.eval()
    with torch.no_grad():
        for _ in range(base.TARGET_TOKENS):
            z=model(torch.tensor([seq]))[0,-1];t=int(z.argmax());out.append(t);seq.append(t)
    text=tok.dec(out).decode('utf-8','replace')
    return {'prompt':prompt,'plan':[pieces.get(x,str(x)) for x in plan],'continuation':text,**gen_metrics(text)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1024);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--lr',type=float,default=2e-4);a=ap.parse_args();torch.set_num_threads(a.threads)
    random.seed(SEED);torch.manual_seed(SEED);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tok=Tok(a.tokenizer_model)
    rawdocs=base.load_jsonl(a.train_docs);docs=base.tokenize_docs(tok,rawdocs);memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));cids,pieces=candidate_vocab(tok,rawdocs);cset=set(cids);cpos={x:i for i,x in enumerate(cids)};ct=torch.tensor(cids)
    model=OrderedLM();ck=torch.load(a.checkpoint,map_location='cpu');miss,unexp=model.load_state_dict(ck['state_dict'],strict=False);assert set(miss)=={'plan_queries'} and not unexp,(miss,unexp);params=base.param_count(model);assert params==2999772,params
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(SEED+516);tb_total=0;hist=[];t0=time.perf_counter();model.train()
    for step in range(a.steps):
        ex=[sample_ex(rng,docs,tok) for _ in range(a.batch)];sc=planner_scores(model,ex,tok,memory,ct);pr=predicted(sc.detach(),ct);teach=max(0.,.8*(1-step/max(1,a.steps*.50)));plans=[]
        for e,pred in zip(ex,pr):
            if e['meta'].get('kind')!='main':plans.append([]);continue
            g=gold_plan(e,cset);plans.append(g if g and rng.random()<teach else pred)
        x,y,m,tb=pack(ex,plans,tok,memory,True);opt.zero_grad(set_to_none=True);z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);lm=(ce*m).sum()/tb;pl=plan_ce(sc,ex,cpos);loss=lm+.10*pl;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();tb_total+=tb;hist.append((float(lm),float(pl),teach))
        if (step+1)%256==0:print('TRAIN',step+1,'lm',sum(q[0] for q in hist[-64:])/64,'plan',sum(q[1] for q in hist[-64:])/64,'teach',teach,flush=True)
    tests={Path(x).stem:Path(x).read_bytes() for x in a.tests};ev={}
    for i,(name,raw) in enumerate(tests.items()):
        ex=base.build_main_eval(tok,raw,7100+i*31,n=256);ev[name]={m:evaluate(model,ex,m,tok,memory,ct,cset) for m in ('true','shuffled','null','oracle')}
    logic=base.build_aux_eval(tok,'logic',88116,n=192);cyber=base.build_aux_eval(tok,'cyber',99116,n=192);ev['logic']={m:evaluate(model,logic,m,tok,memory,ct,cset) for m in ('true','shuffled','null','oracle')};ev['cyber']={m:evaluate(model,cyber,m,tok,memory,ct,cset) for m in ('true','shuffled','null','oracle')}
    prompts=base.PROMPTS+['Россия занимает большую территорию, поэтому','Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','Причина отличается от простой корреляции тем, что','После ошибки система изменила состояние и'];gens=[generate(model,p,tok,memory,ct,pieces,SEED+i) for i,p in enumerate(prompts)]
    r={'format':'nexus-r516-ordered-plan/1','protocol':{'warmstart':'R5.12 32K','params':params,'slots':SLOTS,'candidate_count':len(cids),'ordered_slot_queries':True,'teacher_anneal':'0.8 -> 0 by 50%','controls':['true','shuffled','null','oracle'],'plan_loss_weight':.10},'training':{'steps':a.steps,'batch':a.batch,'target_bytes':tb_total,'seconds':time.perf_counter()-t0,'last64_lm':sum(q[0] for q in hist[-64:])/max(1,len(hist[-64:])),'last64_plan':sum(q[1] for q in hist[-64:])/max(1,len(hist[-64:]))},'eval':ev,'generation':gens};(out/'00_R516_RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(f"{g['prompt']}\nPLAN: {', '.join(g['plan'])}\n{g['continuation']}\nMETRICS {g['unique_word_ratio']:.3f}/{g['repeated_trigram_rate']:.3f}" for g in gens),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':r['protocol']},out/'R516_ORDERED_PLAN.pt');print(json.dumps({'eval':ev,'training':r['training']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
