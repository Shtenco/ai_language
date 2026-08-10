#!/usr/bin/env python3
import argparse, json, math, random, re, time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER'
PLAN_SLOTS=8
SEED=20260810
CYR_WORD=re.compile(r'^\s*[А-Яа-яЁё][А-Яа-яЁё-]{2,}$')

class Tok:
    def __init__(self, model_path):
        self.model_path=Path(model_path)
        self.sp=spm.SentencePieceProcessor(model_file=str(model_path))
        self.name='UNIGRAM4096-WARM'
    def enc(self,b): return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids): return self.sp.decode([int(x) for x in ids]).encode('utf-8')
    def vocab_bytes(self): return self.model_path.stat().st_size

def hidden(model, ids):
    x=model.emb(ids)
    for block in model.blocks: x=block(x)
    return model.norm(x)

def content_vocab(tok):
    ids=[]; pieces={}
    for i in range(base.VOCAB):
        try:s=tok.dec([i]).decode('utf-8','ignore')
        except Exception:continue
        if CYR_WORD.fullmatch(s or '') and len(s.strip())>=3:
            ids.append(i); pieces[i]=s.strip()
    if len(ids)<200: raise RuntimeError(f'content vocabulary unexpectedly small: {len(ids)}')
    return ids,pieces

def d_prefix(tok, ex, memory):
    return base.prefix_ids(MODE,tok,ex['ctx_text'],memory,ex['meta'])

def planner_batch(model, examples, tok, memory, cand_ids_t):
    seq=[]
    for ex in examples:
        pfx=d_prefix(tok,ex,memory)
        pfx=list(pfx); pfx[:PLAN_SLOTS]=[0]*PLAN_SLOTS
        ctx=list(ex['ctx'][-base.CONTEXT_TOKENS:])
        ctx=[0]*(base.CONTEXT_TOKENS-len(ctx))+ctx
        seq.append(pfx+ctx)
    x=torch.tensor(seq,dtype=torch.long)
    h=hidden(model,x)[:,-1,:]
    h=F.normalize(h,dim=-1)
    e=F.normalize(model.emb.weight[cand_ids_t],dim=-1)
    return (h@e.T)*12.0

def gold_anchors(ex, cand_set):
    out=[]; seen=set()
    for t in ex['tgt']:
        t=int(t)
        if t in cand_set and t not in seen:
            seen.add(t); out.append(t)
        if len(out)>=PLAN_SLOTS: break
    return out

def predicted_plans(scores, cand_ids_t):
    k=min(PLAN_SLOTS,scores.shape[1])
    idx=scores.topk(k,dim=-1).indices
    ids=cand_ids_t[idx]
    return [[int(x) for x in row.tolist()] for row in ids]

def make_plan_prefix(tok, ex, memory, anchors):
    pfx=list(d_prefix(tok,ex,memory))
    a=list(anchors[:PLAN_SLOTS]); a=a+[0]*(PLAN_SLOTS-len(a))
    pfx[:PLAN_SLOTS]=a
    return pfx

def pack_with_plans(examples, plans, tok, memory):
    xs=[];ys=[];masks=[];tb=0;tt=0
    for ex,plan in zip(examples,plans):
        pfx=make_plan_prefix(tok,ex,memory,plan)
        ctx=list(ex['ctx'][-base.CONTEXT_TOKENS:])
        ctxslot=[0]*(base.CONTEXT_TOKENS-len(ctx))+ctx
        tgt=list(ex['tgt'][:base.TARGET_TOKENS]); tb+=max(1,len(tok.dec(tgt))); tt+=len(tgt)
        seq=pfx+ctxslot+tgt+[0]*(base.TARGET_TOKENS-len(tgt))
        x,y=seq[:-1],seq[1:]
        mask=[0]*len(x); start=base.PREFIX_TOKENS+base.CONTEXT_TOKENS-1
        for j in range(start,min(start+len(tgt),len(mask))): mask[j]=1
        xs.append(x);ys.append(y);masks.append(mask)
    return torch.tensor(xs),torch.tensor(ys),torch.tensor(masks,dtype=torch.bool),tb,tt

def sample_ex(rng,docs,tok):
    u=rng.random()
    if u<0.94:return base.make_main_example(rng,docs,tok)
    if u<0.97:return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)

def plan_loss(scores, examples, cand_pos, rng):
    rows=[]; labels=[]
    for i,ex in enumerate(examples):
        if ex['meta'].get('kind')!='main': continue
        g=[cand_pos[x] for x in gold_anchors(ex,cand_pos) if x in cand_pos]
        if not g: continue
        rows.append(i); labels.append(rng.choice(g))
    if not rows:return scores.sum()*0.0
    return F.cross_entropy(scores[torch.tensor(rows)],torch.tensor(labels,dtype=torch.long))

def plan_recall(examples, plans, cand_set):
    vals=[]
    for ex,p in zip(examples,plans):
        g=set(gold_anchors(ex,cand_set))
        if g: vals.append(len(g.intersection(p))/len(g))
    return sum(vals)/max(1,len(vals))

def evaluate(model, examples, mode, tok, memory, cand_ids_t, cand_set, batch=16):
    nll=0.0;bts=0;toks=0;correct=0;rec=[]
    model.eval()
    with torch.no_grad():
        for off in range(0,len(examples),batch):
            ex=examples[off:off+batch]
            scores=planner_batch(model,ex,tok,memory,cand_ids_t)
            truep=predicted_plans(scores,cand_ids_t)
            rec.append(plan_recall(ex,truep,cand_set))
            if mode=='true':plans=truep
            elif mode=='null':plans=[[0]*PLAN_SLOTS for _ in ex]
            elif mode=='shuffled':plans=truep[1:]+truep[:1] if len(truep)>1 else truep
            else:raise ValueError(mode)
            x,y,m,tb,tt=pack_with_plans(ex,plans,tok,memory)
            z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y)
            nll+=float((ce*m).sum());bts+=tb;toks+=int(m.sum());correct+=int(((z.argmax(-1)==y)&m).sum())
    return {'bpb':nll/max(1,bts)/math.log(2),'nats_per_byte':nll/max(1,bts),'token_top1':correct/max(1,toks),'target_bytes':bts,'target_tokens':toks,'plan_recall_at8':sum(rec)/max(1,len(rec))}

def metrics(text):
    words=re.findall(r'[А-Яа-яЁёA-Za-z]+',text)
    cyr=sum(bool(re.search(r'[А-Яа-яЁё]',w)) for w in words)
    tri=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    rep=1-len(set(tri))/max(1,len(tri)) if tri else 0.0
    return {'words':len(words),'cyrillic_word_share':cyr/max(1,len(words)),'unique_word_ratio':len(set(w.lower() for w in words))/max(1,len(words)),'repeated_trigram_rate':rep}

def predict_prompt_plan(model,prompt,tok,memory,cand_ids_t):
    ctx=tok.enc(prompt.encode('utf-8'))[-base.CONTEXT_TOKENS:]
    ex={'ctx':ctx,'tgt':[],'ctx_text':prompt,'meta':{'kind':'main'}}
    with torch.no_grad():scores=planner_batch(model,[ex],tok,memory,cand_ids_t)
    return predicted_plans(scores,cand_ids_t)[0],ex

def generate(model,prompt,tok,memory,cand_ids_t,decode='greedy',seed=0):
    plan,ex=predict_prompt_plan(model,prompt,tok,memory,cand_ids_t)
    pfx=make_plan_prefix(tok,ex,memory,plan)
    ctx=list(ex['ctx']); ctxslot=[0]*(base.CONTEXT_TOKENS-len(ctx))+ctx
    seq=pfx+ctxslot; gen=[]; rng=torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for _ in range(base.TARGET_TOKENS):
            z=model(torch.tensor([seq],dtype=torch.long))[0,-1]/0.85
            if decode=='greedy':nxt=int(z.argmax())
            else:
                p=F.softmax(z,dim=-1); vals,idx=torch.sort(p,descending=True); cs=torch.cumsum(vals,0); keep=cs<=0.92; keep[0]=True
                vals=vals[keep];idx=idx[keep];vals=vals/vals.sum();nxt=int(idx[torch.multinomial(vals,1,generator=rng)])
            gen.append(nxt);seq.append(nxt)
    text=tok.dec(gen).decode('utf-8','replace')
    return {'prompt':prompt,'decode':decode,'plan_ids':plan,'continuation':text,**metrics(text)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1024);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--lr',type=float,default=2e-4);a=ap.parse_args()
    torch.set_num_threads(a.threads); random.seed(SEED); torch.manual_seed(SEED)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tok=Tok(a.tokenizer_model); probe='Иерархический план NEXUS №514.\n'.encode(); assert tok.dec(tok.enc(probe))==probe
    docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));assert len(docs)>=50 and len(memory.chunks)>=100
    cand_ids,pieces=content_vocab(tok);cand_set=set(cand_ids);cand_pos={x:i for i,x in enumerate(cand_ids)};cand_ids_t=torch.tensor(cand_ids,dtype=torch.long)
    model=base.LM(); ck=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(ck['state_dict']);assert base.param_count(model)==2_998_620
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01,betas=(.9,.95));rng=random.Random(SEED+514)
    hist=[];target_bytes=0;t0=time.perf_counter();model.train()
    for step in range(a.steps):
        ex=[sample_ex(rng,docs,tok) for _ in range(a.batch)]
        scores=planner_batch(model,ex,tok,memory,cand_ids_t);pred=predicted_plans(scores.detach(),cand_ids_t)
        teacher_p=max(0.0,0.70*(1-step/max(1,a.steps*0.55)))
        plans=[]
        for e,p in zip(ex,pred):
            if e['meta'].get('kind')!='main':plans.append([0]*PLAN_SLOTS);continue
            g=gold_anchors(e,cand_set)
            plans.append(g if g and rng.random()<teacher_p else p)
        x,y,m,tb,_=pack_with_plans(ex,plans,tok,memory)
        for pg in opt.param_groups:pg['lr']=a.lr*base.lr_factor(step,a.steps)
        opt.zero_grad(set_to_none=True);z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);lm=(ce*m).sum()/tb;pl=plan_loss(scores,ex,cand_pos,rng);loss=lm+0.08*pl;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();target_bytes+=tb
        hist.append((float(lm.detach()),float(pl.detach()),teacher_p))
        if (step+1)%256==0:print('TRAIN',step+1,'lm_npb',sum(x[0] for x in hist[-64:])/64,'plan_ce',sum(x[1] for x in hist[-64:])/64,'teacher_p',teacher_p,'MB',target_bytes/1048576,flush=True)
    tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};evals={}
    for i,(name,raw) in enumerate(tests.items()):
        ex=base.build_main_eval(tok,raw,7000+i*31,n=256);evals[name]={m:evaluate(model,ex,m,tok,memory,cand_ids_t,cand_set) for m in ('true','shuffled','null')}
    logic=base.build_aux_eval(tok,'logic',88001,n=192);cyber=base.build_aux_eval(tok,'cyber',99001,n=192)
    evals['logic']={m:evaluate(model,logic,m,tok,memory,cand_ids_t,cand_set) for m in ('true','shuffled','null')};evals['cyber']={m:evaluate(model,cyber,m,tok,memory,cand_ids_t,cand_set) for m in ('true','shuffled','null')}
    prompts=base.PROMPTS+['Россия занимает большую территорию, поэтому','Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','Причина отличается от простой корреляции тем, что','Для достижения цели система должна','После ошибки программа изменила своё состояние и']
    gens=[]
    for i,p in enumerate(prompts):
        for dec in ('greedy','sample'):gens.append(generate(model,p,tok,memory,cand_ids_t,dec,SEED+900+i))
    for g in gens:g['plan_pieces']=[pieces.get(i,tok.dec([i]).decode('utf-8','replace')) for i in g['plan_ids']]
    result={'format':'nexus-r514-hierarchical-planner/1','protocol':{'warmstart':'R5.12 32768-step D_LOGIC_CYBER','params':base.param_count(model),'steps':a.steps,'batch':a.batch,'plan_slots':PLAN_SLOTS,'content_candidates':len(cand_ids),'planner':'two-pass shared-cortex future lexical-semantic anchor prediction','teacher_plan_anneal':'0.70 -> 0 by 55% of finetune','plan_loss_weight':0.08,'controls':['true','shuffled','null']},'training':{'seconds':time.perf_counter()-t0,'target_bytes':target_bytes,'last64_lm_npb':sum(x[0] for x in hist[-64:])/max(1,len(hist[-64:])),'last64_plan_ce':sum(x[1] for x in hist[-64:])/max(1,len(hist[-64:]))},'eval':evals,'generation':gens}
    (out/'00_R514_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\nPLAN: {', '.join(g['plan_pieces'])}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_word_share','unique_word_ratio','repeated_trigram_rate')},ensure_ascii=False)}")
    (out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R514_HIERARCHICAL_PLANNER.pt')
    print(json.dumps({'candidate_vocab':len(cand_ids),'eval':evals,'train':result['training']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
