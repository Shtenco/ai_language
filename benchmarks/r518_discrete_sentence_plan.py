#!/usr/bin/env python3
import argparse
import collections
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import r57_concept_graph_language as base57
import r515_encoder_decoder as base515

VOCAB=base515.VOCAB
D=base515.D_MODEL
PLAN_K=8
BATCH=8
SEED=20260810
CYR=re.compile(r'[А-Яа-яЁё]')
WORD=re.compile(r'[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z-]{1,}')

PROMPTS=[
 'Наука развивается потому, что',
 'Причинная связь отличается от корреляции тем, что',
 'Если эксперимент не подтвердил гипотезу, исследователь должен',
 'Память помогает рассуждению, потому что',
 'Когда программа обнаружила противоречие, она',
 'Чтобы проверить утверждение, необходимо',
 'Современная компьютерная система обрабатывает данные и',
 'После ошибки система изменила своё состояние и',
 'Хорошее объяснение начинается с того, что',
 'Исследователь сравнил результаты и обнаружил, что',
 'Если два факта противоречат друг другу, нужно',
 'В городе открыли новый научный центр, где',
]

class NEXUS518(base515.NEXUS515):
    def __init__(self):
        super().__init__()
        self.content_head=nn.Linear(D,VOCAB)
    def plan_logits(self, mem):
        pooled=mem[:,base515.GRAPH_TOKENS:,:].mean(dim=1)
        return self.content_head(pooled)
    def decode_with_plan(self, dec_ids, mem, plan_ids):
        # Plan is discrete and stable for the whole sentence; decoder cross-attends to it as extra semantic memory.
        p=self.emb(plan_ids)
        full=torch.cat([mem,p],dim=1)
        return self.decode(dec_ids,full)


def allowed_vocab(tok):
    mask=torch.zeros(VOCAB,dtype=torch.bool)
    for i in range(VOCAB):
        try:s=tok.dec([i]).decode('utf-8','replace').strip()
        except Exception:continue
        letters=[c for c in s if c.isalpha()]
        if len(letters)>=2 and sum(1 for c in letters if CYR.match(c))/len(letters)>=0.65:
            mask[i]=True
    # Token 0 is used as internal start/pad in this lab.
    mask[0]=False
    return mask


def token_counts(docs):
    c=collections.Counter()
    for d in docs:
        c.update(d['ids'])
    return c


def target_plan(tgt, allow, counts):
    seen=[]
    for t in tgt:
        t=int(t)
        if allow[t] and t not in seen:seen.append(t)
    # Prefer informative/rarer pieces while preserving deterministic tie-breaking by first appearance.
    pos={t:i for i,t in enumerate(seen)}
    seen=sorted(seen,key=lambda t:(counts.get(t,0),pos[t]))[:PLAN_K]
    if not seen:seen=[1 if allow[1] else int(torch.where(allow)[0][0])]
    return seen+[0]*(PLAN_K-len(seen))


def predicted_plan(logits,allow):
    z=logits.masked_fill(~allow.to(logits.device),float('-inf'))
    return torch.topk(z,PLAN_K,dim=-1).indices


def multi_hot(plans):
    y=torch.zeros(len(plans),VOCAB)
    for i,p in enumerate(plans):
        for t in p:
            if t!=0:y[i,t]=1.0
    return y


def sample_ex(rng,docs,tok):
    u=rng.random()
    if u<.94:return base57.make_main_example(rng,docs,tok)
    if u<.97:return base57.make_aux_example('logic',rng,tok,train=True)
    return base57.make_aux_example('cyber',rng,tok,train=True)


def batch_base(exs,tok,memory):
    return base515.batchify(exs,tok,memory,'true')


def train(model,tok,docs,memory,steps,allow,counts):
    random.seed(SEED);torch.manual_seed(SEED);rng=random.Random(SEED+518)
    opt=torch.optim.AdamW(model.parameters(),lr=2.5e-4,betas=(.9,.95),weight_decay=.01)
    hist=[];ph=[];target_bytes=0;t0=time.perf_counter();model.train()
    for step in range(steps):
        exs=[sample_ex(rng,docs,tok) for _ in range(BATCH)]
        enc,dec,tgt,mask,rel,bcs=batch_base(exs,tok,memory)
        mem=model.encode(enc)
        plog=model.plan_logits(mem)
        teacher=[target_plan(ex['tgt'],allow,counts) for ex in exs]
        py=multi_hot(teacher)
        # Weighted multi-label content loss; positive tokens are sparse.
        content_loss=F.binary_cross_entropy_with_logits(plog,py,pos_weight=torch.full((VOCAB,),40.0))
        pred=predicted_plan(plog.detach(),allow)
        # Scheduled teacher-plan exposure. Most batches must learn to survive their own predicted plan.
        teacher_prob=max(.20,.65-.45*(step/max(1,steps-1)))
        use_teacher=torch.rand(BATCH)<teacher_prob
        plan_ids=pred.clone()
        for i,use in enumerate(use_teacher.tolist()):
            if use:plan_ids[i]=torch.tensor(teacher[i],dtype=torch.long)
        logits,h=model.decode_with_plan(dec,mem,plan_ids)
        ce=F.cross_entropy(logits.reshape(-1,VOCAB),tgt.reshape(-1),reduction='none').view(BATCH,base515.TARGET_TOKENS)
        token_loss=(ce*mask).sum()/mask.sum().clamp_min(1)
        pooled=mem[:,base515.GRAPH_TOKENS:,:].mean(dim=1)
        rel_loss=F.cross_entropy(model.rel_head(pooled),rel)
        loss=token_loss+.12*content_loss+.025*rel_loss
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        hist.append(float(token_loss));ph.append(float(content_loss));target_bytes+=sum(bcs)
        if (step+1)%512==0:
            print(json.dumps({'step':step+1,'token_ce':sum(hist[-64:])/len(hist[-64:]),'content_loss':sum(ph[-64:])/len(ph[-64:]),'teacher_plan_p':teacher_prob,'target_bytes':target_bytes},ensure_ascii=False),flush=True)
    return {'steps':steps,'batch':BATCH,'train_s':time.perf_counter()-t0,'target_bytes':target_bytes,'last64_token_ce':sum(hist[-64:])/len(hist[-64:]),'last64_content_loss':sum(ph[-64:])/len(ph[-64:])}

@torch.no_grad()
def eval_examples(model,exs,tok,memory,allow,counts,plan_mode='predicted'):
    model.eval();nll=0.;bts=0;correct=0;toks=0;plan_hits=0;plan_total=0
    for s in range(0,len(exs),16):
        b=exs[s:s+16];enc,dec,tgt,mask,rel,bcs=batch_base(b,tok,memory);mem=model.encode(enc);plog=model.plan_logits(mem)
        pred=predicted_plan(plog,allow)
        teacher=[target_plan(ex['tgt'],allow,counts) for ex in b]
        if plan_mode=='teacher':plan=torch.tensor(teacher,dtype=torch.long)
        elif plan_mode=='shuffled':plan=torch.roll(pred,1,0)
        elif plan_mode=='null':plan=torch.zeros_like(pred)
        else:plan=pred
        for i,tplan in enumerate(teacher):
            ts={x for x in tplan if x};ps={int(x) for x in pred[i] if int(x)};plan_hits+=len(ts&ps);plan_total+=len(ts)
        z,_=model.decode_with_plan(dec,mem,plan);ce=F.cross_entropy(z.reshape(-1,VOCAB),tgt.reshape(-1),reduction='none').view(len(b),base515.TARGET_TOKENS);nll+=float((ce*mask).sum());bts+=sum(bcs);correct+=int(((z.argmax(-1)==tgt)&mask.bool()).sum());toks+=int(mask.sum())
    return {'bpb':nll/max(1,bts)/math.log(2),'token_top1':correct/max(1,toks),'plan_recall':plan_hits/max(1,plan_total),'target_bytes':bts,'target_tokens':toks}

@torch.no_grad()
def generate(model,tok,memory,prompt,allow,decode='greedy',seed=1):
    ctx=tok.enc(prompt.encode('utf-8'))[-base515.CONTEXT_TOKENS:]
    ex={'ctx':ctx,'tgt':[],'ctx_text':prompt,'meta':{'kind':'main'}}
    gi=base515.graph_ids(tok,memory,ex,'true',None);enc=torch.tensor([gi+base515.pad_left(ctx,base515.CONTEXT_TOKENS)],dtype=torch.long)
    model.eval();mem=model.encode(enc);plog=model.plan_logits(mem);plan=predicted_plan(plog,allow);out=[];gen=torch.Generator().manual_seed(seed)
    for _ in range(base515.TARGET_TOKENS):
        dec=torch.tensor([[0]+out],dtype=torch.long);z,_=model.decode_with_plan(dec,mem,plan);q=z[0,-1]
        if decode=='greedy':n=int(q.argmax())
        else:
            q=base515.filter_topk(q/.85,40);n=int(torch.multinomial(F.softmax(q,-1),1,generator=gen))
        out.append(n)
    return tok.dec(out).decode('utf-8','replace'),[int(x) for x in plan[0]]


def plan_text(tok,ids):
    return [tok.dec([i]).decode('utf-8','replace') for i in ids if i]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--warm-checkpoint',required=True);ap.add_argument('--steps',type=int,default=8192);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shutil.copy2(a.tokenizer_model,out/'sp_unigram_4096.model');tok=base515.FixedSP(a.tokenizer_model)
    docs=base57.tokenize_docs(tok,base57.load_jsonl(a.train_docs));memory=base57.MemoryIndex(base57.load_jsonl(a.memory_docs));allow=allowed_vocab(tok);counts=token_counts(docs);assert int(allow.sum())>500
    model=NEXUS518();copied=base515.warm_start(model,a.warm_checkpoint);params=base515.param_count(model);assert params==6194895,params
    print(json.dumps({'params':params,'warm_tensors':len(copied),'docs':len(docs),'memory_chunks':len(memory.chunks),'allowed_plan_tokens':int(allow.sum())},ensure_ascii=False),flush=True)
    tr=train(model,tok,docs,memory,a.steps,allow,counts)
    tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};ev={}
    for i,(name,b) in enumerate(tests.items()):
        xs=base57.build_main_eval(tok,b,7000+i*31,n=256);ev[name]={'predicted':eval_examples(model,xs,tok,memory,allow,counts,'predicted'),'teacher':eval_examples(model,xs,tok,memory,allow,counts,'teacher'),'shuffled':eval_examples(model,xs,tok,memory,allow,counts,'shuffled'),'null':eval_examples(model,xs,tok,memory,allow,counts,'null')}
    gens=[]
    for i,p in enumerate(PROMPTS):
        for dec in ('greedy','sample'):
            txt,pl=generate(model,tok,memory,p,allow,dec,SEED+1800+i);gens.append({'prompt':p,'decode':dec,'plan_ids':pl,'plan_pieces':plan_text(tok,pl),'continuation':txt,**base515.gen_metrics(txt)})
    result={'format':'nexus-r518-discrete-sentence-plan/1','protocol':{'params':params,'base':'R5.15 warm-start semantic encoder-decoder','tokenizer':'frozen R5.12 Unigram4096','steps':a.steps,'batch':BATCH,'plan':'8 predicted informative token-concepts; multi-label plan loss; scheduled teacher-plan exposure decays 65%→20%; inference always predicted plan','purpose':'separate content selection from lexical realization and test plan content via predicted/teacher/shuffled/null controls'},'training':tr,'eval':ev,'generation':gens,'warm_tensors':len(copied),'allowed_plan_tokens':int(allow.sum())}
    (out/'00_R518_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\nPLAN {g['plan_pieces']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','replacement_chars')},ensure_ascii=False)}")
    (out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R518_DISCRETE_SENTENCE_PLAN.pt');print(json.dumps({'train':tr,'eval_pred':{k:v['predicted'] for k,v in ev.items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
