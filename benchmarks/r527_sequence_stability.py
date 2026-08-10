#!/usr/bin/env python3
import argparse,json,math,random,re
from pathlib import Path
import torch
import torch.nn.functional as F
import r524_capacity_sweep as cap

SEED=20260810;CTX=48;TGT=48

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def rep3(tok,ids):
    text=tok.dec(ids);w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))]
    return 1-len(set(tri))/max(1,len(tri)) if tri else 0.0

def examples(tok,raw,n,seed):
    ids=tok.enc(raw);rng=random.Random(seed);out=[]
    for _ in range(n):
        s=rng.randrange(0,len(ids)-CTX-TGT);out.append((ids[s:s+CTX],ids[s+CTX:s+CTX+TGT]))
    return out

@torch.no_grad()
def one(model,tok,ctx,gold):
    ids=list(ctx);pred=[]
    for _ in range(TGT):
        t=int(model(torch.tensor([ids]))[0,-1].argmax());ids.append(t);pred.append(t)
    exact=sum(a==b for a,b in zip(pred,gold))/TGT
    run=0
    for a,b in zip(pred,gold):
        if a!=b:break
        run+=1
    ed=edit_distance(pred,gold);return {'first_correct':float(pred[0]==gold[0]),'token_exact':exact,'correct_run':run,'edit_similarity':1-ed/TGT,'repeat3':rep3(tok,pred),'pred_text':tok.dec(pred),'gold_text':tok.dec(gold)}

def teacher_nll(model,ctx,gold):
    x=torch.tensor([ctx+gold[:-1]]);y=torch.tensor(gold);with_logits=model(x)[0,-TGT:];return float(F.cross_entropy(with_logits,y,reduction='mean'))

def bootstrap_delta(a,b,key,iters=10000):
    rng=random.Random(SEED+527);d=[x[key]-y[key] for x,y in zip(a,b)];n=len(d);means=[]
    for _ in range(iters):means.append(sum(d[rng.randrange(n)] for __ in range(n))/n)
    means.sort();return {'mean_delta':sum(d)/n,'ci95':[means[int(.025*iters)],means[int(.975*iters)-1]],'wins':sum(x>0 for x in d),'ties':sum(x==0 for x in d),'n':n}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--ckpt3',required=True);ap.add_argument('--ckpt8',required=True);ap.add_argument('--ckpt11',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--n',type=int,default=128);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    tok=cap.Tok(a.tokenizer_model);models={}
    for name,path in [('3M',a.ckpt3),('8M',a.ckpt8),('11M',a.ckpt11)]:
        m=cap.LM(cap.CONFIGS[name]);m.load_state_dict(torch.load(path,map_location='cpu')['state_dict']);m.eval();models[name]=m
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);result={'format':'nexus-r527-sequence-stability/1','protocol':{'paired_context_tokens':CTX,'greedy_target_tokens':TGT,'examples_per_domain':a.n,'same_examples_all_capacities':True,'padding':'none'},'domains':{},'paired':{}}
    all_rows={k:[] for k in models}
    for di,p in enumerate(a.tests):
        ex=examples(tok,Path(p).read_text(encoding='utf-8'),a.n,SEED+di*1000);dom=Path(p).stem;result['domains'][dom]={}
        for name,m in models.items():
            rows=[]
            for ctx,gold in ex:
                r=one(m,tok,ctx,gold);r['teacher_nll']=teacher_nll(m,ctx,gold);rows.append(r);all_rows[name].append(r)
            result['domains'][dom][name]={'first_correct':sum(x['first_correct'] for x in rows)/len(rows),'token_exact':sum(x['token_exact'] for x in rows)/len(rows),'correct_run':sum(x['correct_run'] for x in rows)/len(rows),'edit_similarity':sum(x['edit_similarity'] for x in rows)/len(rows),'repeat3':sum(x['repeat3'] for x in rows)/len(rows),'teacher_nll':sum(x['teacher_nll'] for x in rows)/len(rows),'samples':rows[:8]}
    for big in ('8M','11M'):
        result['paired'][f'{big}_minus_3M']={k:bootstrap_delta(all_rows[big],all_rows['3M'],k) for k in ('first_correct','token_exact','correct_run','edit_similarity')}
    result['paired']['11M_minus_8M']={k:bootstrap_delta(all_rows['11M'],all_rows['8M'],k) for k in ('first_correct','token_exact','correct_run','edit_similarity')}
    (out/'00_R527_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for dom,v in result['domains'].items():
        lines.append('=== '+dom+' ===')
        for name,m in v.items():lines.append(f"{name}: first={m['first_correct']:.4f} exact={m['token_exact']:.4f} run={m['correct_run']:.4f} edit={m['edit_similarity']:.4f} rep3={m['repeat3']:.4f} nll={m['teacher_nll']:.4f}")
    lines.append('\nPAIRED\n'+json.dumps(result['paired'],ensure_ascii=False,indent=2));(out/'01_SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
