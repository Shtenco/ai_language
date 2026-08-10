#!/usr/bin/env python3
import argparse,json,random,re
from pathlib import Path
import sentencepiece as spm
import torch
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER';SEED=20260810;KS=[0,1,2,4,8,16,24,32]

class Tok:
    def __init__(self,path):self.sp=spm.SentencePieceProcessor(model_file=str(path));self.model_path=Path(path);self.name='UNIGRAM4096-WARM'
    def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids):return self.sp.decode([int(x) for x in ids]).encode('utf-8')
    def vocab_bytes(self):return self.model_path.stat().st_size

def text_metrics(text):
    w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))]
    return {'unique_word_ratio':len(set(x.lower() for x in w))/max(1,len(w)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0.0}

@torch.no_grad()
def run_case(model,tok,memory,ex,k):
    ctx=list(ex['ctx']);gold=list(ex['tgt'][:base.TARGET_TOKENS]);p=list(base.prefix_ids(MODE,tok,ex['ctx_text'],memory,ex['meta']));ids=p+ctx+gold[:k];out=[]
    for _ in range(base.TARGET_TOKENS-k):
        t=int(model(torch.tensor([ids],dtype=torch.long))[0,-1].argmax());ids.append(t);out.append(t)
    pred=gold[:k]+out;suffix_gold=gold[k:];suffix_pred=out
    exact=sum(a==b for a,b in zip(suffix_gold,suffix_pred))/max(1,len(suffix_gold));prefix_run=0
    for a,b in zip(suffix_gold,suffix_pred):
        if a!=b:break
        prefix_run+=1
    text=tok.dec(suffix_pred).decode('utf-8','replace')
    return {'k':k,'suffix_exact':exact,'correct_run_after_release':prefix_run,'suffix_text':text,'gold_suffix':tok.dec(suffix_gold).decode('utf-8','replace'),**text_metrics(text)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--heldout',required=True);ap.add_argument('--out',required=True);ap.add_argument('--n',type=int,default=64);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tok=Tok(a.tokenizer_model);memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));model=base.LM();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['state_dict']);model.eval();assert base.param_count(model)==2998620
    raw=Path(a.heldout).read_bytes();examples=base.build_main_eval(tok,raw,20520,n=a.n);rows=[]
    for i,e in enumerate(examples):
        for k in KS:rows.append({'case':i,**run_case(model,tok,memory,e,k)})
    agg={}
    for k in KS:
        q=[r for r in rows if r['k']==k];agg[str(k)]={m:sum(float(x[m]) for x in q)/len(q) for m in ('suffix_exact','correct_run_after_release','unique_word_ratio','repeated_trigram_rate')}
    r={'format':'nexus-r520-prefix-rescue/1','protocol':{'base':'R5.12 32K frozen','params':2998620,'heldout_cases':len(examples),'context_tokens':48,'release_after_gold_tokens':KS,'padding':'none in context'},'aggregate':agg,'rows':rows}
    (out/'00_R520_RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for case in range(min(8,len(examples))):
        lines.append(f'=== CASE {case} ===')
        for k in KS:
            x=next(z for z in rows if z['case']==case and z['k']==k);lines.append(f"k={k} exact={x['suffix_exact']:.3f} run={x['correct_run_after_release']} rep3={x['repeated_trigram_rate']:.3f}\nGEN {x['suffix_text']}\nGOLD {x['gold_suffix']}")
    (out/'01_RAW_RESCUE.txt').write_text('\n\n'.join(lines),encoding='utf-8');print(json.dumps(agg,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
