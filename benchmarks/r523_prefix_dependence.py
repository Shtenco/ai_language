#!/usr/bin/env python3
import argparse,json,math,random
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn.functional as F
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER';SEED=20260810

class Tok:
    def __init__(self,path):self.sp=spm.SentencePieceProcessor(model_file=str(path));self.model_path=Path(path);self.name='UNIGRAM4096-WARM'
    def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids):return self.sp.decode([int(x) for x in ids]).encode('utf-8')
    def vocab_bytes(self):return self.model_path.stat().st_size

def prefix_for(kind,tok,memory,src):
    if kind=='true':return base.prefix_ids(MODE,tok,src['ctx_text'],memory,src['meta'])
    if kind=='fixed':return base.prefix_ids(MODE,tok,'Система продолжает русский текст, сохраняя тему, причинную связь и факты.',memory,{'kind':'main'})
    if kind=='legacy_zero':return [0]*base.PREFIX_TOKENS
    raise ValueError(kind)

def eval_main(model,examples,kind,tok,memory,batch=16):
    nll=0.;bts=0;toks=0;correct=0;zeros=[]
    model.eval()
    with torch.no_grad():
        for o in range(0,len(examples),batch):
            ex=examples[o:o+batch];seqs=[];starts=[];lens=[]
            for i,e in enumerate(ex):
                src=e if kind!='shuffled' else ex[(i+1)%len(ex)]
                p=base.prefix_ids(MODE,tok,src['ctx_text'],memory,src['meta']) if kind=='shuffled' else prefix_for(kind,tok,memory,e)
                zeros.append(p.count(0));ctx=list(e['ctx'][-base.CONTEXT_TOKENS:]);tgt=list(e['tgt'][:base.TARGET_TOKENS]);assert len(ctx)==48 and len(tgt)==48
                seqs.append(p+ctx+tgt);starts.append(base.PREFIX_TOKENS+base.CONTEXT_TOKENS-1);lens.append(len(tgt));bts+=len(tok.dec(tgt))
            x=torch.tensor([s[:-1] for s in seqs]);y=torch.tensor([s[1:] for s in seqs]);m=torch.zeros_like(y,dtype=torch.bool)
            for i,(st,L) in enumerate(zip(starts,lens)):m[i,st:st+L]=True
            z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);nll+=float((ce*m).sum());toks+=int(m.sum());correct+=int(((z.argmax(-1)==y)&m).sum())
    return {'bpb':nll/max(1,bts)/math.log(2),'token_top1':correct/max(1,toks),'mean_prefix_zero_ids':sum(zeros)/max(1,len(zeros)),'max_prefix_zero_ids':max(zeros) if zeros else 0}

def eval_aux(model,examples,kind,tok,memory,batch=16):
    # Variable-length actual contexts, right-padding only after complete target. This avoids injecting UNK before the context.
    nll=0.;bts=0;toks=0;correct=0;zeros=[]
    model.eval()
    with torch.no_grad():
        for o in range(0,len(examples),batch):
            ex=examples[o:o+batch];seqs=[];starts=[];lens=[]
            for i,e in enumerate(ex):
                src=e if kind!='shuffled' else ex[(i+1)%len(ex)]
                if kind=='true':p=base.prefix_ids(MODE,tok,e['ctx_text'],memory,e['meta'])
                elif kind=='shuffled':p=base.prefix_ids(MODE,tok,src['ctx_text'],memory,src['meta'])
                elif kind=='fixed':p=prefix_for('fixed',tok,memory,e)
                elif kind=='legacy_zero':p=[0]*base.PREFIX_TOKENS
                else:raise ValueError(kind)
                zeros.append(p.count(0));ctx=list(e['ctx']);tgt=list(e['tgt'][:base.TARGET_TOKENS]);seq=p+ctx+tgt;seqs.append(seq);starts.append(len(p)+len(ctx)-1);lens.append(len(tgt));bts+=max(1,len(tok.dec(tgt)))
            mx=max(map(len,seqs));xs=[];ys=[];ms=[]
            for s,st,L in zip(seqs,starts,lens):
                q=s+[0]*(mx-len(s));x=q[:-1];y=q[1:];m=[0]*len(x)
                for j in range(st,min(st+L,len(m))):m[j]=1
                xs.append(x);ys.append(y);ms.append(m)
            x=torch.tensor(xs);y=torch.tensor(ys);m=torch.tensor(ms,dtype=torch.bool);z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);nll+=float((ce*m).sum());toks+=int(m.sum());correct+=int(((z.argmax(-1)==y)&m).sum())
    return {'bpb':nll/max(1,bts)/math.log(2),'token_top1':correct/max(1,toks),'mean_prefix_zero_ids':sum(zeros)/max(1,len(zeros))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);random.seed(SEED);torch.manual_seed(SEED)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tok=Tok(a.tokenizer_model);assert tok.sp.unk_id()==0 and tok.sp.pad_id()==-1
    memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));model=base.LM();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['state_dict']);assert base.param_count(model)==2998620
    modes=('true','shuffled','fixed','legacy_zero');result={'format':'nexus-r523-prefix-dependence/1','protocol':{'base':'R5.12 32K D cortex frozen','params':2998620,'controls':modes,'primary_causal_contrast':'true vs shuffled; both use same D-prefix construction and 32 slots','tokenizer_unk_id':tok.sp.unk_id(),'pad_id':tok.sp.pad_id(),'memory':'UD-derived direct-source memory; same index for all controls'},'main':{}}
    for i,p in enumerate(a.tests):
        ex=base.build_main_eval(tok,Path(p).read_bytes(),52300+i*41,n=256);result['main'][Path(p).stem]={m:eval_main(model,ex,m,tok,memory) for m in modes}
    logic=base.build_aux_eval(tok,'logic',52391,n=192);cyber=base.build_aux_eval(tok,'cyber',52392,n=192)
    result['logic']={m:eval_aux(model,logic,m,tok,memory) for m in modes};result['cyber']={m:eval_aux(model,cyber,m,tok,memory) for m in modes}
    # Summarize true-shuffled differences; negative delta means matching state helps.
    result['true_minus_shuffled_bpb']={k:v['true']['bpb']-v['shuffled']['bpb'] for k,v in {**result['main'],'logic':result['logic'],'cyber':result['cyber']}.items()}
    (out/'00_R523_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
