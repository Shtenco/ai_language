#!/usr/bin/env python3
import argparse,json,math,random,re,time
from pathlib import Path
import torch
import torch.nn.functional as F
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER';SEED=20260810;STEPS=4096;BATCH=8
PROMPTS=base.PROMPTS+['Россия занимает большую территорию, поэтому','Исследователь проверил данные и пришёл к выводу, что','Память помогает рассуждению, потому что','Причина отличается от простой корреляции тем, что','Для достижения цели система должна','После ошибки программа изменила своё состояние и','В книге автор рассказывает о том, как','Утром город проснулся, и на улицах']

def shorten_main(ex,rng,tok,min_ctx=4):
    if ex['meta'].get('kind')!='main':return ex
    L=rng.randint(min_ctx,base.CONTEXT_TOKENS)
    q=dict(ex);q['ctx']=list(ex['ctx'][-L:]);q['ctx_text']=tok.dec(q['ctx']).decode('utf-8','replace')
    return q

def pack_variable(examples,tok,memory):
    seqs=[];starts=[];lens=[];tb=0
    for ex in examples:
        p=list(base.prefix_ids(MODE,tok,ex['ctx_text'],memory,ex['meta']));ctx=list(ex['ctx']);tgt=list(ex['tgt'][:base.TARGET_TOKENS]);tb+=max(1,len(tok.dec(tgt)))
        seq=p+ctx+tgt;seqs.append(seq);starts.append(len(p)+len(ctx)-1);lens.append(len(tgt))
    maxlen=max(len(s) for s in seqs);xs=[];ys=[];ms=[]
    for seq,st,L in zip(seqs,starts,lens):
        padded=seq+[0]*(maxlen-len(seq));x=padded[:-1];y=padded[1:];m=[0]*len(x)
        for j in range(st,min(st+L,len(m))):m[j]=1
        xs.append(x);ys.append(y);ms.append(m)
    return torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long),torch.tensor(ms,dtype=torch.bool),tb

def sample_ex(rng,docs,tok):
    u=rng.random()
    if u<.94:return base.make_main_example(rng,docs,tok)
    if u<.97:return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)

def eval_variable(model,examples,tok,memory,ctx_len):
    nll=0.;bts=0;toks=0;corr=0;model.eval()
    with torch.no_grad():
        for o in range(0,len(examples),16):
            ex=[]
            for q in examples[o:o+16]:
                z=dict(q)
                if z['meta'].get('kind')=='main':
                    z['ctx']=list(z['ctx'][-ctx_len:]);z['ctx_text']=tok.dec(z['ctx']).decode('utf-8','replace')
                ex.append(z)
            x,y,m,tb=pack_variable(ex,tok,memory);logits=model(x);ce=F.cross_entropy(logits.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);nll+=float((ce*m).sum());bts+=tb;toks+=int(m.sum());corr+=int(((logits.argmax(-1)==y)&m).sum())
    return {'bpb':nll/max(1,bts)/math.log(2),'token_top1':corr/max(1,toks)}

def gen_input(tok,memory,prompt):
    ctx=tok.enc(prompt.encode('utf-8'))[-base.CONTEXT_TOKENS:];return list(base.prefix_ids(MODE,tok,prompt,memory,{'kind':'main'}))+ctx,len(ctx)

def met(text):
    w=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in w[i:i+3]) for i in range(max(0,len(w)-2))]
    return {'words':len(w),'unique_word_ratio':len(set(x.lower() for x in w))/max(1,len(w)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0.0,'cyrillic_share':sum(bool(re.search(r'[А-Яа-яЁё]',x)) for x in w)/max(1,len(w))}

@torch.no_grad()
def generate(model,tok,memory,prompt,decode,seed):
    ids,L=gen_input(tok,memory,prompt);g=torch.Generator().manual_seed(seed);out=[];model.eval()
    for _ in range(base.TARGET_TOKENS):
        z=model(torch.tensor([ids]))[0,-1]
        if decode=='greedy':t=int(z.argmax())
        else:
            p=F.softmax(z/.8,dim=-1);v,ix=torch.sort(p,descending=True);cs=torch.cumsum(v,0);keep=cs<=.92;keep[0]=True;v=v[keep];ix=ix[keep];v=v/v.sum();t=int(ix[torch.multinomial(v,1,generator=g)])
        ids.append(t);out.append(t)
    text=tok.dec(out).decode('utf-8','replace');return {'prompt':prompt,'decode':decode,'context_tokens':L,'continuation':text,**met(text)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--train-text',required=True);ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--steps',type=int,default=STEPS);a=ap.parse_args();torch.set_num_threads(a.threads);random.seed(SEED);torch.manual_seed(SEED)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    # Reuse exact tokenizer from R5.12; ID0 is UNK and is never inserted before real context by this trainer.
    import sentencepiece as spm
    class Tok:
        def __init__(self,path):self.model_path=Path(path);self.sp=spm.SentencePieceProcessor(model_file=str(path));self.name='UNIGRAM4096-WARM'
        def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
        def dec(self,ids):return self.sp.decode([int(x) for x in ids]).encode('utf-8')
        def vocab_bytes(self):return self.model_path.stat().st_size
    tok=Tok(a.tokenizer_model);assert tok.sp.unk_id()==0 and tok.sp.pad_id()==-1
    docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));model=base.LM();ck=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(ck['state_dict']);assert base.param_count(model)==2998620
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01,betas=(.9,.95));rng=random.Random(SEED+518);hist=[];ctxhist=[];target_bytes=0;t0=time.perf_counter();model.train()
    for step in range(a.steps):
        ex=[]
        for _ in range(BATCH):
            q=sample_ex(rng,docs,tok)
            if q['meta'].get('kind')=='main':
                q=shorten_main(q,rng,tok);ctxhist.append(len(q['ctx']))
            ex.append(q)
        for pg in opt.param_groups:pg['lr']=2e-4*base.lr_factor(step,a.steps)
        x,y,m,tb=pack_variable(ex,tok,memory);opt.zero_grad(set_to_none=True);z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);loss=(ce*m).sum()/tb;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();target_bytes+=tb;hist.append(float(loss))
        if (step+1)%256==0:print('TRAIN',step+1,'npb',sum(hist[-64:])/64,'mean_ctx',sum(ctxhist[-512:])/max(1,len(ctxhist[-512:])),flush=True)
    tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};ev={}
    for i,(name,raw) in enumerate(tests.items()):
        ex=base.build_main_eval(tok,raw,7500+i*29,n=256);ev[name]={str(L):eval_variable(model,ex,tok,memory,L) for L in (8,16,32,48)}
    gens=[]
    for i,p in enumerate(PROMPTS):
        for d in ('greedy','sample'):gens.append(generate(model,tok,memory,p,d,SEED+5000+i))
    logic=base.build_aux_eval(tok,'logic',88518,n=192);cyber=base.build_aux_eval(tok,'cyber',99518,n=192);lev=base.evaluate_examples(model,MODE,logic,tok,memory);cev=base.evaluate_examples(model,MODE,cyber,tok,memory);la=base.aux_generation_accuracy(model,MODE,tok,memory,'logic',12518,n=64);ca=base.aux_generation_accuracy(model,MODE,tok,memory,'cyber',22518,n=64)
    r={'format':'nexus-r518-variable-context/2','protocol':{'warmstart':'R5.12 32768','params':base.param_count(model),'steps':a.steps,'batch':BATCH,'main_context_tokens':'uniform integer 4..48; keep suffix immediately preceding same target','left_context_padding':'NONE','graph_context':'recomputed exactly from the same truncated context tokens','batch_padding':'right of complete prefix+context+target only; causal real positions cannot attend to it','tokenizer_unk_id':tok.sp.unk_id(),'pad_id':tok.sp.pad_id()},'training':{'seconds':time.perf_counter()-t0,'target_bytes':target_bytes,'last64_npb':sum(hist[-64:])/max(1,len(hist[-64:])),'mean_main_ctx':sum(ctxhist)/max(1,len(ctxhist))},'eval_by_context_length':ev,'logic_teacher_forced':lev,'cyber_teacher_forced':cev,'logic_generation':la,'cyber_generation':ca,'generation':gens}
    (out/'00_R518_RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(f"[{g['decode']}] ctx={g['context_tokens']} {g['prompt']}\n{g['continuation']}\nuniq={g['unique_word_ratio']:.3f} rep3={g['repeated_trigram_rate']:.3f}" for g in gens),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':r['protocol']},out/'R518_VARIABLE_CONTEXT_3M.pt');print(json.dumps({'training':r['training'],'eval':ev,'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
