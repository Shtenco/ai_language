#!/usr/bin/env python3
import argparse,json,random,re
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn.functional as F
import r57_concept_graph_language as base

MODE='D_LOGIC_CYBER';SEED=20260810
PROMPTS=base.PROMPTS+[
 'Россия занимает большую территорию, поэтому',
 'Исследователь проверил данные и пришёл к выводу, что',
 'Память помогает рассуждению, потому что',
 'Причина отличается от простой корреляции тем, что',
 'Для достижения цели система должна',
 'После ошибки программа изменила своё состояние и',
 'В книге автор рассказывает о том, как',
 'Утром город проснулся, и на улицах']

class Tok:
    def __init__(self,path):self.model_path=Path(path);self.sp=spm.SentencePieceProcessor(model_file=str(path));self.name='UNIGRAM4096-WARM'
    def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids):return self.sp.decode([int(x) for x in ids]).encode('utf-8')
    def vocab_bytes(self):return self.model_path.stat().st_size

def metrics(text):
    words=re.findall(r'[А-Яа-яЁёA-Za-z]+',text);tri=[tuple(x.lower() for x in words[i:i+3]) for i in range(max(0,len(words)-2))]
    return {'words':len(words),'unique_word_ratio':len(set(x.lower() for x in words))/max(1,len(words)),'repeated_trigram_rate':1-len(set(tri))/max(1,len(tri)) if tri else 0.0,'cyrillic_share':sum(bool(re.search(r'[А-Яа-яЁё]',x)) for x in words)/max(1,len(words)),'dash_per_100':100*text.count('—')/max(1,len(text))}

def prefix(tok,memory,prompt):
    return base.prefix_ids(MODE,tok,prompt,memory,{'kind':'main'})

def input_ids(kind,tok,memory,prompt):
    ctx=tok.enc(prompt.encode('utf-8'))[-base.CONTEXT_TOKENS:];p=prefix(tok,memory,prompt)
    if kind=='legacy_unk_leftpad':return p+[0]*(base.CONTEXT_TOKENS-len(ctx))+ctx,len(ctx),base.CONTEXT_TOKENS-len(ctx),p.count(0)
    if kind=='clean_variable':return p+ctx,len(ctx),0,p.count(0)
    raise ValueError(kind)

@torch.no_grad()
def generate(model,kind,tok,memory,prompt,decode,seed,max_new=48):
    ids,ctx_len,ctx_unk_pad,pfx_zeros=input_ids(kind,tok,memory,prompt);gen=torch.Generator().manual_seed(seed);out=[];model.eval()
    for _ in range(max_new):
        z=model(torch.tensor([ids],dtype=torch.long))[0,-1]
        if decode=='greedy':t=int(z.argmax())
        else:
            z=z/.80;p=F.softmax(z,dim=-1);vals,ix=torch.sort(p,descending=True);cs=torch.cumsum(vals,0);keep=cs<=.92;keep[0]=True;vals=vals[keep];ix=ix[keep];vals=vals/vals.sum();t=int(ix[torch.multinomial(vals,1,generator=gen)])
        ids.append(t);out.append(t)
    text=tok.dec(out).decode('utf-8','replace')
    return {'kind':kind,'decode':decode,'prompt':prompt,'context_tokens':ctx_len,'context_unk_padding':ctx_unk_pad,'prefix_zero_ids':pfx_zeros,'continuation':text,**metrics(text)}

def heldout_cases(tok,raw,n=12):
    ids=tok.enc(raw);rng=random.Random(517);cases=[]
    for _ in range(n):
        start=rng.randrange(0,len(ids)-base.CONTEXT_TOKENS-base.TARGET_TOKENS-1);ctx=ids[start:start+base.CONTEXT_TOKENS];gold=ids[start+base.CONTEXT_TOKENS:start+base.CONTEXT_TOKENS+base.TARGET_TOKENS]
        cases.append({'prompt':tok.dec(ctx).decode('utf-8','replace'),'gold':tok.dec(gold).decode('utf-8','replace')})
    return cases

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--heldout',required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);tok=Tok(a.tokenizer_model);assert tok.sp.unk_id()==0 and tok.sp.pad_id()==-1
    memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));model=base.LM();ck=torch.load(a.checkpoint,map_location='cpu');model.load_state_dict(ck['state_dict']);assert base.param_count(model)==2998620
    rows=[]
    for i,prompt in enumerate(PROMPTS):
        for kind in ('legacy_unk_leftpad','clean_variable'):
            for dec in ('greedy','sample'):rows.append(generate(model,kind,tok,memory,prompt,dec,SEED+1000+i))
    held=heldout_cases(tok,Path(a.heldout).read_bytes())
    full=[]
    for i,c in enumerate(held):
        # 48-token contexts have zero context padding under both routes; they diagnose training-geometry generation itself.
        arow=generate(model,'legacy_unk_leftpad',tok,memory,c['prompt'],'greedy',SEED+3000+i);arow['gold']=c['gold'];full.append(arow)
    def agg(sub):
        return {k:sum(float(r[k]) for r in sub)/max(1,len(sub)) for k in ('words','unique_word_ratio','repeated_trigram_rate','cyrillic_share','dash_per_100')}
    groups={}
    for kind in ('legacy_unk_leftpad','clean_variable'):
        for dec in ('greedy','sample'):
            q=[r for r in rows if r['kind']==kind and r['decode']==dec];groups[kind+'_'+dec]=agg(q)
    result={'format':'nexus-r517-pad-clean/1','protocol':{'checkpoint':'R5.12 32768-step D_LOGIC_CYBER','params':base.param_count(model),'tokenizer_unk_id':tok.sp.unk_id(),'tokenizer_pad_id':tok.sp.pad_id(),'diagnosis':'legacy free-generation left-padded short prompts with token id 0 even though id0 is <unk>; clean route removes context left padding and relies on RoPE variable length'},'aggregate':groups,'short_prompt_generations':rows,'heldout_48token_generations':full}
    (out/'00_R517_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for r in rows:lines.append(f"[{r['kind']} / {r['decode']}] ctx={r['context_tokens']} unkpad={r['context_unk_padding']} pfx0={r['prefix_zero_ids']}\n{r['prompt']}\n{r['continuation']}\nMETRICS uniq={r['unique_word_ratio']:.3f} rep3={r['repeated_trigram_rate']:.3f} cyr={r['cyrillic_share']:.3f}")
    lines.append('\n=== HELDOUT EXACT-48 TOKEN CONTEXT ===')
    for r in full:lines.append(f"CTX: {r['prompt']}\nGEN: {r['continuation']}\nGOLD: {r['gold']}\nMETRICS uniq={r['unique_word_ratio']:.3f} rep3={r['repeated_trigram_rate']:.3f}")
    (out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');print(json.dumps({'aggregate':groups,'unk_id':tok.sp.unk_id(),'pad_id':tok.sp.pad_id(),'sample_padding':[(r['context_tokens'],r['context_unk_padding']) for r in rows[:8]]},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
