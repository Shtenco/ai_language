#!/usr/bin/env python3
import argparse,collections,csv,hashlib,json,math,os,re,tempfile
from pathlib import Path

SIZES=(512,1024,2048,4096)
BYTE_IDS=256

WORD_RE=re.compile(r"\s+|[\w]+|[^\w\s]+",re.UNICODE)
LEX_RE=re.compile(r" ?\w+| ?[^\w\s]+|\s+",re.UNICODE)

def utf8_chunks(text, mode='word'):
    rx=WORD_RE if mode=='word' else LEX_RE
    return [x.encode('utf-8') for x in rx.findall(text)]

def entropy_bits(ids):
    c=collections.Counter(ids); n=len(ids)
    return sum(v*math.log2(n/v) for v in c.values()) if n else 0.0

class GreedyDict:
    def __init__(self,pieces,name):
        self.name=name; self.pieces=[bytes([i]) for i in range(256)]+pieces
        self.trie={}
        for i,p in enumerate(self.pieces[256:],256):
            n=self.trie
            for b in p:n=n.setdefault(b,{})
            n[-1]=i
    def enc(self,b):
        out=[];i=0
        while i<len(b):
            n=self.trie;j=i;best=None;bj=i
            while j<len(b) and b[j] in n:
                n=n[b[j]];j+=1
                if -1 in n:best=n[-1];bj=j
            if best is None:out.append(b[i]);i+=1
            else:out.append(best);i=bj
        return out
    def dec(self,ids):return b''.join(self.pieces[i] for i in ids)
    def vocab_bytes(self):return sum(len(x) for x in self.pieces)

def candidate_counts(raw,kind):
    text=raw.decode('utf-8')
    c=collections.Counter()
    if kind=='word':
        for q in utf8_chunks(text,'word'):
            if 2<=len(q)<=80:c[q]+=1
    else:
        for q in utf8_chunks(text,'lex'):
            if 2<=len(q)<=80:c[q]+=1
            if 5<=len(q)<=80:
                for n in (2,3,4,5,6,8,12,16):
                    if n<len(q):c[q[:n]]+=1;c[q[-n:]]+=1
    return c

def build_dict(raw,vocab,kind):
    c=candidate_counts(raw,kind)
    # Estimated MDL gain: repeated byte saving minus one-time dictionary description.
    def score(it):
        p,f=it
        saving=(f-1)*(len(p)-1)
        dict_cost=len(p)+2
        return (saving-dict_cost,f,len(p),p)
    ranked=sorted(c.items(),key=score,reverse=True)
    pieces=[];seen={bytes([i]) for i in range(256)}
    for p,f in ranked:
        if p in seen:continue
        if score((p,f))[0] <= 0:continue
        pieces.append(p);seen.add(p)
        if len(pieces)>=vocab-256:break
    # deterministic fill if positive-gain pool is smaller
    if len(pieces)<vocab-256:
        for p,f in sorted(c.items(),key=lambda z:(-z[1],-len(z[0]),z[0])):
            if p not in seen:pieces.append(p);seen.add(p)
            if len(pieces)>=vocab-256:break
    return GreedyDict(pieces[:vocab-256],f'{kind.upper()}DICT{vocab}')

class SPWrap:
    def __init__(self,sp,name,model_path):self.sp=sp;self.name=name;self.model_path=Path(model_path)
    def enc(self,b):return self.sp.encode(b.decode('utf-8'),out_type=int)
    def dec(self,ids):return self.sp.decode(ids).encode('utf-8')
    def vocab_bytes(self):return self.model_path.stat().st_size

def train_sp(raw,vocab,model_type,outdir):
    import sentencepiece as spm
    txt=Path(outdir)/f'_train_{model_type}_{vocab}.txt';txt.write_bytes(raw)
    prefix=str(Path(outdir)/f'sp_{model_type}_{vocab}')
    # byte fallback keeps open vocabulary; identity normalization preserves the source surface.
    spm.SentencePieceTrainer.train(
        input=str(txt),model_prefix=prefix,vocab_size=vocab,model_type=model_type,
        character_coverage=1.0,byte_fallback=True,normalization_rule_name='identity',
        remove_extra_whitespaces=False,add_dummy_prefix=False,split_by_whitespace=False,
        bos_id=-1,eos_id=-1,pad_id=-1,unk_id=0,hard_vocab_limit=False,
        input_sentence_size=1000000,shuffle_input_sentence=False,num_threads=2)
    sp=spm.SentencePieceProcessor(model_file=prefix+'.model')
    return SPWrap(sp,f'{model_type.upper()}{vocab}',prefix+'.model')

def metrics(tok,raw):
    ids=tok.enc(raw); dec=tok.dec(ids); exact=(dec==raw)
    if not exact:
        # SentencePiece may not preserve a terminal newline byte-for-byte; report it instead of hiding it.
        common=sum(a==b for a,b in zip(dec,raw)); exact_ratio=common/max(1,max(len(dec),len(raw)))
    else:exact_ratio=1.0
    eb=entropy_bits(ids); vb=tok.vocab_bytes()*8
    return {
        'raw_bytes':len(raw),'tokens':len(ids),'bytes_per_token':len(raw)/max(1,len(ids)),
        'token_entropy_bits_per_token':eb/max(1,len(ids)),
        'ideal_stream_bpb':eb/max(1,len(raw)),
        'vocab_bytes':tok.vocab_bytes(),
        'mdl_bpb_with_vocab':(eb+vb)/max(1,len(raw)),
        'roundtrip_exact':exact,'roundtrip_ratio':exact_ratio,
        'sha_ids':hashlib.sha256(','.join(map(str,ids[:100000])).encode()).hexdigest()
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    train=Path(a.train).read_bytes(); tests={Path(p).stem:Path(p).read_bytes() for p in a.tests}
    toks=[]
    # Byte baseline
    class Byte:
        name='BYTE256'
        def enc(self,b):return list(b)
        def dec(self,ids):return bytes(ids)
        def vocab_bytes(self):return 256
    toks.append(Byte())
    for v in SIZES:
        toks.append(build_dict(train,v,'word'));toks.append(build_dict(train,v,'lex'))
    for v in SIZES:
        for typ in ('bpe','unigram'):
            try:toks.append(train_sp(train,v,typ,out))
            except Exception as e:
                (out/f'ERROR_{typ}_{v}.txt').write_text(repr(e));print('SP ERROR',typ,v,repr(e),flush=True)
    rows=[]
    for tok in toks:
        print('TOKENIZER',tok.name,flush=True)
        for split,raw in [('train',train),*tests.items()]:
            r={'tokenizer':tok.name,'split':split,**metrics(tok,raw)};rows.append(r)
            print(json.dumps({k:r[k] for k in ('tokenizer','split','bytes_per_token','ideal_stream_bpb','mdl_bpb_with_vocab','roundtrip_exact')},ensure_ascii=False),flush=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with open(out/'01_TOKENIZER_METRICS.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    (out/'00_TOKENIZER_METRICS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    # Pareto ranking: in-domain compression + shift robustness + dictionary overhead. Exclude non-exact tokenizers from finalists.
    by=collections.defaultdict(dict)
    for r in rows:by[r['tokenizer']][r['split']]=r
    split_names=list(tests)
    score=[]
    for name,d in by.items():
        exact=all(x['roundtrip_exact'] for x in d.values())
        testvals=[d[s]['mdl_bpb_with_vocab'] for s in split_names if s in d]
        score.append({'tokenizer':name,'all_exact':exact,'train_bpt':d['train']['bytes_per_token'],'mean_test_mdl_bpb':sum(testvals)/len(testvals),'worst_test_mdl_bpb':max(testvals),'vocab_bytes':d['train']['vocab_bytes']})
    score.sort(key=lambda x:(not x['all_exact'],x['mean_test_mdl_bpb'],x['worst_test_mdl_bpb']))
    (out/'02_RANKING.json').write_text(json.dumps(score,indent=2))
    with open(out/'02_RANKING.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=score[0].keys());w.writeheader();w.writerows(score)
    (out/'README_RU.md').write_text('# NEXUS R5.2 TOKENIZER TOURNAMENT\n\nTokenizer-only preselection. Families: BYTE256, lossless WORDDICT, lossless MDL/lexical DICT, SentencePiece BPE and Unigram; vocab 512/1024/2048/4096. Primary preselection quantities are bytes/token and ideal/MDL bits per original byte. Exact round-trip is a hard gate.\n\nTop exact candidates:\n'+ '\n'.join(f"- {x['tokenizer']}: mean test MDL-BPB={x['mean_test_mdl_bpb']:.4f}, train B/token={x['train_bpt']:.3f}" for x in score[:10]))
if __name__=='__main__':main()
