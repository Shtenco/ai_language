#!/usr/bin/env python3
import argparse,hashlib,json,re
from pathlib import Path


def cap_utf8(b,n):
    if len(b)<=n:return b
    x=b[:n]
    while x:
        try:x.decode('utf-8');return x
        except UnicodeDecodeError:x=x[:-1]
    return b''


def conllu_text(path):
    out=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('# text = '):
            t=line[9:].strip()
            if t:out.append(t)
    return ('\n'.join(out)+'\n').encode('utf-8')


def clean_text(s):
    s=s.replace('\r','\n').replace('\x00',' ')
    s=re.sub(r'[ \t]+',' ',s)
    s=re.sub(r'\n{4,}','\n\n',s)
    return s.strip()


def add_until(parts, text, limit):
    if sum(len(x) for x in parts)>=limit:return
    b=(clean_text(text)+'\n\n').encode('utf-8')
    left=limit-sum(len(x) for x in parts)
    if left<=0:return
    parts.append(cap_utf8(b,left))


def stream_dataset(name, config=None, seed=2026):
    from datasets import load_dataset
    kwargs={'split':'train','streaming':True}
    ds=load_dataset(name,config,**kwargs) if config else load_dataset(name,**kwargs)
    try:ds=ds.shuffle(seed=seed,buffer_size=1000)
    except Exception:pass
    return ds


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--synt-train',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--heritage-train-mb',type=int,default=10)
    ap.add_argument('--wiki-train-mb',type=int,default=6)
    ap.add_argument('--synt-train-mb',type=int,default=4)
    ap.add_argument('--heldout-mb',type=int,default=2)
    a=ap.parse_args()
    o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
    MB=1024*1024
    train=[];heritage_test=[];wiki_test=[];meta={'sources':{},'errors':[]}

    # 1) Curated Russian classical prose/literature. Documents are split by stable hash,
    # so held-out works are never copied into training.
    try:
        ds=stream_dataset('maxzt/RuHeritage-Corpus',seed=20260809)
        train_lim=a.heritage_train_mb*MB;test_lim=a.heldout_mb*MB
        seen=0
        for r in ds:
            text=r.get('text') or ''
            if len(text)<200:continue
            key=(str(r.get('author',''))+'\x1f'+str(r.get('title',''))).encode('utf-8','ignore')
            bucket=hashlib.sha256(key).digest()[0]%10
            if bucket<8:add_until(train,text,train_lim)
            else:add_until(heritage_test,text,test_lim)
            seen+=1
            if sum(map(len,train))>=train_lim and sum(map(len,heritage_test))>=test_lim:break
        meta['sources']['RuHeritage']={'records_seen':seen,'train_bytes':sum(map(len,train)),'test_bytes':sum(map(len,heritage_test))}
    except Exception as e:
        meta['errors'].append('RuHeritage: '+repr(e))

    # 2) Modern encyclopedic Russian. Optional but strongly preferred; robust fallback
    # leaves the run usable if the remote Wikipedia configuration changes.
    try:
        wiki_train=[]
        ds=stream_dataset('wikimedia/wikipedia','20231101.ru',seed=20260810)
        tr_lim=a.wiki_train_mb*MB;te_lim=a.heldout_mb*MB
        seen=0
        for r in ds:
            text=r.get('text') or ''
            if len(text)<300:continue
            key=(str(r.get('id',''))+'\x1f'+str(r.get('title',''))).encode('utf-8','ignore')
            bucket=hashlib.sha256(key).digest()[0]%10
            if bucket<8:add_until(wiki_train,text,tr_lim)
            else:add_until(wiki_test,text,te_lim)
            seen+=1
            if sum(map(len,wiki_train))>=tr_lim and sum(map(len,wiki_test))>=te_lim:break
        train.extend(wiki_train)
        meta['sources']['WikipediaRU']={'records_seen':seen,'train_bytes':sum(map(len,wiki_train)),'test_bytes':sum(map(len,wiki_test))}
    except Exception as e:
        meta['errors'].append('WikipediaRU: '+repr(e))

    # 3) Modern syntactically curated sentences from SynTagRus.
    synt=cap_utf8(conllu_text(a.synt_train),a.synt_train_mb*MB)
    train.append(synt)
    meta['sources']['SynTagRus_train']={'train_bytes':len(synt)}

    # Interleave fixed blocks from all collected content instead of leaving source domains
    # in one large contiguous region.
    raw=b''.join(train)
    block=65536
    chunks=[raw[i:i+block] for i in range(0,len(raw),block)]
    # Deterministic multiplicative permutation: avoids language/source runs without randomness.
    if chunks:
        order=sorted(range(len(chunks)),key=lambda i:((i+1)*2654435761)&0xffffffff)
        raw=b'\n'.join(chunks[i] for i in order)
    (o/'ru_train.txt').write_bytes(raw)
    (o/'ru_heritage_test.txt').write_bytes(b''.join(heritage_test))
    (o/'ru_wikipedia_test.txt').write_bytes(b''.join(wiki_test))
    meta['ru_train_bytes']=len(raw)
    meta['ru_heritage_test_bytes']=sum(map(len,heritage_test))
    meta['ru_wikipedia_test_bytes']=sum(map(len,wiki_test))
    meta['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in o.glob('*.txt')}
    (o/'CORPUS_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    if len(raw)<8*MB:
        raise RuntimeError(f'Russian train corpus too small: {len(raw)} bytes')

if __name__=='__main__':main()
