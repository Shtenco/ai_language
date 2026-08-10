#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path

MIN_DOC_BYTES = 200

def clean_text(s: str) -> str:
    s=s.replace('\r','\n').replace('\x00',' ');s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{4,}','\n\n',s);return s.strip()
def cap_utf8_text(s,max_bytes):
    b=clean_text(s).encode('utf-8')
    if len(b)<=max_bytes:return b.decode('utf-8')
    b=b[:max_bytes]
    while b:
        try:return b.decode('utf-8')
        except UnicodeDecodeError:b=b[:-1]
    return ''
def sha_bucket(key,n=100):return int.from_bytes(hashlib.sha256(key.encode('utf-8','ignore')).digest()[:8],'big')%n
def nbytes(docs):return sum(x['bytes'] for x in docs)
def saturated(docs,limit):return nbytes(docs)>=max(0,limit-(MIN_DOC_BYTES-1))
def conllu_docs(path,sentences_per_doc=32):
    batch=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('# text = '):
            t=clean_text(line[9:])
            if t:
                batch.append(t)
                if len(batch)>=sentences_per_doc:yield '\n'.join(batch);batch=[]
    if batch:yield '\n'.join(batch)
def add_doc(dst,source,doc_id,text,limit,max_doc_bytes=24000):
    used=nbytes(dst)
    if used>=limit:return False
    text=cap_utf8_text(text,max_doc_bytes)
    if len(text)<MIN_DOC_BYTES:return False
    raw=(text+'\n').encode('utf-8');left=limit-used
    if len(raw)>left:text=cap_utf8_text(text,max(0,left-1));raw=(text+'\n').encode('utf-8')
    if len(raw)<MIN_DOC_BYTES:return False
    dst.append({'source':source,'id':str(doc_id),'text':text,'bytes':len(raw)});return True
def write_jsonl(path,docs):
    with Path(path).open('w',encoding='utf-8') as f:
        for d in docs:f.write(json.dumps({k:d[k] for k in ('source','id','text')},ensure_ascii=False)+'\n')
def write_text(path,docs):Path(path).write_text('\n\n'.join(d['text'] for d in docs)+'\n',encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--synt-train',required=True);ap.add_argument('--out',required=True);ap.add_argument('--wiki-train-mb',type=int,default=20);ap.add_argument('--wiki-memory-mb',type=int,default=5);ap.add_argument('--wiki-test-mb',type=int,default=3);ap.add_argument('--heritage-train-mb',type=int,default=6);ap.add_argument('--heritage-test-mb',type=int,default=2);ap.add_argument('--synt-train-mb',type=int,default=4);ap.add_argument('--max-doc-kb',type=int,default=24);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);MB=1024*1024;max_doc=a.max_doc_kb*1024;train=[];memory=[];wiki_test=[];heritage_test=[];meta={'format':'nexus-r57-russian-corpus/2','utf8_safe_document_shuffle':True,'saturation_slack_bytes':199,'max_doc_bytes':max_doc,'sources':{},'errors':[]}
    try:
        from datasets import load_dataset
        ds=load_dataset('wikimedia/wikipedia','20231101.ru',split='train',streaming=True)
        try:ds=ds.shuffle(seed=20260810,buffer_size=5000)
        except Exception:pass
        lt=a.wiki_train_mb*MB;lm=a.wiki_memory_mb*MB;le=a.wiki_test_mb*MB;seen=0
        for r in ds:
            text=r.get('text') or '';title=str(r.get('title') or '');rid=str(r.get('id') or title)
            if len(text)<500:continue
            b=sha_bucket(rid+'\x1f'+title)
            if b<10:add_doc(wiki_test,'wikimedia/wikipedia:20231101.ru',rid,text,le,max_doc)
            elif b<25:add_doc(memory,'wikimedia/wikipedia:20231101.ru',rid,text,lm,max_doc)
            else:add_doc(train,'wikimedia/wikipedia:20231101.ru',rid,text,lt,max_doc)
            seen+=1
            if saturated(train,lt) and saturated(memory,lm) and saturated(wiki_test,le):break
        meta['sources']['Wikipedia_ru']={'records_seen':seen,'train_bytes':sum(d['bytes'] for d in train if d['source'].startswith('wikimedia/')),'memory_bytes':nbytes(memory),'test_bytes':nbytes(wiki_test)}
    except Exception as e:meta['errors'].append('Wikipedia: '+repr(e))
    try:
        from datasets import load_dataset
        ds=load_dataset('maxzt/RuHeritage-Corpus',split='train',streaming=True)
        try:ds=ds.shuffle(seed=20260810,buffer_size=2000)
        except Exception:pass
        lt=a.heritage_train_mb*MB;le=a.heritage_test_mb*MB;start=nbytes(train);seen=0
        for r in ds:
            text=r.get('text') or ''
            if len(text)<500:continue
            key=str(r.get('author',''))+'\x1f'+str(r.get('title',''));b=sha_bucket(key)
            if b<20:add_doc(heritage_test,'maxzt/RuHeritage-Corpus',key,text,le,max_doc)
            elif nbytes(train)-start<lt:add_doc(train,'maxzt/RuHeritage-Corpus',key,text,start+lt,max_doc)
            seen+=1
            if nbytes(train)-start>=lt-199 and saturated(heritage_test,le):break
        meta['sources']['RuHeritage']={'records_seen':seen,'train_bytes':sum(d['bytes'] for d in train if d['source'].startswith('maxzt/')),'test_bytes':nbytes(heritage_test)}
    except Exception as e:meta['errors'].append('RuHeritage: '+repr(e))
    sd=[];sl=a.synt_train_mb*MB
    for i,text in enumerate(conllu_docs(a.synt_train)):
        add_doc(sd,'UD_Russian-SynTagRus',i,text,sl,max_doc)
        if saturated(sd,sl):break
    train.extend(sd);meta['sources']['SynTagRus_train']={'train_docs':len(sd),'train_bytes':nbytes(sd)}
    train.sort(key=lambda d:hashlib.sha256((d['source']+'\x1f'+d['id']).encode()).digest());memory.sort(key=lambda d:hashlib.sha256((d['source']+'\x1f'+d['id']).encode()).digest());wiki_test.sort(key=lambda d:d['id']);heritage_test.sort(key=lambda d:d['id'])
    write_jsonl(out/'ru_train_docs.jsonl',train);write_jsonl(out/'graph_memory_docs.jsonl',memory);write_jsonl(out/'ru_wiki_test_docs.jsonl',wiki_test);write_jsonl(out/'ru_heritage_test_docs.jsonl',heritage_test);write_text(out/'ru_train.txt',train);write_text(out/'graph_memory.txt',memory);write_text(out/'ru_wiki_test.txt',wiki_test);write_text(out/'ru_heritage_test.txt',heritage_test)
    meta['totals']={'train_docs':len(train),'train_bytes':nbytes(train),'memory_docs':len(memory),'memory_bytes':nbytes(memory),'wiki_test_docs':len(wiki_test),'wiki_test_bytes':nbytes(wiki_test),'heritage_test_docs':len(heritage_test),'heritage_test_bytes':nbytes(heritage_test)};meta['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*')) if p.is_file()};(out/'CORPUS_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2));
    if meta['totals']['train_bytes']<12*MB:raise RuntimeError('train corpus too small')
    if meta['totals']['memory_bytes']<1*MB:raise RuntimeError('graph memory too small')
if __name__=='__main__':main()
