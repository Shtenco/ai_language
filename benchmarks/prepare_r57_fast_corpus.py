#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


def clean(s):
    s=s.replace('\r','\n').replace('\x00',' ')
    s=re.sub(r'[ \t]+',' ',s)
    s=re.sub(r'\n{4,}','\n\n',s)
    return s.strip()


def cap(s,n=24576):
    b=clean(s).encode('utf-8')[:n]
    while b:
        try:return b.decode('utf-8')
        except UnicodeDecodeError:b=b[:-1]
    return ''


def bucket(key):
    return int.from_bytes(hashlib.sha256(key.encode('utf-8','ignore')).digest()[:8],'big')%100


def add(dst,source,key,text,limit):
    have=sum(x['bytes'] for x in dst)
    if have>=limit:return
    text=cap(text)
    if len(text)<200:return
    raw=(text+'\n').encode('utf-8')
    if len(raw)>limit-have:
        text=cap(text,max(0,limit-have-1));raw=(text+'\n').encode('utf-8')
    if len(raw)>=200:dst.append({'source':source,'id':key,'text':text,'bytes':len(raw)})


def conllu_docs(path,n=32):
    q=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('# text = '):
            t=line[9:].strip()
            if t:q.append(t)
            if len(q)>=n:
                yield '\n'.join(q);q=[]
    if q:yield '\n'.join(q)


def write_jsonl(p,docs):
    with Path(p).open('w',encoding='utf-8') as f:
        for d in docs:f.write(json.dumps({k:d[k] for k in ('source','id','text')},ensure_ascii=False)+'\n')


def write_text(p,docs):Path(p).write_text('\n\n'.join(x['text'] for x in docs)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--synt-train',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    o=Path(a.out);o.mkdir(parents=True,exist_ok=True);MB=1024*1024
    train=[];memory=[];test=[];meta={'format':'nexus-r57-fast-corpus/1','errors':[]}
    try:
        from datasets import load_dataset
        ds=load_dataset('maxzt/RuHeritage-Corpus',split='train',streaming=True)
        seen=0
        for r in ds:
            text=r.get('text') or ''
            if len(text)<500:continue
            key=str(r.get('author',''))+'\x1f'+str(r.get('title',''))
            b=bucket(key)
            if b<15:add(test,'maxzt/RuHeritage-Corpus',key,text,2*MB)
            elif b<35:add(memory,'maxzt/RuHeritage-Corpus',key,text,3*MB)
            else:add(train,'maxzt/RuHeritage-Corpus',key,text,12*MB)
            seen+=1
            if sum(x['bytes'] for x in train)>=12*MB and sum(x['bytes'] for x in memory)>=3*MB and sum(x['bytes'] for x in test)>=2*MB:break
        del ds
        meta['heritage_records_seen']=seen
    except Exception as e:meta['errors'].append('RuHeritage: '+repr(e))
    synt=[]
    for i,t in enumerate(conllu_docs(a.synt_train)):
        add(synt,'UD_Russian-SynTagRus',str(i),t,4*MB)
        if sum(x['bytes'] for x in synt)>=4*MB:break
    train.extend(synt)
    train.sort(key=lambda d:hashlib.sha256((d['source']+'\x1f'+d['id']).encode()).digest())
    memory.sort(key=lambda d:d['id']);test.sort(key=lambda d:d['id'])
    write_jsonl(o/'ru_train_docs.jsonl',train);write_jsonl(o/'graph_memory_docs.jsonl',memory);write_jsonl(o/'ru_heritage_test_docs.jsonl',test)
    write_text(o/'ru_train.txt',train);write_text(o/'graph_memory.txt',memory);write_text(o/'ru_heritage_test.txt',test)
    meta['totals']={'train_docs':len(train),'train_bytes':sum(x['bytes'] for x in train),'memory_docs':len(memory),'memory_bytes':sum(x['bytes'] for x in memory),'test_docs':len(test),'test_bytes':sum(x['bytes'] for x in test)}
    meta['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in o.glob('*') if p.is_file()}
    (o/'CORPUS_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    if meta['totals']['train_bytes']<8*MB:raise RuntimeError('train corpus too small')
    if meta['totals']['memory_bytes']<1*MB:raise RuntimeError('memory corpus too small')
if __name__=='__main__':main()
