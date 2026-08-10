#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path

PUNCT_RIGHT=re.compile(r'\s+([,.;:!?%»\)\]])')
PUNCT_LEFT=re.compile(r'([«\(\[])\s+')


def clean(s):
    s=s.replace('\r',' ').replace('\x00',' ')
    s=re.sub(r'\s+',' ',s).strip()
    s=PUNCT_RIGHT.sub(r'\1',s);s=PUNCT_LEFT.sub(r'\1',s)
    return s


def sha_bucket(key,n=100):return int.from_bytes(hashlib.sha256(key.encode('utf-8','ignore')).digest()[:8],'big')%n

def conllu_sentences(path):
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('# text = '):
            t=clean(line[9:])
            if len(t)>=20:yield t


def gicrya_sentences(root):
    files=sorted(p for p in Path(root).rglob('*') if p.is_file())
    for p in files:
        try:lines=p.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            try:lines=p.read_text(encoding='cp1251').splitlines()
            except Exception:continue
        toks=[]
        for line in lines+['']:
            line=line.strip()
            if not line:
                if toks:
                    s=clean(' '.join(toks));toks=[]
                    if len(s)>=20:yield s
                continue
            if line.startswith('#') or line.startswith('=='):continue
            parts=line.split('\t')
            if len(parts)<2:parts=re.split(r'\s+',line,maxsplit=4)
            if len(parts)>=2:
                tok=parts[1].strip()
                if tok and tok!='_':toks.append(tok)


def group_docs(sentences,source,prefix,per=28):
    out=[];buf=[];idx=0
    for s in sentences:
        buf.append(s)
        if len(buf)>=per:
            out.append({'source':source,'id':f'{prefix}-{idx}','text':' '.join(buf)});idx+=1;buf=[]
    if buf:out.append({'source':source,'id':f'{prefix}-{idx}','text':' '.join(buf)})
    return out


def utf8_bytes(d):return len((d['text']+'\n').encode('utf-8'))

def write_jsonl(path,docs):
    with Path(path).open('w',encoding='utf-8') as f:
        for d in docs:f.write(json.dumps(d,ensure_ascii=False)+'\n')

def write_text(path,docs):Path(path).write_text('\n\n'.join(d['text'] for d in docs)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gicrya-root',required=True);ap.add_argument('--ud',nargs='*',default=[]);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    train=[];memory=[];test=[]
    gd=group_docs(gicrya_sentences(a.gicrya_root),'morphoRuEval/GIKRYA','gic',per=28)
    for d in gd:
        b=sha_bucket(d['id'])
        if b<10:test.append(d)
        elif b<20:memory.append(d)
        else:train.append(d)
    for p in a.ud:
        name=Path(p).name
        train.extend(group_docs(conllu_sentences(p),'UniversalDependencies',name,per=28))
    train.sort(key=lambda d:hashlib.sha256((d['source']+'\x1f'+d['id']).encode()).digest())
    memory.sort(key=lambda d:d['id']);test.sort(key=lambda d:d['id'])
    write_jsonl(out/'ru_train_docs.jsonl',train);write_jsonl(out/'graph_memory_docs.jsonl',memory);write_jsonl(out/'gicrya_test_docs.jsonl',test)
    write_text(out/'ru_train.txt',train);write_text(out/'graph_memory.txt',memory);write_text(out/'gicrya_test.txt',test)
    meta={'format':'nexus-r519-direct-modern/1','sources':['morphoRuEval/GIKRYA','UD_Russian-SynTagRus','UD_Russian-GSD','UD_Russian-Taiga'],'document_split':True,'hf_streaming':False,'train_docs':len(train),'train_bytes':sum(utf8_bytes(d) for d in train),'memory_docs':len(memory),'memory_bytes':sum(utf8_bytes(d) for d in memory),'gicrya_test_docs':len(test),'gicrya_test_bytes':sum(utf8_bytes(d) for d in test)}
    meta['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'CORPUS_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
    if meta['train_bytes']<8*1024*1024:raise RuntimeError('direct modern train corpus unexpectedly small')
    if meta['memory_bytes']<256*1024:raise RuntimeError('direct graph memory unexpectedly small')

if __name__=='__main__':main()
