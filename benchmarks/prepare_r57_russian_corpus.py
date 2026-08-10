#!/usr/bin/env python3
import argparse,hashlib,json,re
from pathlib import Path
MIN_DOC_BYTES=200
def clean_text(s):s=s.replace('\r','\n').replace('\x00',' ');s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{4,}','\n\n',s);return s.strip()
def cap_utf8_text(s,n):
 b=clean_text(s).encode('utf-8');
 if len(b)<=n:return b.decode('utf-8')
 b=b[:n]
 while b:
  try:return b.decode('utf-8')
  except UnicodeDecodeError:b=b[:-1]
 return ''
def sha_bucket(k,n=100):return int.from_bytes(hashlib.sha256(k.encode('utf-8','ignore')).digest()[:8],'big')%n
def nbytes(ds):return sum(x['bytes'] for x in ds)
def saturated(ds,lim):return nbytes(ds)>=max(0,lim-199)
def conllu_docs(path,per=32):
 b=[]
 for l in Path(path).read_text(encoding='utf-8').splitlines():
  if l.startswith('# text = '):
   t=clean_text(l[9:])
   if t:
    b.append(t)
    if len(b)>=per:yield '\n'.join(b);b=[]
 if b:yield '\n'.join(b)
def add_doc(dst,source,i,text,lim,maxdoc=24000):
 used=nbytes(dst)
 if used>=lim:return False
 text=cap_utf8_text(text,maxdoc)
 if len(text)<200:return False
 raw=(text+'\n').encode();left=lim-used
 if len(raw)>left:text=cap_utf8_text(text,max(0,left-1));raw=(text+'\n').encode()
 if len(raw)<200:return False
 dst.append({'source':source,'id':str(i),'text':text,'bytes':len(raw)});return True
def wj(p,ds):
 with Path(p).open('w',encoding='utf-8') as f:
  for d in ds:f.write(json.dumps({k:d[k] for k in ('source','id','text')},ensure_ascii=False)+'\n')
def wt(p,ds):Path(p).write_text('\n\n'.join(d['text'] for d in ds)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--synt-train',required=True);ap.add_argument('--out',required=True);ap.add_argument('--wiki-train-mb',type=int,default=20);ap.add_argument('--wiki-memory-mb',type=int,default=5);ap.add_argument('--wiki-test-mb',type=int,default=3);ap.add_argument('--heritage-train-mb',type=int,default=6);ap.add_argument('--heritage-test-mb',type=int,default=2);ap.add_argument('--synt-train-mb',type=int,default=4);ap.add_argument('--max-doc-kb',type=int,default=24);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);MB=1048576;md=a.max_doc_kb*1024;tr=[];mem=[];wte=[];hte=[];meta={'format':'nexus-r57-russian-corpus/2','utf8_safe_document_shuffle':True,'saturation_slack_bytes':199,'sources':{},'errors':[]}
 try:
  from datasets import load_dataset
  ds=load_dataset('wikimedia/wikipedia','20231101.ru',split='train',streaming=True)
  try:ds=ds.shuffle(seed=20260810,buffer_size=5000)
  except:pass
  lt=a.wiki_train_mb*MB;lm=a.wiki_memory_mb*MB;le=a.wiki_test_mb*MB;seen=0
  for r in ds:
   text=r.get('text') or '';title=str(r.get('title') or '');rid=str(r.get('id') or title)
   if len(text)<500:continue
   b=sha_bucket(rid+'\x1f'+title)
   if b<10:add_doc(wte,'wikimedia/wikipedia:20231101.ru',rid,text,le,md)
   elif b<25:add_doc(mem,'wikimedia/wikipedia:20231101.ru',rid,text,lm,md)
   else:add_doc(tr,'wikimedia/wikipedia:20231101.ru',rid,text,lt,md)
   seen+=1
   if saturated(tr,lt) and saturated(mem,lm) and saturated(wte,le):break
  meta['sources']['Wikipedia_ru']={'records_seen':seen,'train_bytes':sum(d['bytes'] for d in tr),'memory_bytes':nbytes(mem),'test_bytes':nbytes(wte)}
 except Exception as e:meta['errors'].append('Wikipedia: '+repr(e))
 try:
  from datasets import load_dataset
  ds=load_dataset('maxzt/RuHeritage-Corpus',split='train',streaming=True)
  try:ds=ds.shuffle(seed=20260810,buffer_size=2000)
  except:pass
  lt=a.heritage_train_mb*MB;le=a.heritage_test_mb*MB;start=nbytes(tr);seen=0
  for r in ds:
   text=r.get('text') or ''
   if len(text)<500:continue
   key=str(r.get('author',''))+'\x1f'+str(r.get('title',''));b=sha_bucket(key)
   if b<20:add_doc(hte,'maxzt/RuHeritage-Corpus',key,text,le,md)
   elif nbytes(tr)-start<lt:add_doc(tr,'maxzt/RuHeritage-Corpus',key,text,start+lt,md)
   seen+=1
   if nbytes(tr)-start>=lt-199 and saturated(hte,le):break
  meta['sources']['RuHeritage']={'records_seen':seen,'train_bytes':sum(d['bytes'] for d in tr if d['source'].startswith('maxzt/')),'test_bytes':nbytes(hte)}
 except Exception as e:meta['errors'].append('RuHeritage: '+repr(e))
 sd=[];sl=a.synt_train_mb*MB
 for i,t in enumerate(conllu_docs(a.synt_train)):
  add_doc(sd,'UD_Russian-SynTagRus',i,t,sl,md)
  if saturated(sd,sl):break
 tr.extend(sd);meta['sources']['SynTagRus_train']={'train_docs':len(sd),'train_bytes':nbytes(sd)}
 tr.sort(key=lambda d:hashlib.sha256((d['source']+'\x1f'+d['id']).encode()).digest());mem.sort(key=lambda d:hashlib.sha256((d['source']+'\x1f'+d['id']).encode()).digest());wte.sort(key=lambda d:d['id']);hte.sort(key=lambda d:d['id'])
 for n,ds in [('ru_train',tr),('graph_memory',mem),('ru_wiki_test',wte),('ru_heritage_test',hte)]:wj(out/f'{n}_docs.jsonl',ds);wt(out/f'{n}.txt',ds)
 meta['totals']={'train_docs':len(tr),'train_bytes':nbytes(tr),'memory_docs':len(mem),'memory_bytes':nbytes(mem),'wiki_test_docs':len(wte),'wiki_test_bytes':nbytes(wte),'heritage_test_docs':len(hte),'heritage_test_bytes':nbytes(hte)};(out/'CORPUS_META.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
 if nbytes(tr)<12*MB or nbytes(mem)<MB:raise RuntimeError('corpus too small')
if __name__=='__main__':main()
