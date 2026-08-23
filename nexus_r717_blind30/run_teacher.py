import json, os, re, time, urllib.request, hashlib, base64, zlib
from pathlib import Path
MODEL=os.environ['MODEL']; MODEL_ID=os.environ['MODEL_ID']; OUT=Path(os.environ['OUT'])
BASE='http://127.0.0.1:11434'
EXPECTED='afeca41d3313bde64673c99fbf059554a524ab946d21f85269ef4f0d7b5e1f6e'
here=Path(__file__).parent
packed=''.join((here/f'b30_chunk_{i}.txt').read_text(encoding='utf-8') for i in range(4))
raw=zlib.decompress(base64.b85decode(packed.encode('ascii')))
bpath=here/'BLIND30_FROZEN.json'; bpath.write_bytes(raw); bsha=hashlib.sha256(raw).hexdigest()
if bsha!=EXPECTED: raise SystemExit(f'BLIND30 HASH MISMATCH {bsha} != {EXPECTED}')
bench=json.loads(raw.decode('utf-8'))

def post(path,obj,timeout=900):
    data=json.dumps(obj).encode(); req=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def pick(text):
    s=(text or '').strip().upper(); m=re.search(r'(?:^|[^A-ZА-Я])([ABCD])(?:$|[^A-ZА-Я])',s)
    if m:return m.group(1)
    return s[0] if s and s[0] in 'ABCD' else None
def option_lp(resp):
    best={x:None for x in 'ABCD'}
    for row in (resp.get('logprobs') or [])[:12]:
        for c in [row]+list(row.get('top_logprobs') or []):
            tok=(c.get('token') or '').strip().upper()
            if tok in best and best[tok] is None:
                try:best[tok]=float(c.get('logprob'))
                except:pass
        if any(v is not None for v in best.values()):break
    return best
try:show=post('/api/show',{'model':MODEL},120)
except Exception as e:show={'error':repr(e)}
system='Ты проходишь закрытый benchmark. Выбери ровно один вариант. Не объясняй решение. Выведи только латинскую букву A, B, C или D.'
rows=[]
for it in bench['items']:
    prompt=it['q']+'\n'+'\n'.join(f'{k}) {v}' for k,v in it['options'].items())+'\nОтвет:'
    body={'model':MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':prompt}],
          'stream':False,'think':False,'logprobs':True,'top_logprobs':20,
          'options':{'temperature':0,'seed':301717,'num_predict':8,'top_p':1.0}}
    t=time.time()
    try:r=post('/api/chat',body); raw_answer=((r.get('message') or {}).get('content') or ''); pred=pick(raw_answer); err=None
    except Exception as e:r={}; raw_answer=''; pred=None; err=repr(e)
    rows.append({'id':it['id'],'blind_index':it['blind_index'],'domain':it['domain'],'category':it['category'],'gold':it['answer'],'pred':pred,'correct':pred==it['answer'],
                 'raw':raw_answer,'option_logprobs':option_lp(r),'elapsed_wall_s':time.time()-t,'error':err,
                 'ollama_metrics':{k:r.get(k) for k in ['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']}})
    print(MODEL,it['id'],pred,it['answer'],rows[-1]['correct'],flush=True)
valid=[x for x in rows if isinstance(x['pred'],str) and x['pred'] in 'ABCD']
summary={'model':MODEL,'model_id':MODEL_ID,'blind30_sha256':bsha,'n':len(rows),'valid':len(valid),
         'accuracy':sum(x['correct'] for x in rows)/len(rows),
         'valid_accuracy':sum(x['correct'] for x in valid)/len(valid) if valid else None,'by_category':{}}
for d in sorted(set(x['category'] for x in rows)):
    z=[x for x in rows if x['category']==d]; summary['by_category'][d]={'n':len(z),'accuracy':sum(x['correct'] for x in z)/len(z)}
OUT.write_text(json.dumps({'summary':summary,'model_show':show,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
