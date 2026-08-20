import json, os, re, time, urllib.request, hashlib
from pathlib import Path
from BLIND25_DATA import benchmark
MODEL=os.environ['MODEL']; MODEL_ID=os.environ['MODEL_ID']; OUT=Path(os.environ['OUT'])
BASE='http://127.0.0.1:11434'
EXPECTED='3b0fa89aca603eb5cc70c3aa2c80f1e9e9660de9238be0527bce7552e9cf0bd0'
bench=benchmark(); bpath=Path(__file__).with_name('BLIND25_FROZEN.json')
bpath.write_text(json.dumps(bench,ensure_ascii=False,indent=2),encoding='utf-8')
bsha=hashlib.sha256(bpath.read_bytes()).hexdigest()
if bsha!=EXPECTED: raise SystemExit(f'BLIND25 HASH MISMATCH {bsha} != {EXPECTED}')

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
          'options':{'temperature':0,'seed':25078,'num_predict':8,'top_p':1.0}}
    t=time.time()
    try:r=post('/api/chat',body);raw=((r.get('message') or {}).get('content') or '');pred=pick(raw);err=None
    except Exception as e:r={};raw='';pred=None;err=repr(e)
    rows.append({'id':it['id'],'domain':it['domain'],'gold':it['answer'],'pred':pred,'correct':pred==it['answer'],
                 'raw':raw,'option_logprobs':option_lp(r),'elapsed_wall_s':time.time()-t,'error':err,
                 'ollama_metrics':{k:r.get(k) for k in ['total_duration','load_duration','prompt_eval_count','prompt_eval_duration','eval_count','eval_duration']}})
    print(MODEL,it['id'],pred,it['answer'],rows[-1]['correct'],flush=True)
valid=[x for x in rows if isinstance(x['pred'],str) and x['pred'] in 'ABCD']
summary={'model':MODEL,'model_id':MODEL_ID,'blind25_sha256':bsha,'n':len(rows),'valid':len(valid),
         'accuracy':sum(x['correct'] for x in rows)/len(rows),
         'valid_accuracy':sum(x['correct'] for x in valid)/len(valid) if valid else None,'by_domain':{}}
for d in sorted(set(x['domain'] for x in rows)):
    z=[x for x in rows if x['domain']==d];summary['by_domain'][d]={'n':len(z),'accuracy':sum(x['correct'] for x in z)/len(z)}
OUT.write_text(json.dumps({'summary':summary,'model_show':show,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
