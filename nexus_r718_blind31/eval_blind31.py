"""PRE-INFERENCE evaluator for frozen R7.18 Blind-31.
Scientific rule: this file is hashed/committed before teacher inference; do not edit after evidence exists.
Requires env R718_ROOT pointing to exact frozen R7.18 directory and EVIDENCE_DIR with five raw teacher JSONs.
"""
import os,sys,json,math,hashlib,pathlib,statistics
BLIND_SHA='49c587e0cf9f310df7720a7623169f3f24c6eae380f4778509af7b3bf7920dbb'
R718_SHA='12a5c190d2f6d3a88a938b2b064ff0e7341df6ec9c169a7462c9ab4ee904021d'
T=['ministral3_3b','phi4mini_3_8b','granite33_2b','qwen3_1_7b','gemma3_1b']
root=pathlib.Path(os.environ['R718_ROOT']);sys.path.insert(0,str(root/'source'));import runtime_r718 as r718
# frozen parent comparisons
sys.path.insert(0,str(root/'base_r717'/'base_r716'/'source'));import runtime_r716 as r716
sys.path.insert(0,str(root/'base_r717'/'source'));import runtime_r717 as r717
benchp=pathlib.Path(os.environ.get('BLIND31_JSON','BLIND31_FROZEN.json'));assert hashlib.sha256(benchp.read_bytes()).hexdigest()==BLIND_SHA;b=json.load(open(benchp))
evdir=pathlib.Path(os.environ['EVIDENCE_DIR']);ev={}
for t in T:
 rr=json.load(open(evdir/f'blind31_{t}.json'));assert rr['summary']['blind31_sha256']==BLIND_SHA;ev[t]={x['id']:x for x in rr['rows']}
def summarize(name,rows):
 n=len(rows);c=sum(x['correct'] for x in rows);by={}
 for d in sorted(set(x['category'] for x in rows)):
  z=[x for x in rows if x['category']==d];by[d]={'n':len(z),'correct':sum(x['correct'] for x in z),'accuracy':sum(x['correct'] for x in z)/len(z)}
 return {'name':name,'n':n,'correct':c,'accuracy':c/n,'mean_calls':sum(x['calls'] for x in rows)/n,'by_category':by}
def run(fn):
 rows=[]
 for it in b['items']:
  tr={t:ev[t][it['id']] for t in T};o=fn(it,tr);rows.append({'id':it['id'],'category':it['category'],'gold':it['answer'],'pred':o.get('answer'),'correct':o.get('answer')==it['answer'],'calls':o.get('calls',0),'stage':o.get('stage')})
 return rows
R16=run(r716.decide);R17=run(r717.decide);R18=run(r718.decide)
# references
sing={t:sum(ev[t][it['id']]['pred']==it['answer'] for it in b['items']) for t in T};oracle=sum(any(ev[t][it['id']]['pred']==it['answer'] for t in T) for it in b['items'])
# paired exact two-sided sign/binomial test
def pair(a,b):
 corr=sum((not x['correct']) and y['correct'] for x,y in zip(a,b));reg=sum(x['correct'] and (not y['correct']) for x,y in zip(a,b));n=corr+reg
 if n==0:p=1.0
 else:
  k=min(corr,reg);p=min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
 return {'corrections':corr,'regressions':reg,'discordant':n,'exact_two_sided_p':p}
out={'protocol':{'blind31_sha256':BLIND_SHA,'r718_freeze_sha256':R718_SHA},'teachers':{'single_correct':sing,'five_teacher_oracle':oracle},'summaries':[summarize('R7.16',R16),summarize('R7.17',R17),summarize('R7.18',R18)],'comparisons':{'r718_vs_r716':pair(R16,R18),'r718_vs_r717':pair(R17,R18)},'per_item':[{**x,'r716':R16[i]['pred'],'r717':R17[i]['pred']} for i,x in enumerate(R18)]}
pathlib.Path('BLIND31_CONFIRMATORY_REPORT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out['summaries'],ensure_ascii=False,indent=2));print(json.dumps(out['comparisons'],ensure_ascii=False,indent=2))
