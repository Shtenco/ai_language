#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, hashlib, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT17=Path('/mnt/data/blind30_work/r717/NEXUS_R717_ADAPTIVE_RECURRENT_REASONING_FREEZE_2026-08-23')
sys.path.insert(0,str(ROOT17/'source'))
import runtime_r717 as r17
r16=r17.r716
sf=r16.sf; qc=r16.qc; bip=r16.bip; g=r16.g
L=g.LABELS; T=g.TEACHERS
EXPECTED_BLIND='afeca41d3313bde64673c99fbf059554a524ab946d21f85269ef4f0d7b5e1f6e'
EXPECTED_R717='2b86e4beb2a23b691914c638185db4bad8f66cb966baa5fcb0a77ba7dfcad5c0'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bench(path):
    if sha(path)!=EXPECTED_BLIND: raise SystemExit('Blind30 SHA mismatch')
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_evidence(folder):
    folder=Path(folder); by_teacher={}
    for t in T:
        p=folder/f'blind30_{t}.json'
        if not p.exists(): raise FileNotFoundError(p)
        obj=json.loads(p.read_text(encoding='utf-8'))
        if obj['summary']['blind30_sha256']!=EXPECTED_BLIND: raise RuntimeError(f'{t}: evidence blind hash mismatch')
        by_teacher[t]={row['id']:row for row in obj['rows']}
    return by_teacher

def rows_for(item, evidence, names=None):
    names=T if names is None else names
    return {t:evidence[t][item['id']] for t in names}

def metrics(raw): return r16._metrics(raw)

def active_decide(item, all_rows, advice='one'):
    q=item['q']; formal=g.formal_pass(item)
    ls=r16.r714.language_state(q); cs=r16.r715.compression_state(item)
    initial=bip.route_pack(q,0,None,'INIT',0.0)
    if formal['status']=='VERIFIED' and formal.get('answer') in g.L2I:
        return {'answer':formal['answer'],'stage':'FORMAL_VERIFIED','calls':0,'used':[],'formal':formal,'query_chain':[],'override_events':[]}
    mask,top2,comp,route=sf.initial_route(q); top2names=[T[i] for i in top2]
    revealed={t:all_rows[t] for t in top2names}; used=top2names.copy(); calls=2
    current=0; previous_raw=None; stable=0; trace=[]; overrides=[]
    for cycle in range(qc.GATE['max_extra_calls']+1):
        if cycle==0:
            df=sf.record(item,revealed,formal); X,B,aux=sf.called_features(df,mask); current=int(np.argmax(B[0])); control=np.array([.75,.75,0,0,0],float); mode='INIT'
        else:
            control=qc.state_delta(previous_raw,None,stable); mode='VERIFY' if stable else 'DISCRIMINATE'
        pack=qc.make_query_pack(bip,q,cycle,current,mask,control,mode)
        raw,B,aux=sf.evaluate(item,revealed,formal,mask,cycle,current,control,mode,pack)
        old=current; sug=int(np.argmax(raw)); p,gap,ent=metrics(raw)
        if cycle==0: current=sug
        elif sug==old: stable+=1
        else:
            adv=float(raw[sug]-raw[old])
            if adv>=.18: current=sug; stable=0
            else: stable+=1
        trace.append({'cycle':cycle,'called':[T[j] for j in np.where(mask>0)[0]],'hypothesis_before':L[old],'suggested':L[sug],'hypothesis_after':L[current],'confidence':float(p[current]),'entropy':float(ent),'margin':float(gap)})
        previous_raw=raw.copy()
        if cycle==0 and not qc.in_query_gain_window(float(p[current]),ent):
            return {'answer':L[current],'stage':'SEQUENTIAL_TOP2_FIXED_POINT','calls':calls,'used':used,'formal':formal,'query_chain':trace,'override_events':overrides}
        if cycle>0 and ((p[current]>=.72 and gap>=.22 and ent<=.62) or stable>=2 or cycle==qc.GATE['max_extra_calls'] or np.sum(mask)>=5):
            return {'answer':L[current],'stage':'ACTIVE_QUERY_FIXED_POINT','calls':calls,'used':used,'formal':formal,'query_chain':trace,'override_events':overrides}
        control2=qc.state_delta(previous_raw,None,stable); qmode=qc.query_mode(stable); qpack=qc.make_query_pack(bip,q,cycle+1,current,mask,control2,qmode)
        frozen_j=qc.choose_next_teacher(q,cycle+1,current,mask,aux['comp'],sf.INDEP,qpack,True)
        if frozen_j is None:
            return {'answer':L[current],'stage':'NO_MORE_ORGANS','calls':calls,'used':used,'formal':formal,'query_chain':trace,'override_events':overrides}
        frozen_t=T[frozen_j]
        meta=r17.recurrent_state(item,revealed,used,formal)
        key='suggested_teacher_two_step' if advice=='two' else 'suggested_teacher_one_step'
        suggested=meta.get(key)
        chosen=frozen_t; applied=False
        if suggested in T and suggested not in used:
            chosen=suggested; applied=(chosen!=frozen_t)
        chosen_j=T.index(chosen)
        overrides.append({'cycle':cycle+1,'r716_teacher':frozen_t,'r717_suggested':suggested,'chosen':chosen,'override_applied':applied,'feedback_digest':meta.get('feedback_digest'),'risk':meta.get('risk_current_hypothesis')})
        mask[chosen_j]=1; revealed[chosen]=all_rows[chosen]; calls+=1; used.append(chosen)
    return {'answer':L[current],'stage':'MAX_QUERY_CYCLES','calls':calls,'used':used,'formal':formal,'query_chain':trace,'override_events':overrides}

def binom_two_sided(k,n):
    if n==0:return 1.0
    pk=math.comb(n,k)/(2**n)
    return min(1.0,sum(math.comb(n,i)/(2**n) for i in range(n+1) if math.comb(n,i)/(2**n)<=pk+1e-15))

def summarize(name, results, items):
    n=len(items); correct=[r['answer']==it['answer'] for r,it in zip(results,items)]
    by=defaultdict(list)
    for c,it in zip(correct,items): by[it['category']].append(c)
    stages=Counter(r.get('stage') for r in results); calls=[r.get('calls',0) for r in results]
    formal=[r for r in results if r.get('stage')=='FORMAL_VERIFIED']
    return {'name':name,'n':n,'correct':sum(correct),'accuracy':sum(correct)/n,'mean_calls':float(np.mean(calls)),'by_category':{k:{'n':len(v),'correct':sum(v),'accuracy':sum(v)/len(v)} for k,v in sorted(by.items())},'stage_counts':dict(stages),'formal_coverage':len(formal)/n}

def compare(base, alt, items):
    corr=reg=0; changed=[]
    for b,a,it in zip(base,alt,items):
        bc=b['answer']==it['answer']; ac=a['answer']==it['answer']
        if (not bc) and ac: corr+=1
        if bc and (not ac): reg+=1
        if b['answer']!=a['answer']:
            changed.append({'id':it['id'],'category':it['category'],'gold':it['answer'],'base':b['answer'],'alt':a['answer'],'base_correct':bc,'alt_correct':ac})
    return {'corrections':corr,'regressions':reg,'net':corr-reg,'discordant':corr+reg,'paired_exact_p':binom_two_sided(min(corr,reg),corr+reg),'answer_changes':changed}

def teacher_stats(items,evidence):
    out={}; oracle=0
    for t in T:
        c=sum(evidence[t][it['id']]['pred']==it['answer'] for it in items); out[t]={'correct':c,'accuracy':c/len(items)}
    for it in items: oracle += any(evidence[t][it['id']]['pred']==it['answer'] for t in T)
    return {'single':out,'five_teacher_oracle':{'correct':oracle,'accuracy':oracle/len(items)}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bench',default='/mnt/data/blind30_work/BLIND30_FROZEN.json'); ap.add_argument('--evidence',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    bench=load_bench(args.bench); items=bench['items']; evidence=load_evidence(args.evidence)
    base=[]; advisory=[]; one=[]; two=[]
    for idx,it in enumerate(items):
        allr=rows_for(it,evidence)
        b=r16.decide(it,allr); a=r17.decide(it,allr); o=active_decide(it,allr,'one'); q=active_decide(it,allr,'two')
        base.append(b); advisory.append(a); one.append(o); two.append(q)
        print(idx,it['id'],it['answer'],b.get('answer'),a.get('answer'),o.get('answer'),q.get('answer'),flush=True)
    report={'protocol':{'blind30_sha256':EXPECTED_BLIND,'r717_freeze_zip_sha256':EXPECTED_R717,'primary_active_rule':'substitute only next extra-teacher choice with frozen one-step recurrent suggestion if unused; fallback to frozen R7.16 BIP39 teacher','secondary_two_step':'same substitution using frozen two-step quantum-tree suggestion'},'teachers':teacher_stats(items,evidence),'summaries':[summarize('R7.16_frozen',base,items),summarize('R7.17_advisory',advisory,items),summarize('R7.17_active_one_step',one,items),summarize('R7.17_active_two_step',two,items)],'comparisons':{'advisory_vs_r716':compare(base,advisory,items),'active_one_vs_r716':compare(base,one,items),'active_two_vs_r716':compare(base,two,items)},'per_item':[]}
    for it,b,a,o,q in zip(items,base,advisory,one,two):
        report['per_item'].append({'id':it['id'],'category':it['category'],'gold':it['answer'],'r716':b.get('answer'),'advisory':a.get('answer'),'active_one':o.get('answer'),'active_two':q.get('answer'),'calls_r716':b.get('calls',0),'calls_active_one':o.get('calls',0),'calls_active_two':q.get('calls',0),'override_events_one':o.get('override_events',[]),'override_events_two':q.get('override_events',[])})
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summaries':report['summaries'],'comparisons':{k:{x:v[x] for x in ['corrections','regressions','net','paired_exact_p']} for k,v in report['comparisons'].items()}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
