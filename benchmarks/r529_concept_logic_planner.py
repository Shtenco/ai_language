#!/usr/bin/env python3
import argparse, collections, json, random, time
from pathlib import Path
from r525_persistent_graphrag import Memory

SEED=20260810

class ConceptLogicPlanner:
    def __init__(self,memory): self.m=memory
    def _id(self,label):
        r=self.m.db.execute('SELECT id FROM node WHERE label=?',(label,)).fetchone();return r[0] if r else None
    def path(self,src_label,dst_label,relations,max_depth=8):
        src=self._id(src_label);dst=self._id(dst_label)
        if src is None or dst is None:return {'status':'missing','path':[]}
        q=collections.deque([(src,[])]);seen={src};rels=tuple(relations)
        while q:
            nid,path=q.popleft()
            if len(path)>=max_depth:continue
            marks=','.join('?'*len(rels))
            rows=self.m.db.execute(f'''SELECT e.relation,e.dst,b.label,e.source,e.confidence FROM edge e JOIN node b ON b.id=e.dst WHERE e.src=? AND e.relation IN ({marks}) ORDER BY e.id''',(nid,*rels)).fetchall()
            for rel,nxt,label,source,conf in rows:
                step={'src_id':nid,'relation':rel,'dst_id':nxt,'dst':label,'source':source,'confidence':conf};np=path+[step]
                if nxt==dst:return {'status':'proved','path':np}
                if nxt not in seen:seen.add(nxt);q.append((nxt,np))
        return {'status':'not_proved','path':[]}
    def is_a(self,entity,concept,max_depth=8):
        r=self.path(entity,concept,['INSTANCE_OF','IS_A'],max_depth)
        r['task']='concept_membership';r['entity']=entity;r['concept']=concept;return r
    def causal(self,cause,effect,max_depth=8):
        r=self.path(cause,effect,['CAUSES','ENABLES'],max_depth)
        r['task']='causal_path';r['cause']=cause;r['effect']=effect;return r
    def define(self,concept,limit=8):
        nid=self._id(concept)
        if nid is None:return {'task':'definition','status':'missing','concept':concept}
        row=self.m.db.execute('SELECT kind,content FROM node WHERE id=?',(nid,)).fetchone()
        parents=self.m.db.execute("SELECT b.label,e.relation,e.source FROM edge e JOIN node b ON b.id=e.dst WHERE e.src=? AND e.relation='IS_A' LIMIT ?",(nid,limit)).fetchall()
        instances=self.m.db.execute("SELECT a.label,e.source FROM edge e JOIN node a ON a.id=e.src WHERE e.dst=? AND e.relation='INSTANCE_OF' LIMIT ?",(nid,limit)).fetchall()
        children=self.m.db.execute("SELECT a.label,e.source FROM edge e JOIN node a ON a.id=e.src WHERE e.dst=? AND e.relation='IS_A' LIMIT ?",(nid,limit)).fetchall()
        return {'task':'definition','status':'resolved','concept':concept,'kind':row[0],'content':row[1],'parents':[x[0] for x in parents],'instances':[x[0] for x in instances],'subconcepts':[x[0] for x in children],'provenance':[x[-1] for x in parents+instances+children if x[-1]]}
    def compare(self,a,b,relation,t):
        ca=self.m.current(a,relation,t);cb=self.m.current(b,relation,t)
        if ca['status']!='resolved' or cb['status']!='resolved':return {'task':'compare','status':'blocked','a':ca,'b':cb}
        va=ca['claim']['value'];vb=cb['claim']['value']
        return {'task':'compare','status':'resolved','entity_a':a,'entity_b':b,'relation':relation,'time':t,'value_a':va,'value_b':vb,'equal':va==vb,'provenance':[ca['claim']['source'],cb['claim']['source']]}
    def contradiction(self,e,relation,t):
        r=self.m.current(e,relation,t);return {'task':'contradiction','status':r['status'],'entity':e,'relation':relation,'time':t,'conflict':r['conflict'],'claim':r['claim']}
    def plan_answer(self,op,**kw):
        if op=='is_a':r=self.is_a(kw['entity'],kw['concept'])
        elif op=='causal':r=self.causal(kw['cause'],kw['effect'])
        elif op=='define':r=self.define(kw['concept'])
        elif op=='compare':r=self.compare(kw['a'],kw['b'],kw['relation'],kw['time'])
        elif op=='contradiction':r=self.contradiction(kw['entity'],kw['relation'],kw['time'])
        else:raise ValueError(op)
        return self.surface_plan(r)
    def surface_plan(self,r):
        task=r['task'];status=r['status']
        slots={};steps=[]
        if task=='concept_membership':
            slots={'entity':r['entity'],'concept':r['concept']};steps=[f"{x['relation']}→{x['dst']}" for x in r['path']]
        elif task=='causal_path':
            slots={'cause':r['cause'],'effect':r['effect']};steps=[f"{x['relation']}→{x['dst']}" for x in r['path']]
        elif task=='definition':
            slots={'concept':r['concept'],'content':r.get('content',''),'parents':r.get('parents',[]),'instances':r.get('instances',[]),'subconcepts':r.get('subconcepts',[])}
        elif task=='compare':
            if status=='resolved':slots={k:r[k] for k in ('entity_a','entity_b','relation','time','value_a','value_b','equal')}
        elif task=='contradiction':slots={'entity':r['entity'],'relation':r['relation'],'time':r['time'],'conflict':r['conflict']}
        return {'task':task,'status':status,'slots':slots,'steps':steps,'provenance':r.get('provenance',[]),'authority_locked':True}


def build(db):
    m=Memory(db)
    # 10 roots × 5 middle concepts × 5 leaves; 5k instances.
    for root in range(10):
        R=f'класс-{root}';m.node(R,'CONCEPT',f'Общее понятие класса {root}.')
        for mid in range(5):
            M=f'класс-{root}-{mid}';m.node(M,'CONCEPT',f'Подкласс {mid} класса {root}.');m.edge(M,'IS_A',R,'taxonomy')
            for leaf in range(5):
                L=f'класс-{root}-{mid}-{leaf}';m.node(L,'CONCEPT',f'Конкретное понятие {leaf} подкласса {mid}.');m.edge(L,'IS_A',M,'taxonomy')
    for i in range(5000):
        e=f'объект-{i:04d}';root=i%10;mid=(i//10)%5;leaf=(i//50)%5;L=f'класс-{root}-{mid}-{leaf}';m.node(e,'ENTITY',f'Экземпляр {i}.');m.edge(e,'INSTANCE_OF',L,'registry');m.claim(e,'состояние',str(i%7),0,49,'archive',.9);m.claim(e,'состояние',str((i+1)%7),50,999,'sensor',.99)
    # Causal chains by groups of 20.
    for i in range(5000):
        n=f'событие-{i:04d}';m.node(n,'EVENT',f'Событие {i}.')
        if i%20:m.edge(f'событие-{i-1:04d}','CAUSES',n,'causal-log')
    m.claim('конфликт-X','режим','А',0,100,'A',.95);m.claim('конфликт-X','режим','Б',0,100,'B',.95)
    m.db.commit();return m


def selftest(out,n):
    db=Path(out)/'planner.sqlite';m=build(db);p=ConceptLogicPlanner(m);rng=random.Random(SEED);ok=0;lat=[];rows=[]
    for i in range(n):
        typ=i%5;t0=time.perf_counter_ns()
        if typ==0:
            idx=rng.randrange(5000);root=idx%10;plan=p.plan_answer('is_a',entity=f'объект-{idx:04d}',concept=f'класс-{root}');good=plan['status']=='proved' and len(plan['steps'])==3 and plan['slots']['entity']==f'объект-{idx:04d}'
        elif typ==1:
            group=rng.randrange(250);start=group*20+rng.randrange(15);dist=rng.randint(1,min(5,19-(start%20)));plan=p.plan_answer('causal',cause=f'событие-{start:04d}',effect=f'событие-{start+dist:04d}');good=plan['status']=='proved' and len(plan['steps'])==dist
        elif typ==2:
            a=rng.randrange(5000);b=rng.randrange(5000);tm=rng.choice([20,80]);plan=p.plan_answer('compare',a=f'объект-{a:04d}',b=f'объект-{b:04d}',relation='состояние',time=tm);ea=str(a%7 if tm<50 else (a+1)%7);eb=str(b%7 if tm<50 else (b+1)%7);good=plan['status']=='resolved' and plan['slots']['value_a']==ea and plan['slots']['value_b']==eb
        elif typ==3:
            root=rng.randrange(10);mid=rng.randrange(5);concept=f'класс-{root}-{mid}';plan=p.plan_answer('define',concept=concept);good=plan['status']=='resolved' and plan['slots']['concept']==concept and plan['slots']['content'] and f'класс-{root}' in plan['slots']['parents']
        else:
            plan=p.plan_answer('contradiction',entity='конфликт-X',relation='режим',time=50);good=plan['status']=='conflict' and plan['slots']['conflict'] is True
        lat.append((time.perf_counter_ns()-t0)/1e6);ok+=int(good)
        if len(rows)<80:rows.append({'type':typ,'plan':plan,'ok':good})
    m.close();lat.sort();res={'format':'nexus-r529-concept-logic-planner/1','n':n,'passed':ok,'accuracy':ok/n,'tasks':['concept membership via INSTANCE_OF/IS_A closure','causal path','temporal comparison','concept definition/content-extension','contradiction detection'],'planner_policy':'surface plan contains typed slots and provenance; authority_locked=true; language layer may verbalize but may not alter slots/steps','latency_ms':{'median':lat[len(lat)//2],'p95':lat[int(.95*len(lat))-1],'max':lat[-1]}};Path(out,'00_R529_RESULTS.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8');Path(out,'01_PLANS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(res,ensure_ascii=False));return res

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=50000);ap.add_argument('--out',default='nexus_r529_results');a=ap.parse_args();Path(a.out).mkdir(parents=True,exist_ok=True);r=selftest(a.out,a.n);assert r['accuracy']==1.0
if __name__=='__main__':main()
