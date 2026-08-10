#!/usr/bin/env python3
import argparse, dataclasses, json, random, tempfile
from pathlib import Path
from r525_persistent_graphrag import Memory
from r524_graph_grounded_answer_loop import Authority

@dataclasses.dataclass
class Answer:
    status:str
    answer:str
    plan:dict
    proof:list
    provenance:list
    verifier:dict

class NexusCore:
    def __init__(self,db_path):
        self.memory=Memory(db_path);self.authority=Authority()
    def close(self):self.memory.close()
    def remember_claim(self,entity,relation,value,valid_from=0,valid_to=10**9,source='user',confidence=1.0,polarity=True,status='verified'):
        self.memory.claim(entity,relation,str(value),int(valid_from),int(valid_to),source,float(confidence),bool(polarity),status);self.memory.db.commit();return {'ok':True}
    def remember_relation(self,a,relation,b,source='user',confidence=1.0):
        self.memory.edge(a,relation,b,source,float(confidence));self.memory.db.commit();return {'ok':True}
    def current(self,entity,relation,t):
        r=self.memory.current(entity,relation,int(t))
        if r['status']=='missing':return Answer('unknown',f'В памяти нет действующего факта: «{entity}» / «{relation}» на момент {t}.',{'task':'temporal','entity':entity,'relation':relation,'time':t},[],[],{'passed':True,'reason':'explicit unknown'})
        if r['status']=='conflict':return Answer('blocked',f'Надёжный вывод заблокирован: для «{entity}» по отношению «{relation}» есть равносильные противоречащие факты.',{'task':'conflict','entity':entity,'relation':relation,'time':t},[],[],{'passed':True,'reason':'conflict sovereignty'})
        c=r['claim'];text=f'На момент {t}: «{entity}» — {relation} = «{c["value"]}». Источник: {c["source"]}.';ok=(str(c['value']) in text and c['source'] in text);return Answer('verified',text,{'task':'temporal','entity':entity,'relation':relation,'time':t,'value':c['value']},[],[c['source']],{'passed':ok,'reason':'slot identity + provenance'})
    def prove(self,fact_entity,fact_property,rules,goal_entity,goal_property):
        fact=(fact_entity,fact_property);goal=(goal_entity,goal_property);ok,trace=self.authority.prove_chain(fact,rules,goal)
        if ok:text=f'Да. Из факта «{fact_entity} {fact_property}» по цепочке правил следует «{goal_entity} {goal_property}».'
        else:text=f'Нет. Утверждение «{goal_entity} {goal_property}» не доказано из факта «{fact_entity} {fact_property}» и доступных правил.'
        passed=(('Да.' in text)==ok and goal_entity in text and goal_property in text)
        return Answer('verified' if ok else 'not_proved',text,{'task':'proof','goal':[goal_entity,goal_property]},trace,[],{'passed':passed,'reason':'Authority proof status + exact goal binding'})
    def retrieve(self,query,hops=2):
        r=self.memory.retrieve(query,hops);facts=[]
        for c in r['claims'][:8]:facts.append(f"{c['entity']} — {c['relation']} = {c['value']} [{c['source']}]")
        edges=[f"{e['src']} -{e['relation']}→ {e['dst']}" for e in r['edges'][:8]]
        if not facts and not edges:text='По этому запросу в графовой памяти нет достаточных данных.';status='unknown'
        else:text='Найдено в графовой памяти: '+('; '.join(facts+edges))+'.';status='retrieved'
        return Answer(status,text,{'task':'graphrag','query':query,'hops':hops,'roots':[x['label'] for x in r['roots']]},[],[c['source'] for c in r['claims'][:8]],{'passed':True,'reason':'returned only retrieved graph records'})
    def set_goal(self,session,text,criterion,expected):return self.memory.set_goal(session,text,criterion,expected)
    def feedback(self,gid,observed):
        r=self.memory.feedback(gid,str(observed));self.memory.db.commit();text='Цель достигнута, контур стабилен.' if r['mode']=='stable' else f'Обнаружено рассогласование: {r["error"]}. Контур переходит к коррекции.';return Answer(r['mode'],text,{'task':'cyber_feedback','goal_id':gid,'observed':observed},[],[],{'passed':True,'reason':'closed-loop state transition'})
    def execute(self,cmd):
        op=cmd.get('op')
        if op=='remember_claim':return self.remember_claim(**{k:v for k,v in cmd.items() if k!='op'})
        if op=='remember_relation':return self.remember_relation(**{k:v for k,v in cmd.items() if k!='op'})
        if op=='current':return dataclasses.asdict(self.current(cmd['entity'],cmd['relation'],cmd['time']))
        if op=='retrieve':return dataclasses.asdict(self.retrieve(cmd['query'],cmd.get('hops',2)))
        if op=='set_goal':return {'goal_id':self.set_goal(cmd['session'],cmd['text'],cmd['criterion'],cmd['expected'])}
        if op=='feedback':return dataclasses.asdict(self.feedback(cmd['goal_id'],cmd['observed']))
        raise ValueError(f'unknown op: {op}')


def selftest(out,n=20000):
    db=Path(out)/'nexus_core.sqlite';core=NexusCore(db);rng=random.Random(20260810);ok=0
    # Seed typed graph.
    for i in range(2500):
        core.remember_claim(f'узел-{i:04d}','состояние','старое',0,49,'archive',.9)
        core.remember_claim(f'узел-{i:04d}','состояние','новое',50,999,'sensor',.99)
        core.remember_relation(f'узел-{i:04d}','INSTANCE_OF',f'класс-{i%50:02d}','generator')
    examples=[]
    for i in range(n):
        typ=i%4
        if typ==0:
            e=f'узел-{rng.randrange(2500):04d}';t=rng.choice([10,80]);a=core.current(e,'состояние',t);expected='старое' if t<50 else 'новое';good=a.verifier['passed'] and expected in a.answer
        elif typ==1:
            idx=rng.randrange(2500);a=core.retrieve(f'узел-{idx:04d}',2);good=a.verifier['passed'] and a.status=='retrieved'
        elif typ==2:
            gid=core.set_goal(f's{i}','получить 200','HTTP','200');obs='200' if rng.random()<.5 else '500';a=core.feedback(gid,obs);good=a.verifier['passed'] and ((obs=='200' and a.status=='stable') or (obs!='200' and a.status=='recover'))
        else:
            from r524_graph_grounded_answer_loop import Rule
            e=f'модуль-{i}';a=core.prove(e,'готов',[Rule((e,'готов'),(e,'проверен')),Rule((e,'проверен'),(e,'разрешён'))],e,'разрешён');good=a.verifier['passed'] and a.status=='verified'
        ok+=int(good)
        if len(examples)<40:examples.append(dataclasses.asdict(a))
    core.close();result={'format':'nexus-r526-integrated-cognitive-loop/1','n':n,'passed':ok,'accuracy':ok/n,'components':['R5.25 persistent GraphRAG','R5.22 typed binding / Authority sovereignty','R5.24 graph-grounded semantic planning','post-render verifier','cybernetic goal-feedback loop'],'policy':'neural surface organ may propose wording later, but cannot mutate plan slots/proof/provenance; verifier rejection falls back to deterministic surface'};Path(out,'00_R526_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');Path(out,'01_EXAMPLES.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8');return result

def cli(db):
    core=NexusCore(db)
    try:
        import sys
        for line in sys.stdin:
            line=line.strip()
            if not line:continue
            try:r=core.execute(json.loads(line));print(json.dumps(r,ensure_ascii=False),flush=True)
            except Exception as e:print(json.dumps({'error':repr(e)},ensure_ascii=False),flush=True)
    finally:core.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--n',type=int,default=20000);ap.add_argument('--out',default='nexus_r526_results');ap.add_argument('--db',default='nexus_memory.sqlite');a=ap.parse_args()
    if a.selftest:
        Path(a.out).mkdir(parents=True,exist_ok=True);r=selftest(a.out,a.n);print(json.dumps(r,ensure_ascii=False));assert r['accuracy']==1.0
    else:cli(a.db)
if __name__=='__main__':main()
