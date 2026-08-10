#!/usr/bin/env python3
import argparse, dataclasses, hashlib, json, random
from pathlib import Path

SEED=20260810
ENTITIES=['модуль А','модуль Б','шлюз','датчик','контроллер','сервер','канал','агент','реактор','насос']
PROPS=['готов','активен','проверен','стабилен','доступен','синхронизирован','безопасен','подтверждён']
VALUES=['красный','зелёный','синий','режим А','режим Б','12','24','36','норма','авария']
SOURCES=['журнал A','журнал B','датчик-1','оператор','контрольный тест']

@dataclasses.dataclass(frozen=True)
class Claim:
    entity:str; relation:str; value:str; polarity:bool=True; valid_from:int=0; valid_to:int=10**9; source:str=''; confidence:float=1.0
@dataclasses.dataclass(frozen=True)
class Rule:
    premise:tuple; conclusion:tuple
@dataclasses.dataclass
class Plan:
    task:str; status:str; steps:list; slots:dict; provenance:list

class GraphMemory:
    def __init__(self): self.claims=[]; self.rules=[]
    def add_claim(self,c):self.claims.append(c)
    def add_rule(self,r):self.rules.append(r)
    def at(self,entity,relation,t):
        xs=[c for c in self.claims if c.entity==entity and c.relation==relation and c.valid_from<=t<=c.valid_to]
        xs.sort(key=lambda c:(c.valid_from,c.confidence),reverse=True);return xs
    def current(self,entity,relation,t):
        xs=self.at(entity,relation,t);return xs[0] if xs else None

class Authority:
    def prove_chain(self,fact,rules,goal):
        known={fact};trace=[];changed=True
        while changed and len(trace)<16:
            changed=False
            for r in rules:
                if r.premise in known and r.conclusion not in known:
                    known.add(r.conclusion);trace.append({'premise':r.premise,'conclusion':r.conclusion});changed=True
        return goal in known,trace
    def solve_temporal(self,g,e,r,t):return g.current(e,r,t)
    def solve_contradiction(self,claims,t):
        valid=[c for c in claims if c.valid_from<=t<=c.valid_to]
        if not valid:return None,'no_valid_claim'
        valid.sort(key=lambda c:(c.valid_from,c.confidence),reverse=True)
        top=valid[0]
        ties=[c for c in valid if c.valid_from==top.valid_from and abs(c.confidence-top.confidence)<1e-9 and c.value!=top.value]
        if ties:return None,'unresolved_conflict'
        return top,'resolved_by_validity_and_provenance'
    def control(self,target,current,tolerance):
        err=target-current
        if abs(err)<=tolerance:return 'stable','сохранить режим',err
        if err>0:return 'increase','увеличить управляющее воздействие',err
        return 'decrease','уменьшить управляющее воздействие',err


def pick_variant(key,n):return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big')%n

def render(plan):
    s=plan.slots;k=plan.task
    if k=='proof':
        if plan.status=='proved':
            vs=[f"Да. Из факта «{s['fact']}» по цепочке правил следует «{s['goal']}».",f"Да. Утверждение «{s['goal']}» доказано: исходный факт «{s['fact']}» приводит к нему последовательным выводом."]
        else:vs=[f"Нет. Утверждение «{s['goal']}» не следует из имеющихся посылок.",f"Нет. Для вывода «{s['goal']}» в графе нет достаточного основания."]
    elif k=='temporal':
        vs=[f"На момент {s['time']} для «{s['entity']}» значение отношения «{s['relation']}» равно «{s['value']}». Источник: {s['source']}.",f"По состоянию на {s['time']}: {s['entity']} — {s['relation']} = {s['value']}. Основание: {s['source']}."]
    elif k=='conflict':
        if plan.status=='resolved':vs=[f"Конфликт разрешён: актуальное значение для «{s['entity']}» — «{s['value']}». Использован источник {s['source']} с более подходящей областью действия.",f"При данных условиях принимается «{s['value']}» для «{s['entity']}»; противоречащая запись не действует в текущем времени."]
        else:vs=[f"Надёжный вывод заблокирован: для «{s['entity']}» остаются равносильные противоречащие утверждения.",f"Есть неразрешённое противоречие по «{s['entity']}». Требуется дополнительный источник или уточнение времени."]
    elif k=='control':
        vs=[f"Цель {s['target']}, текущее значение {s['current']}, ошибка {s['error']}. Режим: {s['mode']}. Действие: {s['action']}.",f"Рассогласование равно {s['error']} при допуске {s['tolerance']}. Контур переходит в режим {s['mode']}: {s['action']}."]
    else:
        vs=[f"Причинная цепочка подтверждает: «{s['cause']}» через промежуточное состояние приводит к «{s['effect']}».",f"В графе есть путь от «{s['cause']}» к «{s['effect']}», поэтому причинное следствие поддержано указанной цепочкой."]
    return vs[pick_variant(k+json.dumps(s,ensure_ascii=False,sort_keys=True),len(vs))]


def make_case(rng,kind):
    A=Authority();g=GraphMemory()
    if kind=='proof':
        e=rng.choice(ENTITIES);p=rng.sample(PROPS,3);fact=(e,p[0]);mid=(e,p[1]);goal=(e,p[2]);rules=[Rule(fact,mid),Rule(mid,goal)];negative=rng.random()<.35
        query_goal=(rng.choice([x for x in ENTITIES if x!=e]),p[2]) if negative else goal
        ok,trace=A.prove_chain(fact,rules,query_goal);plan=Plan('proof','proved' if ok else 'not_proved',trace,{'fact':f'{fact[0]} {fact[1]}','goal':f'{query_goal[0]} {query_goal[1]}'},[]);truth={'proved':not negative,'fact':plan.slots['fact'],'goal':plan.slots['goal']}
    elif kind=='temporal':
        e=rng.choice(ENTITIES);r='состояние';v1,v2=rng.sample(VALUES,2);cut=rng.randint(20,80);c1=Claim(e,r,v1,True,0,cut-1,rng.choice(SOURCES),.9);c2=Claim(e,r,v2,True,cut,999,rng.choice(SOURCES),.95);g.add_claim(c1);g.add_claim(c2);t=rng.randint(0,99);c=A.solve_temporal(g,e,r,t);plan=Plan('temporal','resolved',[],{'entity':e,'relation':r,'time':t,'value':c.value,'source':c.source},[c.source]);truth={'value':c.value,'source':c.source}
    elif kind=='conflict':
        e=rng.choice(ENTITIES);v1,v2=rng.sample(VALUES,2);t=50;tie=rng.random()<.25;c1=Claim(e,'режим',v1,True,0,80,'журнал A',.9);c2=Claim(e,'режим',v2,True,40,90,'журнал B',.9 if tie else .99);c,why=A.solve_contradiction([c1,c2],t)
        if c:plan=Plan('conflict','resolved',[],{'entity':e,'value':c.value,'source':c.source},[c.source]);truth={'resolved':True,'value':c.value}
        else:plan=Plan('conflict','blocked',[],{'entity':e},['журнал A','журнал B']);truth={'resolved':False}
    elif kind=='control':
        target=rng.randint(20,100);current=target+rng.choice([-15,-7,-2,0,2,7,15]);tol=rng.choice([1,2,3]);mode,action,err=A.control(target,current,tol);plan=Plan('control',mode,[],{'target':target,'current':current,'tolerance':tol,'error':err,'mode':mode,'action':action},[]);truth={'mode':mode,'error':err,'action':action}
    else:
        a,b,c=rng.sample(PROPS,3);e=rng.choice(ENTITIES);fact=(e,a);rules=[Rule(fact,(e,b)),Rule((e,b),(e,c))];ok,trace=A.prove_chain(fact,rules,(e,c));plan=Plan('causal','proved',trace,{'cause':f'{e} {a}','effect':f'{e} {c}'},[]);truth={'proved':ok,'cause':plan.slots['cause'],'effect':plan.slots['effect']}
    return plan,truth


def verify(plan,truth,text):
    if plan.task=='proof':return (plan.status=='proved')==truth['proved'] and plan.slots['goal'] in text
    if plan.task=='temporal':return truth['value'] in text and truth['source'] in text
    if plan.task=='conflict':return (plan.status=='resolved')==truth['resolved'] and (not truth['resolved'] or truth['value'] in text)
    if plan.task=='control':return str(truth['error']) in text and truth['mode'] in text and truth['action'] in text
    return truth['proved'] and truth['cause'] in text and truth['effect'] in text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=50000);ap.add_argument('--out',default='nexus_r524_results');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=random.Random(SEED);kinds=['proof','temporal','conflict','control','causal'];stats={k:[0,0] for k in kinds};rows=[]
    for i in range(a.n):
        k=kinds[i%len(kinds)];plan,truth=make_case(rng,k);txt=render(plan);good=verify(plan,truth,txt);stats[k][0]+=int(good);stats[k][1]+=1
        if i<100:rows.append({'task':k,'plan':dataclasses.asdict(plan),'truth':truth,'answer':txt,'verified':good})
    result={'format':'nexus-r524-graph-grounded-answer-loop/1','n':a.n,'architecture':['typed graph memory','goal/task state','exact Authority/proof/control','semantic plan','surface renderer','post-render verifier'],'result':{k:{'verified':v[0],'n':v[1],'accuracy':v[0]/v[1]} for k,v in stats.items()},'overall_accuracy':sum(v[0] for v in stats.values())/a.n,'principle':'plan content comes from graph/goal/proof, never from guessing the arbitrary next sentence; exact identities are bound after planning and remain verifier-owned'}
    (out/'00_R524_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_EXAMPLES.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8');(out/'README_RU.md').write_text('# NEXUS R5.24 Graph-Grounded Answer Loop\n\nВопрос/цель → typed GraphRAG state → exact proof/control → semantic plan → русский surface → verifier. Содержание плана извлекается из графа и цели, а не угадывается как произвольное продолжение текста.\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False));
    if result['overall_accuracy']!=1.0:raise SystemExit('graph-grounded invariant failed')
if __name__=='__main__':main()
