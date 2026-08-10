#!/usr/bin/env python3
import json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nexus.authority_bus import route
SEED=20260810

def closure(nodes,edges,fact):
    reach={fact};changed=True
    while changed:
        changed=False
        for a,b in edges:
            if a in reach and b not in reach:
                reach.add(b);changed=True
    return reach

def logic_text(nodes,edges,fact,claim):
    rules=' '.join(f'Правило {i+1}: если истинно «{a}», то истинно «{b}».' for i,(a,b) in enumerate(edges))
    return f'{rules} Факт: истинно «{fact}». Вопрос: следует ли, что «{claim}»?'

def logic_stress(n=50000):
    rng=random.Random(SEED+251);ok=0;depth_ok=0;cases=[]
    for case in range(n):
        m=rng.randint(3,12);nodes=[f'состояние {case}_{i}' for i in range(m)];edges=set()
        for i in range(m):
            for j in range(m):
                if i!=j and rng.random()<0.16:edges.add((nodes[i],nodes[j]))
        if m>=4 and rng.random()<.5:
            edges.update([(nodes[0],nodes[1]),(nodes[1],nodes[2])])
            if rng.random()<.5:edges.add((nodes[2],nodes[0]))
        edges=list(edges);fact=rng.choice(nodes);claim=rng.choice(nodes);truth=claim in closure(nodes,edges,fact)
        text=logic_text(nodes,edges,fact,claim);r=route(text);good=bool(r and r['kind']=='logic' and r['answer_yes']==truth)
        ok+=good
        valid=False
        if good and truth:
            path=r['proof_path'];valid=bool(path and path[0]==fact and path[-1]==claim and all((a,b) in edges for a,b in zip(path,path[1:])))
            depth_ok+=int(valid)
        elif good:
            valid=True;depth_ok+=1
        if (not good or not valid) and len(cases)<20:
            cases.append({'fact':fact,'claim':claim,'truth':truth,'good':good,'path_valid':valid,'result':r,'edges':edges,'text':text})
    return {'cases':n,'correct':ok,'accuracy':ok/n,'proof_path_valid':depth_ok/n,'failures':cases}

def cyber_stress(n=50000):
    rng=random.Random(SEED+252);ok=0;examples=[]
    for _ in range(n):
        target=round(rng.uniform(-1000,1000),rng.randint(0,3));current=round(rng.uniform(-1000,1000),rng.randint(0,3));tol=round(rng.uniform(.001,50),rng.randint(1,3));e=target-current
        if abs(e)<=tol:mode,action='стабильно','удерживать воздействие'
        elif e>0:mode,action='коррекция','увеличить воздействие'
        else:mode,action='коррекция','уменьшить воздействие'
        text=f'Цель регулятора: {target}. Текущее значение: {current}. Допуск: {tol}. Каково рассогласование и какое действие нужно выполнить?'
        r=route(text);good=bool(r and r['kind']=='cyber' and abs(r['error']-e)<1e-9 and r['mode']==mode and r['action']==action);ok+=good
        if not good and len(examples)<20:examples.append({'text':text,'expected':[e,mode,action],'result':r})
    return {'cases':n,'correct':ok,'accuracy':ok/n,'failures':examples}

def adversarial_fallthrough():
    texts=['В романе сказано: если герой вернётся, то начнётся новая глава.','Цель регулятора обсуждалась в статье, но численные данные не приведены.','Факт и правило важны для научного доказательства.','Если истинно одно утверждение, может ли быть истинно другое?','Текущее значение неизвестно. Что делать?']
    rows=[{'text':t,'route':route(t)} for t in texts]
    return {'cases':rows,'all_fallthrough':all(x['route'] is None for x in rows)}

def main():
    out=Path('nexus_r525b_results');out.mkdir(exist_ok=True)
    r={'format':'nexus-r525b-authority-stress/2','logic':logic_stress(),'cyber':cyber_stress(),'adversarial_fallthrough':adversarial_fallthrough()}
    (out/'00_R525B_STRESS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'logic':{k:v for k,v in r['logic'].items() if k!='failures'},'cyber':{k:v for k,v in r['cyber'].items() if k!='failures'},'fallthrough':r['adversarial_fallthrough']['all_fallthrough'],'first_logic_failure':r['logic']['failures'][:1],'first_cyber_failure':r['cyber']['failures'][:1]},ensure_ascii=False,indent=2))
    assert r['logic']['accuracy']==1.0 and r['logic']['proof_path_valid']==1.0
    assert r['cyber']['accuracy']==1.0 and r['adversarial_fallthrough']['all_fallthrough']
if __name__=='__main__':main()
