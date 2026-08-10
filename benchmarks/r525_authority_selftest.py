#!/usr/bin/env python3
import json,random,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nexus.authority_bus import route,solve_logic,solve_cyber

SEED=20260810
PROPS=[
 'сигнал получен','проверка начата','доступ разрешён','архив проверен','модуль активен',
 'канал открыт','данные загружены','отчёт построен','датчик сработал','задача подтверждена',
 'контроль выполнен','выход доступен','процесс завершён','шлюз готов','сервис отвечает']

def logic_text(p1,p2,p3,fact,claim=None):
 claim=claim or p3
 return (f'Правило 1: если истинно «{p1}», то истинно «{p2}». '
         f'Правило 2: если истинно «{p2}», то истинно «{p3}». '
         f'Факт: истинно «{fact}». Вопрос: следует ли, что «{claim}»?')

def cyber_text(target,current,tol):
 return f'Цель регулятора: {target}. Текущее значение: {current}. Допуск: {tol}. Каково рассогласование и какое действие нужно выполнить?'

def audit_legacy_logic(n=200000):
 rng=random.Random(SEED+1);contradictions=0;negative=0
 examples=[]
 for _ in range(n):
  p1,p2,p3=rng.sample(PROPS,3);yes=rng.random()<.65
  if yes:fact=p1;legacy=True
  else:
   negative+=1;fact=rng.choice([x for x in PROPS if x not in (p1,p2)]);legacy=False
  exact=solve_logic(logic_text(p1,p2,p3,fact,p3))['answer_yes']
  if exact!=legacy:
   contradictions+=1
   if len(examples)<8:examples.append({'p1':p1,'p2':p2,'p3_claim':p3,'fact':fact,'legacy_label':legacy,'exact_truth':exact})
 return {'samples':n,'negative_samples':negative,'contradictions':contradictions,'rate':contradictions/n,'expected_analytic_rate':.35/13,'examples':examples}

def exact_random_selftest(n=50000):
 rng=random.Random(SEED+2);logic_ok=cyber_ok=0
 for _ in range(n):
  p1,p2,p3=rng.sample(PROPS,3);kind=rng.randrange(5)
  if kind==0:fact=p1;claim=p3;truth=True
  elif kind==1:fact=p2;claim=p3;truth=True
  elif kind==2:fact=p3;claim=p3;truth=True
  elif kind==3:fact=p1;claim=p2;truth=True
  else:
   fact=rng.choice([x for x in PROPS if x not in (p1,p2,p3)]);claim=p3;truth=False
  r=route(logic_text(p1,p2,p3,fact,claim));logic_ok+=int(r and r['kind']=='logic' and r['answer_yes']==truth)
  target=rng.randint(10,120);current=target+rng.randint(-30,30);tol=rng.randint(1,5);e=target-current
  if abs(e)<=tol:mode,action='стабильно','удерживать воздействие'
  elif e>0:mode,action='коррекция','увеличить воздействие'
  else:mode,action='коррекция','уменьшить воздействие'
  c=route(cyber_text(target,current,tol));cyber_ok+=int(c and c['kind']=='cyber' and c['error']==e and c['mode']==mode and c['action']==action)
 return {'samples_each':n,'logic_correct':logic_ok,'logic_accuracy':logic_ok/n,'cyber_correct':cyber_ok,'cyber_accuracy':cyber_ok/n}

def fallthrough_selftest():
 texts=['Вечером он вышел из дома и','Объясни причинность в экономике.','Москва — столица России.','Что ты думаешь об этой статье?']
 return {'cases':[{'text':t,'route':route(t)} for t in texts],'all_fallthrough':all(route(t) is None for t in texts)}

def main():
 out=Path('nexus_r525_results');out.mkdir(exist_ok=True)
 result={'format':'nexus-r525-authority-selftest/1','legacy_logic_audit':audit_legacy_logic(),'exact_selftest':exact_random_selftest(),'fallthrough':fallthrough_selftest()}
 assert result['exact_selftest']['logic_accuracy']==1.0
 assert result['exact_selftest']['cyber_accuracy']==1.0
 assert result['fallthrough']['all_fallthrough']
 (out/'00_R525_AUTHORITY_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
