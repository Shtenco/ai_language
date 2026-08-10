#!/usr/bin/env python3
import argparse, json, random, re
from pathlib import Path
from r526_integrated_cognitive_loop import NexusCore

NUM=r'-?\d+(?:[.,]\d+)?'
QUOT=r'[«"]([^»"]+)[»"]'

class SemanticCompiler:
    def __init__(self):
        self.pending_goal={}
    @staticmethod
    def _norm(s):return re.sub(r'\s+',' ',s.strip())
    @staticmethod
    def _num(s):
        s=s.replace(',','.')
        return float(s) if '.' in s else int(s)
    def compile(self,text,session='default'):
        t=self._norm(text);low=t.lower()
        m=re.search(rf'(?:получили|получено|наблюдаем|фактически|результат)\s*[:=]?\s*({NUM})\b',low)
        if m and session in self.pending_goal:
            return {'op':'feedback','goal_id':self.pending_goal[session],'observed':str(self._num(m.group(1)))}
        mg=re.search(rf'(?:цель|нужно|надо)\s*[:—-]?\s*(.+?)(?:[,;]|$)',t,re.I)
        me=re.search(rf'(?:ожидаемое\s+значение|ожидается|ожидаем|критерий)\s*[:=]?\s*({NUM}|[^,;]+)',t,re.I)
        if mg and me:
            expected=str(self._num(me.group(1))) if re.fullmatch(NUM,me.group(1).strip()) else me.group(1).strip()
            criterion='значение'
            if re.search(r'http|код|статус',low):criterion='HTTP'
            return {'op':'set_goal','session':session,'text':mg.group(1).strip(),'criterion':criterion,'expected':expected}
        mr=re.search(r'(?:запомни|сохрани|запиши)(?:,?\s*что)?\s+[«"]?([^,»"]+?)[»"]?\s+(?:имеет\s+)?(?:отношение\s+)?[«"]?([а-яёa-z_-]{2,})[»"]?\s*(?:=|равно|имеет значение|:)\s*[«"]?([^,»";]+)[»"]?',t,re.I)
        if mr:
            e,r,v=[x.strip(' «»"') for x in mr.groups()]
            vf=0;vt=10**9;src='user'
            mfrom=re.search(rf'(?:с|от)\s+({NUM})',low);mto=re.search(rf'(?:по|до)\s+({NUM})',low);ms=re.search(r'(?:источник|source)\s*[:=]?\s*([^,;]+)',t,re.I)
            if mfrom:vf=int(float(mfrom.group(1).replace(',','.')))
            if mto:vt=int(float(mto.group(1).replace(',','.')))
            if ms:src=ms.group(1).strip()
            return {'op':'remember_claim','entity':e,'relation':r,'value':v,'valid_from':vf,'valid_to':vt,'source':src}
        mre=re.search(r'(?:запомни|сохрани|запиши)(?:,?\s*что)?\s+[«"]?([^,»"]+?)[»"]?\s+(причиняет|вызывает|является|часть|связан(?:а|о)?)\s+[«"]?([^,»"]+)[»"]?',t,re.I)
        if mre:
            a,rel,b=[x.strip(' «»"') for x in mre.groups()]
            mp={'причиняет':'CAUSES','вызывает':'CAUSES','является':'IS_A','часть':'PART_OF','связан':'RELATED_TO','связана':'RELATED_TO','связано':'RELATED_TO'}
            return {'op':'remember_relation','a':a,'relation':mp.get(rel.lower(),rel.upper()),'b':b,'source':'user'}
        mq=re.search(rf'(?:какое|каков|каково|что за|покажи)\s+(?:было\s+)?(?:значение\s+)?(?:отношения\s+)?[«"]?([а-яёa-z_-]{{2,}})[»"]?\s+(?:у|для)\s+[«"]?([^?»"]+?)[»"]?\s+(?:на|в)\s+(?:момент\s+)?({NUM})',t,re.I)
        if mq:
            rel,e,tm=mq.groups();return {'op':'current','entity':e.strip(),'relation':rel.strip(),'time':int(float(tm.replace(',','.')))}
        mq2=re.search(rf'(?:какое|каков|каково)\s+(?:состояние|статус)\s+(?:было\s+)?(?:у|для)\s+[«"]?([^?»"]+?)[»"]?\s+(?:на|в)\s+(?:момент\s+)?({NUM})',t,re.I)
        if mq2:
            e,tm=mq2.groups();rel='состояние' if 'состояние' in low else 'статус';return {'op':'current','entity':e.strip(),'relation':rel,'time':int(float(tm.replace(',','.')))}
        q=t
        q=re.sub(r'^(?:что известно (?:о|про)|найди (?:в памяти )?(?:всё )?(?:о|про)|покажи (?:сведения|данные) (?:о|про))\s+','',q,flags=re.I)
        return {'op':'retrieve','query':q.strip(' ?'),'hops':2}
    def execute(self,core,text,session='default'):
        ir=self.compile(text,session)
        r=core.execute(ir)
        if ir['op']=='set_goal' and isinstance(r,dict) and 'goal_id' in r:self.pending_goal[session]=r['goal_id']
        return ir,r


def selftest(out,n=50000):
    rng=random.Random(20260810);core=NexusCore(Path(out)/'semantic_core.sqlite');c=SemanticCompiler();ok=0;rows=[]
    for i in range(2500):core.remember_claim(f'объект-{i:04d}','состояние','старое',0,49,'archive',.9);core.remember_claim(f'объект-{i:04d}','состояние','новое',50,999,'sensor',.99)
    for i in range(n):
        typ=i%5
        if typ==0:
            idx=rng.randrange(2500);tm=rng.choice([10,80]);forms=[f'Какое состояние было у объект-{idx:04d} на момент {tm}?',f'Покажи состояние для объект-{idx:04d} в момент {tm}',f'Каково состояние у объект-{idx:04d} на {tm}?'];text=rng.choice(forms);ir,res=c.execute(core,text);good=ir['op']=='current' and res['status']=='verified' and (('старое' in res['answer'])==(tm<50))
        elif typ==1:
            idx=rng.randrange(2500);forms=[f'Что известно про объект-{idx:04d}?',f'Найди в памяти всё о объект-{idx:04d}',f'Покажи сведения про объект-{idx:04d}'];text=rng.choice(forms);ir,res=c.execute(core,text);good=ir['op']=='retrieve' and res['status'] in ('retrieved','unknown')
        elif typ==2:
            e=f'API-{i}';text=f'Запомни, что {e} статус = готов, с 0 по 100, источник test-{i}';ir,res=c.execute(core,text);good=ir['op']=='remember_claim' and res.get('ok') is True
        elif typ==3:
            s=f's{i}';text='Цель: получить HTTP 200; ожидаемое значение 200';ir,res=c.execute(core,text,s);good=ir['op']=='set_goal' and 'goal_id' in res
        else:
            s=f's{i}';ir0,res0=c.execute(core,'Цель: получить HTTP 200; ожидаемое значение 200',s);obs='200' if rng.random()<.5 else '500';ir,res=c.execute(core,f'Получили {obs}',s);good=ir['op']=='feedback' and ((obs=='200' and res['status']=='stable') or (obs=='500' and res['status']=='recover'))
        ok+=int(good)
        if len(rows)<80:rows.append({'text':text if typ!=4 else f'Получили {obs}','ir':ir,'result':res,'ok':good})
    core.close();result={'format':'nexus-r528-russian-semantic-compiler/2','n':n,'passed':ok,'accuracy':ok/n,'architecture':'deterministic Russian semantic compiler -> typed R5.26 IR; exact spans/numbers remain copied, not generated','intents':['current','retrieve','remember_claim','remember_relation','set_goal','feedback'],'default_policy':'unrecognized factual text compiles to read-only retrieve, never to a write/execute operation'};Path(out,'00_R528_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');Path(out,'01_EXAMPLES.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--n',type=int,default=50000);ap.add_argument('--out',default='nexus_r528_results');ap.add_argument('--db',default='nexus_r528.sqlite');a=ap.parse_args()
    if a.selftest:
        Path(a.out).mkdir(parents=True,exist_ok=True);r=selftest(a.out,a.n);print(json.dumps(r,ensure_ascii=False));assert r['accuracy']==1.0
    else:
        core=NexusCore(a.db);comp=SemanticCompiler();import sys
        try:
            for line in sys.stdin:
                t=line.strip()
                if not t:continue
                try:ir,res=comp.execute(core,t);print(json.dumps({'ir':ir,'result':res},ensure_ascii=False),flush=True)
                except Exception as e:print(json.dumps({'error':repr(e)},ensure_ascii=False),flush=True)
        finally:core.close()
if __name__=='__main__':main()
