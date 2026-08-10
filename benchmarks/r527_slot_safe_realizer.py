#!/usr/bin/env python3
import argparse, json, random, re, shutil, time
from pathlib import Path

import torch
import torch.nn.functional as F

import r57_concept_graph_language as base
from r515_encoder_decoder import FixedSP
from r524_graph_grounded_answer_loop import make_case, render as deterministic_render, verify as verify_bound

SEED=20260810
PREFIX_TOKENS=80
TARGET_TOKENS=48
BATCH=8
END_MARK='<END>'
STYLES=('кратко','формально','объяснительно')
KINDS=('proof','temporal','conflict','control','causal')
SLOT_RE=re.compile(r'\[S(\d+)\]')

SLOT_KEYS={
    'proof_proved': ['fact','goal'],
    'proof_not_proved': ['fact','goal'],
    'temporal': ['entity','relation','time','value','source'],
    'conflict_resolved': ['entity','value','source'],
    'conflict_blocked': ['entity'],
    'control': ['target','current','error','tolerance','mode','action'],
    'causal': ['cause','effect'],
}

TEMPLATES={
 'proof_proved': {
   'кратко':['Да. Из «[S0]» следует «[S1]».'],
   'формально':['Да. Утверждение «[S1]» доказано из исходного факта «[S0]» по цепочке правил.'],
   'объяснительно':['Да. Начинаем с факта «[S0]». Последовательное применение правил приводит к выводу «[S1]».'],
 },
 'proof_not_proved': {
   'кратко':['Нет. «[S1]» не доказано из «[S0]».'],
   'формально':['Нет. Из исходного факта «[S0]» и доступных правил утверждение «[S1]» не выводится.'],
   'объяснительно':['Нет. Имеющегося основания «[S0]» недостаточно, чтобы получить вывод «[S1]».'],
 },
 'temporal': {
   'кратко':['На момент [S2]: «[S0]» — [S1] = «[S3]». Источник: [S4].'],
   'формально':['По состоянию на [S2] для сущности «[S0]» отношение «[S1]» имеет значение «[S3]». Provenance: [S4].'],
   'объяснительно':['Для момента [S2] граф памяти выбирает действующий факт: «[S0]» — [S1] = «[S3]». Основание — источник [S4].'],
 },
 'conflict_resolved': {
   'кратко':['Конфликт разрешён: для «[S0]» принимается «[S1]». Источник: [S2].'],
   'формально':['Противоречие разрешено по времени и provenance: актуальное значение для «[S0]» — «[S1]», источник [S2].'],
   'объяснительно':['Записи конфликтуют, но действующая запись имеет приоритет. Поэтому для «[S0]» принимается «[S1]» из источника [S2].'],
 },
 'conflict_blocked': {
   'кратко':['Вывод заблокирован: по «[S0]» остаётся неразрешённое противоречие.'],
   'формально':['Надёжный вывод по «[S0]» запрещён Authority: равносильные противоречащие утверждения не разрешены.'],
   'объяснительно':['По «[S0]» есть два равноправных несовместимых факта. До уточнения источника или времени вывод делать нельзя.'],
 },
 'control': {
   'кратко':['Цель [S0], текущее [S1], ошибка [S2]. Режим [S4]. Действие: [S5].'],
   'формально':['При цели [S0], текущем значении [S1] и допуске [S3] рассогласование равно [S2]. Контур: [S4]; управляющее действие: [S5].'],
   'объяснительно':['Система сравнивает цель [S0] с текущим значением [S1]. Ошибка равна [S2] при допуске [S3], поэтому выбирается режим [S4] и действие «[S5]».'],
 },
 'causal': {
   'кратко':['Причинная цепочка подтверждает: «[S0]» приводит к «[S1]».'],
   'формально':['В графе доказан причинный путь от «[S0]» к «[S1]» через промежуточное состояние.'],
   'объяснительно':['Начальное состояние «[S0]» запускает цепочку переходов, конечным следствием которой является «[S1]».'],
 },
}


def case_key(plan):
    if plan.task=='proof': return 'proof_'+plan.status
    if plan.task=='conflict': return 'conflict_'+('resolved' if plan.status=='resolved' else 'blocked')
    return plan.task


def abstract_case(plan, style):
    key=case_key(plan); keys=SLOT_KEYS[key]
    slots={f'[S{i}]':plan.slots[k] for i,k in enumerate(keys)}
    prompt=(f'ЗАДАЧА={plan.task}; СТАТУС={plan.status}; СТИЛЬ={style}; '
            f'СЛОТЫ=' + ', '.join(f'[S{i}]={k}' for i,k in enumerate(keys)) + '; '
            f'ШАГОВ_ДОКАЗАТЕЛЬСТВА={len(plan.steps)}; '
            'Сформулируй русский ответ. Значения слотов не выдумывай: оставь маркеры [S0], [S1] и т.д. без изменений. ОТВЕТ:')
    target=TEMPLATES[key][style][0]+' '+END_MARK
    return prompt,target,slots,key


def encode_example(tok,prompt,target):
    c=tok.enc(prompt.encode('utf-8'))[-PREFIX_TOKENS:]
    c=[0]*(PREFIX_TOKENS-len(c))+c
    t=tok.enc(target.encode('utf-8'))[:TARGET_TOKENS]
    x=c+[0]+t[:-1]+[0]*(TARGET_TOKENS-len(t))
    y=[0]*PREFIX_TOKENS+t+[0]*(TARGET_TOKENS-len(t))
    m=[0]*PREFIX_TOKENS+[1]*len(t)+[0]*(TARGET_TOKENS-len(t))
    return x,y,m


def load_warm(path):
    ck=torch.load(path,map_location='cpu');m=base.LM();m.load_state_dict(ck['state_dict'],strict=True);assert base.param_count(m)==2_998_620;return m


def train(model,tok,steps):
    rng=random.Random(SEED+527);torch.manual_seed(SEED);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.95),weight_decay=.01);hist=[];t0=time.perf_counter();model.train();seen=0
    for step in range(steps):
        xs=[];ys=[];ms=[]
        for _ in range(BATCH):
            kind=rng.choice(KINDS);plan,truth=make_case(rng,kind);style=rng.choice(STYLES);p,t,_,_=abstract_case(plan,style);x,y,m=encode_example(tok,p,t);xs.append(x);ys.append(y);ms.append(m);seen+=1
        x=torch.tensor(xs,dtype=torch.long);y=torch.tensor(ys,dtype=torch.long);mask=torch.tensor(ms,dtype=torch.float32)
        z=model(x);ce=F.cross_entropy(z.reshape(-1,base.VOCAB),y.reshape(-1),reduction='none').view_as(y);loss=(ce*mask).sum()/mask.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();hist.append(float(loss))
        if (step+1)%256==0:print(json.dumps({'step':step+1,'ce':sum(hist[-64:])/len(hist[-64:]),'examples':seen},ensure_ascii=False),flush=True)
    return {'steps':steps,'examples':seen,'train_s':time.perf_counter()-t0,'last64_ce':sum(hist[-64:])/len(hist[-64:])}


def batch_generate(model,tok,prompts,max_new=TARGET_TOKENS):
    contexts=[]
    for p in prompts:
        c=tok.enc(p.encode('utf-8'))[-PREFIX_TOKENS:];contexts.append([0]*(PREFIX_TOKENS-len(c))+c)
    outs=[[] for _ in prompts];model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            seq=[contexts[i]+[0]+outs[i] for i in range(len(prompts))]
            x=torch.tensor(seq,dtype=torch.long);z=model(x);nxt=z[:,-1].argmax(-1).tolist()
            for i,n in enumerate(nxt):outs[i].append(int(n))
    texts=[]
    for o in outs:
        s=tok.dec(o).decode('utf-8','replace');texts.append(s.split(END_MARK,1)[0].strip())
    return texts


def skeleton_ok(text,key):
    req=[f'[S{i}]' for i in range(len(SLOT_KEYS[key]))]
    got=set(SLOT_RE.findall(text)); expected={str(i) for i in range(len(req))}
    if got!=expected:return False,'slot_set'
    if '�' in text:return False,'utf8'
    if len(text)<12 or len(text)>600:return False,'length'
    if key=='proof_proved' and not text.startswith('Да.'):return False,'polarity'
    if key=='proof_not_proved' and not text.startswith('Нет.'):return False,'polarity'
    if key=='conflict_blocked' and not any(x in text.lower() for x in ('заблок','нельзя','запрещ')):return False,'conflict_status'
    return True,'ok'


def bind(text,slots):
    out=text
    for k,v in slots.items():out=out.replace(k,str(v))
    return out


def evaluate(model,tok,n=2048):
    rng=random.Random(SEED+9001);cases=[]
    for _ in range(n):
        kind=rng.choice(KINDS);plan,truth=make_case(rng,kind);style=rng.choice(STYLES);p,t,slots,key=abstract_case(plan,style);cases.append((plan,truth,style,p,t,slots,key))
    neural_ok=0;bound_ok=0;fallback=0;reasons={};rows=[]
    for s in range(0,n,64):
        batch=cases[s:s+64];texts=batch_generate(model,tok,[x[3] for x in batch])
        for item,text in zip(batch,texts):
            plan,truth,style,p,target,slots,key=item;ok,reason=skeleton_ok(text,key);neural_ok+=int(ok);reasons[reason]=reasons.get(reason,0)+1
            if ok:
                b=bind(text,slots);good=verify_bound(plan,truth,b);bound_ok+=int(good)
                if not good:fallback+=1;final=deterministic_render(plan)
                else:final=b
            else:
                fallback+=1;final=deterministic_render(plan);good=verify_bound(plan,truth,final);bound_ok+=int(good)
            if len(rows)<80:rows.append({'task':key,'style':style,'neural_skeleton':text,'skeleton_ok':ok,'reason':reason,'slots':slots,'final':final,'final_verified':good,'teacher_skeleton':target.replace(' '+END_MARK,'')})
    return {'n':n,'neural_skeleton_accept':neural_ok/n,'final_verified_accuracy':bound_ok/n,'fallback_rate':fallback/n,'reasons':reasons,'rows':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--warm-checkpoint',required=True);ap.add_argument('--steps',type=int,default=4096);ap.add_argument('--eval-n',type=int,default=2048);ap.add_argument('--out',default='nexus_r527_results');ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shutil.copy2(a.tokenizer_model,out/'sp_unigram_4096.model');tok=FixedSP(a.tokenizer_model);m=load_warm(a.warm_checkpoint);tr=train(m,tok,a.steps);ev=evaluate(m,tok,a.eval_n)
    result={'format':'nexus-r527-slot-safe-neural-realizer/1','protocol':{'params':base.param_count(m),'warm_start':'R5.12 32K Russian organ','task':'typed semantic plan -> slot-preserving Russian skeleton','slot_policy':'neural model never receives slot values; it emits only [S0]...[Sn]; Authority binds exact graph values afterward','fallback':'deterministic verified renderer on malformed skeleton or failed semantic verification','steps':a.steps,'train_examples':tr['examples'],'eval_unseen':a.eval_n},'training':tr,'evaluation':{k:v for k,v in ev.items() if k!='rows'}};(out/'00_R527_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'01_EXAMPLES.json').write_text(json.dumps(ev['rows'],ensure_ascii=False,indent=2),encoding='utf-8');torch.save({'state_dict':m.state_dict(),'protocol':result['protocol']},out/'R527_SLOT_SAFE_REALIZER_3M.pt');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
