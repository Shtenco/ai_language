import json, os, re, time, urllib.request, urllib.error, hashlib
from pathlib import Path

MODEL=os.environ["MODEL"]
MODEL_ID=os.environ.get("MODEL_ID",MODEL.replace(":","_"))
BENCH=Path(__file__).with_name("BLIND22_FROZEN.json")
OUT=Path(os.environ.get("OUT",f"blind22_{MODEL_ID}.json"))
BASE="http://127.0.0.1:11434"

def post(path,obj,timeout=900):
    data=json.dumps(obj).encode()
    req=urllib.request.Request(BASE+path,data=data,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.loads(r.read().decode())

def pick_answer(text):
    s=(text or "").strip().upper()
    m=re.search(r"(?:^|[^A-ZА-Я])([ABCD])(?:$|[^A-ZА-Я])",s)
    if m:return m.group(1)
    if s and s[0] in "ABCD":return s[0]
    return None

def option_logprobs(resp):
    best={x:None for x in "ABCD"}
    rows=resp.get("logprobs") or []
    for row in rows[:12]:
        cand=[row]+list(row.get("top_logprobs") or [])
        for c in cand:
            tok=(c.get("token") or "").strip().upper()
            if tok in best and best[tok] is None:
                try:best[tok]=float(c.get("logprob"))
                except:pass
        if any(v is not None for v in best.values()):break
    return best

bench=json.loads(BENCH.read_text(encoding="utf-8"))
bench_sha=hashlib.sha256(BENCH.read_bytes()).hexdigest()
try:
    show=post("/api/show",{"model":MODEL},timeout=120)
except Exception as e:
    show={"error":repr(e)}

system=("Ты проходишь закрытый benchmark. Для каждого задания выбери ровно один вариант. "
        "Не объясняй ход решения. В поле content выведи только одну латинскую букву A, B, C или D.")
rows=[]
for i,it in enumerate(bench["items"]):
    prompt=it["q"]+"\n"+"\n".join(f"{k}) {v}" for k,v in it["options"].items())+"\nОтвет:"
    body={
      "model":MODEL,
      "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
      "stream":False,
      "think":False,
      "logprobs":True,
      "top_logprobs":20,
      "options":{"temperature":0,"seed":22026,"num_predict":8,"top_p":1.0}
    }
    t=time.time()
    try:
        r=post("/api/chat",body,timeout=900)
        raw=((r.get("message") or {}).get("content") or "")
        pred=pick_answer(raw)
        err=None
    except Exception as e:
        r={}; raw=""; pred=None; err=repr(e)
    rows.append({
      "id":it["id"],"domain":it["domain"],"gold":it["answer"],"pred":pred,"correct":pred==it["answer"],
      "raw":raw,"option_logprobs":option_logprobs(r),"elapsed_wall_s":time.time()-t,
      "ollama_metrics":{k:r.get(k) for k in ["total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration"]},
      "error":err
    })
    print(MODEL,it["id"],pred,it["answer"],rows[-1]["correct"],flush=True)

valid=[x for x in rows if isinstance(x.get("pred"),str) and x["pred"] in "ABCD"]
summary={
 "model":MODEL,"model_id":MODEL_ID,"blind22_sha256":bench_sha,"n":len(rows),"valid":len(valid),
 "accuracy":sum(x["correct"] for x in rows)/len(rows),
 "valid_accuracy":sum(x["correct"] for x in valid)/len(valid) if valid else None,
 "by_domain":{}
}
for d in sorted(set(x["domain"] for x in rows)):
    z=[x for x in rows if x["domain"]==d]
    summary["by_domain"][d]={"n":len(z),"accuracy":sum(x["correct"] for x in z)/len(z)}
result={"summary":summary,"model_show":show,"rows":rows}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
