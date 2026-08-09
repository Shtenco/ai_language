#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import torch
import r52_tokenizer_tournament as tt
import r53_source_matched_lm as r53

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seeds',default='3,7,11,17,23,29,37,47,61,73');ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);train=Path(a.train).read_bytes();tests={Path(p).stem:Path(p).read_bytes() for p in a.tests}
    toks={'BYTE256':r53.ByteTok(),'UNIGRAM2048':tt.train_sp(train,2048,'unigram',out)}
    probe='Hello, мир! Проверка 123.\n'.encode();assert toks['UNIGRAM2048'].dec(toks['UNIGRAM2048'].enc(probe))==probe
    rows=[]
    for s in map(int,a.seeds.split(',')):
        for name,tok in toks.items():
            print('RUN',s,name,flush=True);m,tm=r53.train_one(name,'r4',s,tok,train);row={'seed':s,'model':'R4_'+name,'params':r53.pc(m),**tm}
            for sp,raw in tests.items():
                e=r53.evaluate(m,tok,raw,s);row.update({f'{sp}_{k}':v for k,v in e.items()})
            rt=r53.runtime(m,tok,tests[next(iter(tests))],s);row.update({'rt_'+k:v for k,v in rt.items()});rows.append(row);print(json.dumps({'seed':s,'model':row['model'],**{sp:row[f'{sp}_bpb'] for sp in tests},'byte_s':row['rt_source_byte_s']}),flush=True)
    bpb=[f'{sp}_bpb' for sp in tests]
    for row in rows:row['mean_eval_bpb']=sum(row[k] for k in bpb)/len(bpb)
    agg=[]
    for model in ('R4_BYTE256','R4_UNIGRAM2048'):
        rr=[x for x in rows if x['model']==model];z={'model':model,'n':len(rr),'params':999978}
        for k in bpb+['mean_eval_bpb','rt_source_byte_s','train_s','train_source_bytes']:
            v=[x[k] for x in rr];mu=sum(v)/len(v);z[k+'_mean']=mu;z[k+'_sd']=(sum((x-mu)**2 for x in v)/len(v))**0.5
        agg.append(z)
    paired=[]
    by={(x['seed'],x['model']):x for x in rows}
    for s in map(int,a.seeds.split(',')):
        u=by[(s,'R4_UNIGRAM2048')];b=by[(s,'R4_BYTE256')];paired.append({'seed':s,'byte_mean_bpb':b['mean_eval_bpb'],'unigram_mean_bpb':u['mean_eval_bpb'],'delta_bpb':u['mean_eval_bpb']-b['mean_eval_bpb'],'relative_pct':100*(u['mean_eval_bpb']/b['mean_eval_bpb']-1)})
    result={'protocol':{'source_context_bytes':128,'steps':256,'batch':8,'params':999978,'topology':'R4','seeds':[x['seed'] for x in paired]},'aggregate':agg,'paired':paired,'per_seed':rows}
    (out/'00_CONFIRM_RESULTS.json').write_text(json.dumps(result,indent=2));
    for fn,arr in [('01_PER_SEED.csv',rows),('02_PAIRED.csv',paired),('03_AGGREGATE.csv',agg)]:
        keys=list(arr[0]);
        with open(out/fn,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(arr)
    wins=sum(x['delta_bpb']<0 for x in paired);mean_delta=sum(x['delta_bpb'] for x in paired)/len(paired);mean_rel=sum(x['relative_pct'] for x in paired)/len(paired)
    (out/'README_RU.md').write_text(f'# NEXUS R5.4 — 10-SEED UNIGRAM CONFIRMATION\n\nR4 Unigram2048 vs R4 Byte256. Exactly 999,978 params, matched 128 source-byte context and matched source-byte LM exposure.\n\nUnigram wins: {wins}/{len(paired)} seeds. Mean paired delta BPB: {mean_delta:.6f}; mean paired relative change: {mean_rel:.3f}%.\n')
    print('DONE',json.dumps({'wins':wins,'mean_delta':mean_delta,'mean_rel_pct':mean_rel,'aggregate':agg},indent=2),flush=True)
if __name__=='__main__':main()
