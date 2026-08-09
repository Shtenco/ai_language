#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
import torch
import r52_tokenizer_tournament as tt
import r56_3m_russian_adequacy as r56


def conllu_text(path):
    out=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('# text = '):
            t=line[9:].strip()
            if t: out.append(t)
    return ('\n'.join(out)+'\n').encode('utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--synt-test',required=True);ap.add_argument('--gsd-test',required=True);ap.add_argument('--vocab',type=int,choices=[4096,8192],required=True);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args()
    torch.set_num_threads(a.threads);V=a.vocab;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(a.train).read_bytes();tok=tt.train_sp(raw,V,'unigram',out)
    probe='Привет, мир! Проверка lossless №123.\n'.encode('utf-8');assert tok.dec(tok.enc(probe))==probe
    seed=20260809
    m,tm=r56.train_model(V,seed,tok,raw,r56.FINAL_STEPS,r56.FINAL_BATCH,base_lr=1.2e-3)
    tests={'ru_synt_test':conllu_text(a.synt_test),'ru_gsd_shift':conllu_text(a.gsd_test)}
    ev={k:r56.evaluate(m,tok,b,seed,batches=64,batch=8) for k,b in tests.items()}
    rt=r56.runtime(m,tok,tests['ru_synt_test'],seed,reps=30)
    samples=[]
    for i,p in enumerate(r56.PROMPTS):
        for mode in ('greedy','sample'):
            text,cont=r56.generate(m,tok,p,seed+i*17+(0 if mode=='greedy' else 1),mode=mode)
            samples.append({'prompt':p,'mode':mode,'text':text,'continuation':cont,**r56.text_metrics(cont)})
    # Parameter allocation is the central hypothesis of this experiment.
    emb=V*r56.SPECS[V][0];total=tm['params']
    result={'vocab':V,'spec':{'d':r56.SPECS[V][0],'heads':r56.SPECS[V][1],'ff':r56.SPECS[V][2]},'params':total,'token_embedding_params':emb,'token_embedding_fraction':emb/total,'non_token_embedding_params':total-emb,'train':tm,'eval':ev,'runtime':rt,'generation_samples':samples,'source_train_sha256':__import__('hashlib').sha256(raw).hexdigest()}
    (out/'00_R56B_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'01_GENERATIONS.txt').write_text('\n\n'.join(f"[{x['mode']}] {x['prompt']}\n{x['text']}\nMETRICS: {json.dumps({k:x[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','unique_word_trigram_ratio','replacement_chars')},ensure_ascii=False)}" for x in samples),encoding='utf-8')
    torch.save({'state_dict':m.state_dict(),'vocab':V,'spec':r56.SPECS[V],'seed':seed},out/f'R56B_3M_UNIGRAM{V}.pt')
    print(json.dumps({'vocab':V,'params':total,'embedding_fraction':emb/total,'train_MB':tm['train_source_bytes']/1048576,'eval_bpb':{k:v['bpb'] for k,v in ev.items()},'runtime_B_s':rt['source_byte_s']},ensure_ascii=False,indent=2),flush=True)
    for x in samples:print('\nGEN',x['mode'],x['prompt'],'\n',x['text'],flush=True)
if __name__=='__main__':main()
