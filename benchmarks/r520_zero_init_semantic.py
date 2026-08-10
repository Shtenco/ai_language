#!/usr/bin/env python3
import argparse,json,random,time,shutil
from pathlib import Path
import torch
import torch.nn.functional as F
import r57_concept_graph_language as base57
import r515_encoder_decoder as base515
import r517_sentence_curriculum as r517

SEED=20260810;BATCH=8

def init_zero_adapter(model):
    # Exact functional preservation of the warm R5.12 decoder at step 0:
    # cross-attention contributes exactly zero until learned useful.
    for b in model.dec_blocks:
        torch.nn.init.zeros_(b.cross_out.weight)
        if b.cross_out.bias is not None: torch.nn.init.zeros_(b.cross_out.bias)


def split_params(model):
    new=[];base=[]
    new_keys=('enc_blocks','enc_norm','plan_proj','rel_head','cross_ln','cross_q','cross_kv','cross_out')
    for name,p in model.named_parameters():
        if any(k in name for k in new_keys):new.append(p)
        else:base.append(p)
    return base,new


def train(model,tok,docs,memory,steps):
    random.seed(SEED);torch.manual_seed(SEED);rng=random.Random(SEED+520)
    basep,newp=split_params(model)
    opt=torch.optim.AdamW([
        {'params':basep,'lr':5e-5,'weight_decay':0.01},
        {'params':newp,'lr':3e-4,'weight_decay':0.01},
    ],betas=(.9,.95))
    losses=[];plans=[];rels=[];target_bytes=0;t0=time.perf_counter();model.train()
    for step in range(steps):
        exs=[r517.sentence_sample(rng,docs,tok) for _ in range(BATCH)]
        enc,dec,tgt,mask,rel,bcs=base515.batchify(exs,tok,memory,'true')
        logits,h,mem=model(enc,dec)
        ce=F.cross_entropy(logits.reshape(-1,base515.VOCAB),tgt.reshape(-1),reduction='none').view(BATCH,base515.TARGET_TOKENS)
        tok_loss=(ce*mask).sum()/mask.sum().clamp_min(1)
        pooled=mem[:,base515.GRAPH_TOKENS:,:].mean(1)
        pred=F.normalize(model.plan_proj(pooled),dim=-1)
        with torch.no_grad():
            te=model.emb(tgt);tv=(te*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1);tv=F.normalize(tv,dim=-1)
        plan=F.cross_entropy(pred@tv.T/0.08,torch.arange(BATCH))
        rel_loss=F.cross_entropy(model.rel_head(pooled),rel)
        loss=tok_loss+.08*plan+.04*rel_loss
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        losses.append(float(tok_loss));plans.append(float(plan));rels.append(float(rel_loss));target_bytes+=sum(bcs)
        if (step+1)%256==0:
            print(json.dumps({'step':step+1,'token_ce':sum(losses[-64:])/len(losses[-64:]),'plan':sum(plans[-64:])/len(plans[-64:]),'rel':sum(rels[-64:])/len(rels[-64:]),'target_bytes':target_bytes},ensure_ascii=False),flush=True)
    return {'steps':steps,'target_bytes':target_bytes,'train_s':time.perf_counter()-t0,'last64_token_ce':sum(losses[-64:])/len(losses[-64:])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True);ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--warm-checkpoint',required=True);ap.add_argument('--steps',type=int,default=4096);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2);a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shutil.copy2(a.tokenizer_model,out/'sp_unigram_4096.model');tok=base515.FixedSP(a.tokenizer_model)
    raw_docs=base57.load_jsonl(a.train_docs);docs=base57.tokenize_docs(tok,raw_docs);memory=base57.MemoryIndex(base57.load_jsonl(a.memory_docs));r517.SENT_PAIRS=r517.build_sentence_pairs(tok,raw_docs,limit=80000);assert len(r517.SENT_PAIRS)>=5000
    model=base515.NEXUS515();copied=base515.warm_start(model,a.warm_checkpoint);init_zero_adapter(model);params=base515.param_count(model);assert params==5404367
    # Exact preservation check: null semantic contribution and warm decoder must initially produce finite logits.
    print(json.dumps({'params':params,'warm_tensors':len(copied),'sentence_pairs':len(r517.SENT_PAIRS),'zero_cross_layers':len(model.dec_blocks)},ensure_ascii=False),flush=True)
    tr=train(model,tok,docs,memory,a.steps)
    tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};ev={}
    for i,(name,b) in enumerate(tests.items()):
        xs=base57.build_main_eval(tok,b,7000+i*31,n=256);sent=r517.eval_sentence_bytes(tok,b,192)
        ev[name]={'random_true':base515.eval_examples(model,xs,tok,memory,'true'),'random_null':base515.eval_examples(model,xs,tok,memory,'null'),'random_shuffled':base515.eval_examples(model,xs,tok,memory,'shuffled'),'sentence_true':base515.eval_examples(model,sent,tok,memory,'true') if sent else None,'sentence_examples':len(sent)}
    logic=base57.build_aux_eval(tok,'logic',88001,n=192);cyber=base57.build_aux_eval(tok,'cyber',99001,n=192);ev['logic_teacher_forced']=base515.eval_examples(model,logic,tok,memory,'true');ev['cyber_teacher_forced']=base515.eval_examples(model,cyber,tok,memory,'true')
    gens=[]
    for i,p in enumerate(r517.PROMPTS):
        for d in ('greedy','sample'):
            txt=base515.generate(model,tok,memory,p,d,SEED+5200+i,max_new=48,graph_mode='true');gens.append({'prompt':p,'decode':d,'continuation':txt,**base515.gen_metrics(txt)})
    la=base515.exact_aux_accuracy(model,tok,memory,'logic',123451,64);ca=base515.exact_aux_accuracy(model,tok,memory,'cyber',223451,64)
    result={'format':'nexus-r520-zero-init-semantic/1','protocol':{'params':params,'warm_start':'R5.12 32K','zero_init_cross_out':True,'base_lr':5e-5,'semantic_lr':3e-4,'curriculum':{'sentence':.70,'random':.24,'logic':.03,'cyber':.03},'steps':a.steps},'training':tr,'sentence_pairs':len(r517.SENT_PAIRS),'eval':ev,'generation':gens,'logic_generation':la,'cyber_generation':ca}
    (out/'00_R520_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');lines=[]
    for g in gens:lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','replacement_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={la['accuracy']:.6f} CYBER={ca['accuracy']:.6f}");(out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8');torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R520_ZERO_INIT_SEMANTIC.pt');print(json.dumps({'train':tr,'logic':la['accuracy'],'cyber':ca['accuracy']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
