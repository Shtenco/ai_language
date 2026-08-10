#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

import r57_concept_graph_language as base

VOCAB = 4096
D_MODEL = 192
HEADS = 6
ENC_LAYERS = 4
DEC_LAYERS = 6
FF_DIM = 570
GRAPH_TOKENS = 32
CONTEXT_TOKENS = 48
TARGET_TOKENS = 48
BATCH = 8
SEED = 20260810
RELATIONS = ['продолжение','причина','следствие','условие','цель','время','речь','определение','вопрос','logic','cyber']
REL2ID = {x:i for i,x in enumerate(RELATIONS)}

PROMPTS = [
    'Наука развивается потому, что',
    'Причинная связь отличается от корреляции тем, что',
    'Если эксперимент не подтвердил гипотезу, исследователь должен',
    'Память помогает рассуждению, потому что',
    'Когда программа обнаружила противоречие, она',
    'Чтобы проверить утверждение, необходимо',
    'Современная компьютерная система обрабатывает данные и',
    'После ошибки система изменила своё состояние и',
    'Москва — это город, в котором',
    'Человек посмотрел в окно и сказал:',
    'Искусственный интеллект может помочь человеку',
    'В городе открыли новый научный центр, где',
]
CYR_RE = re.compile(r'[А-Яа-яЁё]')
WORD_RE = re.compile(r'[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z-]{1,}', re.UNICODE)

class FixedSP:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        assert self.sp.get_piece_size() == VOCAB
    def enc(self, data):
        text = data.decode('utf-8') if isinstance(data, bytes) else data
        return list(self.sp.encode(text, out_type=int))
    def dec(self, ids):
        return self.sp.decode([int(x) for x in ids]).encode('utf-8')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def rope(q, k):
    dh = q.shape[-1]
    pos = torch.arange(q.shape[-2], device=q.device, dtype=q.dtype)
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=q.device, dtype=q.dtype) / dh))
    ang = torch.outer(pos, inv)
    cos = ang.cos()[None, None, :, :]
    sin = ang.sin()[None, None, :, :]
    def rot(x):
        xe, xo = x[..., 0::2], x[..., 1::2]
        y = torch.empty_like(x)
        y[..., 0::2] = xe * cos - xo * sin
        y[..., 1::2] = xe * sin + xo * cos
        return y
    return rot(q), rot(k)


class SelfBlock(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, FF_DIM)
        self.fc2 = nn.Linear(FF_DIM, D_MODEL)
    def self_attend(self, x):
        b,l,d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b,l,3,HEADS,d//HEADS).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        q,k = rope(q,k)
        a = F.scaled_dot_product_attention(q,k,v,is_causal=self.causal)
        return x + self.proj(a.transpose(1,2).contiguous().view(b,l,d))
    def ffn(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
    def forward(self, x):
        return self.ffn(self.self_attend(x))


class DecBlock(SelfBlock):
    def __init__(self):
        super().__init__(causal=True)
        self.cross_ln = nn.LayerNorm(D_MODEL)
        self.cross_q = nn.Linear(D_MODEL, D_MODEL)
        self.cross_kv = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.cross_out = nn.Linear(D_MODEL, D_MODEL)
    def cross_attend(self, x, mem):
        b,l,d = x.shape
        m = mem.shape[1]
        q = self.cross_q(self.cross_ln(x)).view(b,l,HEADS,d//HEADS).transpose(1,2)
        kv = self.cross_kv(mem).view(b,m,2,HEADS,d//HEADS).permute(2,0,3,1,4)
        k,v = kv[0],kv[1]
        a = F.scaled_dot_product_attention(q,k,v,is_causal=False)
        return x + self.cross_out(a.transpose(1,2).contiguous().view(b,l,d))
    def forward(self, x, mem):
        x = self.self_attend(x)
        x = self.cross_attend(x, mem)
        return self.ffn(x)


class NEXUS515(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.enc_blocks = nn.ModuleList([SelfBlock(causal=False) for _ in range(ENC_LAYERS)])
        self.enc_norm = nn.LayerNorm(D_MODEL)
        self.dec_blocks = nn.ModuleList([DecBlock() for _ in range(DEC_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.emb.weight
        self.plan_proj = nn.Linear(D_MODEL, D_MODEL)
        self.rel_head = nn.Linear(D_MODEL, len(RELATIONS))
    def encode(self, enc_ids):
        x = self.emb(enc_ids)
        for b in self.enc_blocks:
            x = b(x)
        return self.enc_norm(x)
    def decode(self, dec_ids, mem):
        x = self.emb(dec_ids)
        for b in self.dec_blocks:
            x = b(x, mem)
        h = self.norm(x)
        return self.head(h), h
    def forward(self, enc_ids, dec_ids):
        mem = self.encode(enc_ids)
        logits, h = self.decode(dec_ids, mem)
        return logits, h, mem


def warm_start(model, checkpoint):
    src = torch.load(checkpoint, map_location='cpu')['state_dict']
    own = model.state_dict()
    copied = []
    if 'emb.weight' in src and own['emb.weight'].shape == src['emb.weight'].shape:
        own['emb.weight'].copy_(src['emb.weight']); copied.append('emb.weight')
    fields = ('ln1.weight','ln1.bias','qkv.weight','qkv.bias','proj.weight','proj.bias','ln2.weight','ln2.bias','fc1.weight','fc1.bias','fc2.weight','fc2.bias')
    for i in range(DEC_LAYERS):
        for sub in fields:
            sk = f'blocks.{i}.{sub}'; dk = f'dec_blocks.{i}.{sub}'
            if sk in src and own[dk].shape == src[sk].shape:
                own[dk].copy_(src[sk]); copied.append(dk)
    for i in range(ENC_LAYERS):
        for sub in fields:
            sk = f'blocks.{i}.{sub}'; dk = f'enc_blocks.{i}.{sub}'
            if sk in src and own[dk].shape == src[sk].shape:
                own[dk].copy_(src[sk]); copied.append(dk)
    for sub in ('weight','bias'):
        sk=f'norm.{sub}'
        if sk in src:
            own[f'norm.{sub}'].copy_(src[sk]); copied.append(f'norm.{sub}')
            own[f'enc_norm.{sub}'].copy_(src[sk]); copied.append(f'enc_norm.{sub}')
    model.load_state_dict(own)
    return copied


def relation_id(ex):
    kind = ex['meta'].get('kind','main')
    if kind == 'logic': return REL2ID['logic']
    if kind == 'cyber': return REL2ID['cyber']
    rel,_ = base.infer_relation_goal(ex['ctx_text'])
    return REL2ID.get(rel, REL2ID['продолжение'])


def pad_left(ids, n, pad=0):
    ids = list(ids)[-n:]
    return [pad]*(n-len(ids)) + ids


def pad_right(ids, n, pad=0):
    ids = list(ids)[:n]
    return ids + [pad]*(n-len(ids))


def graph_ids(tok, memory, ex, mode='true', other=None):
    if mode == 'null':
        return [0]*GRAPH_TOKENS
    src = other if (mode == 'shuffled' and other is not None) else ex
    return base.prefix_ids('D_LOGIC_CYBER', tok, src['ctx_text'], memory, src['meta'])


def batchify(examples, tok, memory, graph_mode='true'):
    enc=[]; dec=[]; tgt=[]; mask=[]; rel=[]; byte_counts=[]
    n=len(examples)
    for i,ex in enumerate(examples):
        other=examples[(i+1)%n]
        gi=graph_ids(tok,memory,ex,graph_mode,other)
        ci=pad_left(ex['ctx'],CONTEXT_TOKENS)
        ti=list(ex['tgt'])[:TARGET_TOKENS]
        enc.append(gi+ci)
        dec.append([0]+pad_right(ti[:-1],TARGET_TOKENS-1))
        tgt.append(pad_right(ti,TARGET_TOKENS))
        mask.append([1]*len(ti)+[0]*(TARGET_TOKENS-len(ti)))
        rel.append(relation_id(ex))
        byte_counts.append(len(tok.dec(ti)))
    return (torch.tensor(enc,dtype=torch.long),torch.tensor(dec,dtype=torch.long),torch.tensor(tgt,dtype=torch.long),torch.tensor(mask,dtype=torch.float32),torch.tensor(rel,dtype=torch.long),byte_counts)


def sample_example(rng, docs, tok):
    u=rng.random()
    if u<0.94: return base.make_main_example(rng,docs,tok)
    if u<0.97: return base.make_aux_example('logic',rng,tok,train=True)
    return base.make_aux_example('cyber',rng,tok,train=True)


def train(model,tok,docs,memory,steps,batch,seed):
    set_seed(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.01)
    rng=random.Random(seed+101)
    losses=[]; plan_losses=[]; rel_losses=[]; target_bytes=0
    t0=time.perf_counter();model.train()
    for step in range(steps):
        exs=[sample_example(rng,docs,tok) for _ in range(batch)]
        enc,dec,tgt,mask,rel,bcs=batchify(exs,tok,memory,'true')
        logits,h,mem=model(enc,dec)
        ce=F.cross_entropy(logits.view(-1,VOCAB),tgt.view(-1),reduction='none').view(batch,TARGET_TOKENS)
        token_loss=(ce*mask).sum()/mask.sum().clamp_min(1)
        pooled=mem[:,GRAPH_TOKENS:,:].mean(dim=1)
        pred=F.normalize(model.plan_proj(pooled),dim=-1)
        with torch.no_grad():
            te=model.emb(tgt)
            target_vec=(te*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
            target_vec=F.normalize(target_vec,dim=-1)
        sim=pred@target_vec.T/0.08
        plan_loss=F.cross_entropy(sim,torch.arange(batch))
        rel_loss=F.cross_entropy(model.rel_head(pooled),rel)
        loss=token_loss+0.08*plan_loss+0.04*rel_loss
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        losses.append(float(token_loss));plan_losses.append(float(plan_loss));rel_losses.append(float(rel_loss));target_bytes+=sum(bcs)
        if (step+1)%512==0:
            print(json.dumps({'step':step+1,'token_ce':sum(losses[-64:])/len(losses[-64:]),'plan':sum(plan_losses[-64:])/len(plan_losses[-64:]),'rel':sum(rel_losses[-64:])/len(rel_losses[-64:]),'target_bytes':target_bytes},ensure_ascii=False),flush=True)
    return {'steps':steps,'batch':batch,'train_s':time.perf_counter()-t0,'target_bytes':target_bytes,'last64_token_ce':sum(losses[-64:])/len(losses[-64:]),'last64_plan_loss':sum(plan_losses[-64:])/len(plan_losses[-64:]),'last64_rel_loss':sum(rel_losses[-64:])/len(rel_losses[-64:])}


def eval_examples(model,examples,tok,memory,graph_mode='true'):
    model.eval();nll=0.0;bytes_total=0;correct=0;tokens=0
    with torch.no_grad():
        for s in range(0,len(examples),16):
            exs=examples[s:s+16]
            enc,dec,tgt,mask,rel,bcs=batchify(exs,tok,memory,graph_mode)
            logits,_,_=model(enc,dec)
            ce=F.cross_entropy(logits.view(-1,VOCAB),tgt.view(-1),reduction='none').view(len(exs),TARGET_TOKENS)
            nll+=float((ce*mask).sum());bytes_total+=sum(bcs)
            pred=logits.argmax(-1);correct+=int(((pred==tgt)*mask.bool()).sum());tokens+=int(mask.sum())
    return {'bpb':nll/max(1,bytes_total)/math.log(2),'nats_per_byte':nll/max(1,bytes_total),'token_top1':correct/max(1,tokens),'target_bytes':bytes_total,'target_tokens':tokens}


def filter_topk(logits,k=40):
    if k<=0 or k>=logits.numel(): return logits
    v,_=torch.topk(logits,k);thr=v[-1]
    return torch.where(logits>=thr,logits,torch.full_like(logits,float('-inf')))


def generate(model,tok,memory,prompt,decode='greedy',seed=0,max_new=48,graph_mode='true'):
    ctx=tok.enc(prompt.encode('utf-8'))[-CONTEXT_TOKENS:]
    ex={'ctx':ctx,'tgt':[],'ctx_text':prompt,'meta':{'kind':'main'}}
    gi=graph_ids(tok,memory,ex,graph_mode,None)
    enc=torch.tensor([gi+pad_left(ctx,CONTEXT_TOKENS)],dtype=torch.long)
    model.eval();out=[];gen=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mem=model.encode(enc)
        for _ in range(max_new):
            di=([0]+out)[-TARGET_TOKENS:]
            dec=torch.tensor([di],dtype=torch.long)
            logits,_=model.decode(dec,mem);z=logits[0,-1]
            if decode=='greedy': nxt=int(z.argmax())
            else:
                z=filter_topk(z/0.85,40);p=F.softmax(z,dim=-1);nxt=int(torch.multinomial(p,1,generator=gen))
            out.append(nxt)
            if len(out)>=TARGET_TOKENS: break
    return tok.dec(out).decode('utf-8','replace')


def gen_metrics(text):
    words=[w.lower() for w in WORD_RE.findall(text)]
    letters=[c for c in text if c.isalpha()]
    cyr=sum(1 for c in letters if CYR_RE.match(c))
    tri=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    rep=1-len(set(tri))/len(tri) if tri else 0.0
    return {'chars':len(text),'words':len(words),'cyrillic_letter_share':cyr/max(1,len(letters)),'unique_word_ratio':len(set(words))/max(1,len(words)),'repeated_trigram_rate':rep,'dash_per_100_chars':100*text.count('—')/max(1,len(text)),'replacement_chars':text.count('�')}


def exact_aux_accuracy(model,tok,memory,kind,seed,n=64):
    rng=random.Random(seed);rows=[];ok=0
    for i in range(n):
        ex=base.make_aux_example(kind,rng,tok,train=False if kind=='logic' else True)
        target=list(ex['tgt'])[:TARGET_TOKENS]
        gi=graph_ids(tok,memory,ex,'true')
        enc=torch.tensor([gi+pad_left(ex['ctx'],CONTEXT_TOKENS)],dtype=torch.long)
        out=[];model.eval()
        with torch.no_grad():
            mem=model.encode(enc)
            for _ in range(len(target)):
                dec=torch.tensor([[0]+out],dtype=torch.long)
                logits,_=model.decode(dec,mem);out.append(int(logits[0,-1].argmax()))
        good=(out==target);ok+=int(good)
        if i<16: rows.append({'target':tok.dec(target).decode('utf-8','replace'),'generated':tok.dec(out).decode('utf-8','replace'),'ok':good})
    return {'accuracy':ok/n,'n':n,'rows':rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-docs',required=True);ap.add_argument('--memory-docs',required=True);ap.add_argument('--tests',nargs='+',required=True)
    ap.add_argument('--tokenizer-model',required=True);ap.add_argument('--warm-checkpoint',required=True);ap.add_argument('--steps',type=int,default=8192);ap.add_argument('--out',required=True);ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();torch.set_num_threads(a.threads)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);shutil.copy2(a.tokenizer_model,out/'sp_unigram_4096.model')
    tok=FixedSP(a.tokenizer_model);probe='NEXUS 5.15 проверяет русский язык №515.'.encode();assert tok.dec(tok.enc(probe))==probe
    docs=base.tokenize_docs(tok,base.load_jsonl(a.train_docs));memory=base.MemoryIndex(base.load_jsonl(a.memory_docs));assert len(docs)>=1000 and len(memory.chunks)>=500
    model=NEXUS515();copied=warm_start(model,a.warm_checkpoint);params=param_count(model);assert params==5404367,params
    print(json.dumps({'params':params,'warm_tensors':len(copied),'docs':len(docs),'memory_chunks':len(memory.chunks)},ensure_ascii=False),flush=True)
    tr=train(model,tok,docs,memory,a.steps,BATCH,SEED)
    tests={Path(p).stem:Path(p).read_bytes() for p in a.tests};ev={}
    for i,(name,b) in enumerate(tests.items()):
        xs=base.build_main_eval(tok,b,7000+i*31,n=256)
        ev[name]={'true_graph':eval_examples(model,xs,tok,memory,'true'),'null_graph':eval_examples(model,xs,tok,memory,'null'),'shuffled_graph':eval_examples(model,xs,tok,memory,'shuffled')}
    logic=base.build_aux_eval(tok,'logic',88001,n=192);cyber=base.build_aux_eval(tok,'cyber',99001,n=192)
    ev['logic_teacher_forced']=eval_examples(model,logic,tok,memory,'true');ev['cyber_teacher_forced']=eval_examples(model,cyber,tok,memory,'true')
    gens=[]
    for i,p in enumerate(PROMPTS):
        for decode in ('greedy','sample'):
            txt=generate(model,tok,memory,p,decode,SEED+1000+i,max_new=48,graph_mode='true');gens.append({'prompt':p,'decode':decode,'continuation':txt,**gen_metrics(txt)})
    logic_acc=exact_aux_accuracy(model,tok,memory,'logic',123451,64);cyber_acc=exact_aux_accuracy(model,tok,memory,'cyber',223451,64)
    result={'format':'nexus-r515-semantic-encoder-decoder/1','protocol':{'params':params,'tokenizer':'frozen R5.12 Unigram4096','warm_start':'R5.12 32K decoder/embedding + first 4 blocks into semantic encoder','encoder':'4-layer bidirectional 192d','decoder':'6-layer causal 192d with cross-attention at every layer','graph_state':'D_LOGIC_CYBER concept graph + disjoint retrieval memory encoded outside decoder history','aux_objectives':'batch InfoNCE future-state + relation classification','steps':a.steps,'batch':BATCH,'seed':SEED},'training':tr,'eval':ev,'generation':gens,'logic_generation':logic_acc,'cyber_generation':cyber_acc,'warm_tensors':len(copied)}
    (out/'00_R515_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    for g in gens: lines.append(f"[{g['decode']}] {g['prompt']}\n{g['continuation']}\nMETRICS {json.dumps({k:g[k] for k in ('words','cyrillic_letter_share','unique_word_ratio','repeated_trigram_rate','dash_per_100_chars','replacement_chars')},ensure_ascii=False)}")
    lines.append(f"LOGIC={logic_acc['accuracy']:.6f} CYBER={cyber_acc['accuracy']:.6f}")
    (out/'01_RAW_GENERATIONS.txt').write_text('\n\n'.join(lines),encoding='utf-8')
    torch.save({'state_dict':model.state_dict(),'protocol':result['protocol']},out/'R515_SEMANTIC_ENCODER_DECODER.pt')
    print(json.dumps({'params':params,'train':tr,'eval_true':{k:(v['true_graph']['bpb'] if isinstance(v,dict) and 'true_graph' in v else v['bpb']) for k,v in ev.items()},'logic':logic_acc['accuracy'],'cyber':cyber_acc['accuracy']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
