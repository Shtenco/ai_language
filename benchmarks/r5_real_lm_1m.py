#!/usr/bin/env python3
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB=256; D_MODEL=195; N_HEAD=5; FF_DIM=789; N_LAYER=2; MAX_CTX=128
assert D_MODEL%N_HEAD==0

def set_seed(s): random.seed(s); torch.manual_seed(s)
def byte_tokens(p): return torch.tensor(list(Path(p).read_bytes()),dtype=torch.long)

def static_sparse_mask(L,window=16,shortcuts=4):
    m=torch.zeros(L,L,dtype=torch.bool)
    for i in range(L):
        m[i,max(0,i-window+1):i+1]=True
        if i>0:
            x=(i+1)*0x9E3779B1
            for s in range(shortcuts):
                x=(1664525*x+1013904223+s*97)&0xFFFFFFFF
                m[i,x%(i+1)]=True
        m[i,0]=True
    return m

def token_wormhole_mask(tokens,base):
    B,L=tokens.shape
    same=tokens[:,:,None].eq(tokens[:,None,:])
    causal=torch.tril(torch.ones(L,L,dtype=torch.bool,device=tokens.device))
    t=tokens; cls=torch.full_like(t,4)
    cls[(t>=65)&(t<=90)]=0; cls[(t>=97)&(t<=122)]=0; cls[(t>=48)&(t<=57)]=1; cls[t==32]=2
    ascii_other=(t<128)&~(((t>=65)&(t<=90))|((t>=97)&(t<=122))|((t>=48)&(t<=57))|(t==32))
    cls[ascii_other]=3
    same_cls=cls[:,:,None].eq(cls[:,None,:])
    idx=torch.arange(L,device=tokens.device)
    recent=(idx[None,:,None]-idx[None,None,:]<=12)&(idx[None,:,None]>=idx[None,None,:])
    dyn=(same|(same_cls&recent))&causal[None,:,:]
    return dyn|base.to(tokens.device)[None,:,:]

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.ln1=nn.LayerNorm(D_MODEL); self.qkv=nn.Linear(D_MODEL,3*D_MODEL); self.proj=nn.Linear(D_MODEL,D_MODEL); self.ln2=nn.LayerNorm(D_MODEL); self.fc1=nn.Linear(D_MODEL,FF_DIM); self.fc2=nn.Linear(FF_DIM,D_MODEL)

class LM(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind; self.tok=nn.Embedding(VOCAB,D_MODEL); self.pos=nn.Embedding(MAX_CTX,D_MODEL); self.blocks=nn.ModuleList([Block() for _ in range(N_LAYER)]); self.lnf=nn.LayerNorm(D_MODEL); self.head=nn.Linear(D_MODEL,VOCAB,bias=False); self.head.weight=self.tok.weight
    def make_mask(self,ids):
        B,L=ids.shape; causal=torch.tril(torch.ones(L,L,dtype=torch.bool,device=ids.device))
        if self.kind=='T0_DENSE': return causal[None,None,:,:]
        base=static_sparse_mask(L).to(ids.device)
        if self.kind=='R4_OMEGA': return base[None,None,:,:]
        dyn=token_wormhole_mask(ids,base)
        mh=dyn[:,None,:,:].expand(B,N_HEAD,L,L).clone(); mh[:,0,:,:]=causal
        return mh
    def route_ff(self,y,ids):
        if self.kind=='T0_DENSE': return y
        B,L,Fd=y.shape; group=torch.arange(Fd,device=y.device)*4//Fd; off=(ids%4)[:,:,None]
        return y*(group[None,None,:]!=off)
    def forward(self,ids):
        B,L=ids.shape; pos=torch.arange(L,device=ids.device); x=self.tok(ids)+self.pos(pos)[None,:,:]; mask=self.make_mask(ids)
        for b in self.blocks:
            H=N_HEAD; hd=D_MODEL//H; y=b.ln1(x); qkv=b.qkv(y).view(B,L,3,H,hd).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=False); a=a.transpose(1,2).contiguous().view(B,L,D_MODEL); x=x+b.proj(a)
            y=F.gelu(b.fc1(b.ln2(x))); y=self.route_ff(y,ids); x=x+b.fc2(y)
        return self.head(self.lnf(x))

def pc(m): return sum(p.numel() for p in m.parameters())
def make_plan(n,seed):
    g=random.Random(seed); plan=[]
    for seq,batch,steps in [(64,16,128),(128,8,128)]:
        for _ in range(steps): plan.append((seq,[g.randrange(0,n-seq-1) for _ in range(batch)]))
    return plan
def mkbatch(data,seq,starts): return torch.stack([data[s:s+seq] for s in starts]),torch.stack([data[s+1:s+seq+1] for s in starts])

def train_one(kind,seed,data):
    set_seed(seed); m=LM(kind); assert pc(m)==999978; opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=.01,betas=(.9,.95)); losses=[]; t0=time.perf_counter(); m.train()
    for seq,starts in make_plan(len(data),seed+10000):
        x,y=mkbatch(data,seq,starts); opt.zero_grad(set_to_none=True); z=m(x); loss=F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1)); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); losses.append(float(loss))
    return m,{'train_s':time.perf_counter()-t0,'train_loss_last32':sum(losses[-32:])/32,'train_tokens':262144}

@torch.no_grad()
def evaluate(m,data,seq,batch=8,n_batches=24,seed=1234):
    g=random.Random(seed+seq); m.eval(); nll=0.; correct=0; nt=0
    for _ in range(n_batches):
        starts=[g.randrange(0,len(data)-seq-1) for _ in range(batch)]; x,y=mkbatch(data,seq,starts); z=m(x); nll+=float(F.cross_entropy(z.reshape(-1,VOCAB),y.reshape(-1),reduction='sum')); correct+=int((z.argmax(-1)==y).sum()); nt+=y.numel()
    a=nll/nt; return {'nll':a,'ppl':math.exp(min(a,20)),'bpc':a/math.log(2),'top1':correct/nt,'tokens':nt}

@torch.no_grad()
def runtime(m,data,seq,batch=16,reps=20):
    starts=[i*17%(len(data)-seq-1) for i in range(batch)]; x,_=mkbatch(data,seq,starts); m.eval()
    for _ in range(4): m(x)
    vv=[]
    for _ in range(reps): t=time.perf_counter(); m(x); vv.append(time.perf_counter()-t)
    vv.sort(); med=vv[len(vv)//2]; return {'median_s':med,'tok_s':batch*seq/med}

def edge_fraction(kind,x):
    m=LM(kind).make_mask(x); B,L=x.shape; full=L*(L+1)/2*N_HEAD
    if m.shape[0]==1: m=m.expand(B,*m.shape[1:])
    if m.shape[1]==1: m=m.expand(B,N_HEAD,L,L)
    return float(m.float().sum((1,2,3)).mean()/full)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True); ap.add_argument('--valid',required=True); ap.add_argument('--test',required=True); ap.add_argument('--shift',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seeds',default='11,29,47'); ap.add_argument('--threads',type=int,default=2); a=ap.parse_args(); torch.set_num_threads(a.threads)
    tr=byte_tokens(a.train); va=byte_tokens(a.valid); te=byte_tokens(a.test); sh=byte_tokens(a.shift); kinds=['T0_DENSE','R4_OMEGA','R5_OMEGA']; seeds=[int(x) for x in a.seeds.split(',')]; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; details={'protocol':{'params':999978,'train_tokens':262144,'tokenizer':'raw UTF-8 bytes, vocab=256','contexts':[64,128],'seeds':seeds},'corpus_bytes':{'train':len(tr),'valid':len(va),'test':len(te),'shift':len(sh)},'runs':[]}
    for seed in seeds:
        for kind in kinds:
            print('RUN',seed,kind,flush=True); m,tm=train_one(kind,seed,tr); r={'seed':seed,'model':kind,'params':pc(m),**tm}
            for nm,data in [('valid',va),('test',te),('shift',sh)]:
                for seq in [64,128]:
                    ev=evaluate(m,data,seq,seed=seed+500)
                    for k,v in ev.items(): r[f'{nm}_{seq}_{k}']=v
            for seq in [64,128]:
                rt=runtime(m,te,seq); r[f'rt_{seq}_tok_s']=rt['tok_s']; r[f'rt_{seq}_median_s']=rt['median_s']
            x,_=mkbatch(te,128,[0,137,911,2048]); r['edge_fraction_128']=edge_fraction(kind,x); r['qat_test128']=r['test_128_top1']*r['rt_128_tok_s']; rows.append(r); details['runs'].append(r); torch.save(m.state_dict(),out/f'{kind}_seed{seed}.pt')
            print(json.dumps({k:r[k] for k in ['model','seed','test_64_ppl','test_128_ppl','shift_128_ppl','test_128_top1','rt_128_tok_s','qat_test128','edge_fraction_128']},indent=2),flush=True)
    agg=[]
    metrics=['test_64_nll','test_64_ppl','test_64_bpc','test_64_top1','test_128_nll','test_128_ppl','test_128_bpc','test_128_top1','valid_128_ppl','shift_128_ppl','shift_128_top1','rt_64_tok_s','rt_128_tok_s','qat_test128','edge_fraction_128','train_s']
    for kind in kinds:
        rs=[r for r in rows if r['model']==kind]; ar={'model':kind,'n_seeds':len(rs),'params':999978}
        for q in metrics:
            vals=[r[q] for r in rs]; mean=sum(vals)/len(vals); sd=(sum((v-mean)**2 for v in vals)/len(vals))**.5; ar[q+'_mean']=mean; ar[q+'_sd']=sd
        agg.append(ar)
    base=agg[0]
    for ar in agg:
        ar['ppl128_ratio_vs_t0']=ar['test_128_ppl_mean']/base['test_128_ppl_mean']; ar['top1_delta_pp_vs_t0']=(ar['test_128_top1_mean']-base['test_128_top1_mean'])*100; ar['tok_s_ratio_vs_t0']=ar['rt_128_tok_s_mean']/base['rt_128_tok_s_mean']; ar['qat_ratio_vs_t0']=ar['qat_test128_mean']/base['qat_test128_mean']
    details['aggregate']=agg; (out/'00_REAL_LM_RESULTS.json').write_text(json.dumps(details,indent=2))
    for fn,arr in [('01_PER_SEED.csv',rows),('02_AGGREGATE.csv',agg)]:
        keys=[]
        for d in arr:
            for k in d:
                if k not in keys: keys.append(k)
        with open(out/fn,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(arr)
    rd=['# NEXUS R5 — REAL LM 1M BENCHMARK','','WikiText-2 train/valid/test; domain shift: Tiny Shakespeare. Raw UTF-8 byte tokenizer (256 symbols).','All models: 999,978 trainable parameters; exactly 262,144 target tokens; identical data plan per seed.','','## Aggregate']
    for ar in agg: rd.append(f"- **{ar['model']}**: test128 PPL {ar['test_128_ppl_mean']:.3f} ± {ar['test_128_ppl_sd']:.3f}; top1 {100*ar['test_128_top1_mean']:.2f}%; shift128 PPL {ar['shift_128_ppl_mean']:.3f}; {ar['rt_128_tok_s_mean']:.0f} tok/s; QAT {ar['qat_test128_mean']:.1f}; edges {100*ar['edge_fraction_128_mean']:.1f}%")
    rd+=['','## Honesty boundary','This is a from-scratch ~1M byte-level real-corpus LM laboratory, not frontier scale. Exact Authority is excluded from perplexity and LM quality.']; (out/'README_RU.md').write_text('\n'.join(rd))
    hs=[]
    for p in sorted(out.iterdir()):
        if p.is_file(): hs.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}')
    (out/'SHA256SUMS.txt').write_text('\n'.join(hs)+'\n'); print('DONE',json.dumps(agg,indent=2),flush=True)
if __name__=='__main__': main()
