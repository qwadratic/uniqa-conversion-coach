"""
names_sizes.py — train 3 GPT sizes on names, probe what each learns.

Knowledge tested per model:
  A. Bigram accuracy         — does it know what follows what? (simple)
  B. Length distribution     — does it know typical name lengths?
  C. Long-name rules         — names >8 chars: different phonology? (hard)
  D. Attention range         — does bigger model attend further back?
  E. Completion coherence    — given first 4 chars, how good is rest?
"""
from __future__ import annotations
import math, random, time, collections
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(42); random.seed(42)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── data ─────────────────────────────────────────────────────────────────────
names = [l.strip().lower() for l in open(Path(__file__).parent/"input.txt") if l.strip()]
random.shuffle(names)
SPLIT = int(0.9*len(names))
train_names, val_names = names[:SPLIT], names[SPLIT:]
train_set = set(train_names)
CHARS = sorted(set("".join(names)))
BOS = len(CHARS); VOCAB = len(CHARS)+1
c2i = {c:i for i,c in enumerate(CHARS)}; i2c = CHARS

def encode(n): return [BOS]+[c2i[c] for c in n]+[BOS]

flat_tr = [t for n in train_names for t in encode(n)]
flat_vl = [t for n in val_names   for t in encode(n)]
BLOCK=16; BATCH=256

def get_batch(split, device=DEVICE):
    data = torch.tensor(flat_tr if split=="train" else flat_vl, dtype=torch.long)
    ix = torch.randint(len(data)-BLOCK, (BATCH,))
    x = torch.stack([data[i:i+BLOCK]   for i in ix]).to(device)
    y = torch.stack([data[i+1:i+BLOCK+1] for i in ix]).to(device)
    return x, y

# ── model ─────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self,d): super().__init__(); self.w=nn.Parameter(torch.ones(d))
    def forward(self,x): return self.w*x*x.pow(2).mean(-1,keepdim=True).add(1e-5).rsqrt()

class Head(nn.Module):
    def __init__(self,d,h,T):
        super().__init__()
        self.h,self.hd=h,d//h
        self.qkv=nn.Linear(d,3*d,bias=False); self.proj=nn.Linear(d,d,bias=False)
        self.register_buffer("mask",torch.tril(torch.ones(T,T)).view(1,1,T,T))
        self.saved_attn=None
    def forward(self,x):
        B,T,C=x.shape
        q,k,v=self.qkv(x).split(C,dim=2)
        q=q.view(B,T,self.h,self.hd).transpose(1,2)
        k=k.view(B,T,self.h,self.hd).transpose(1,2)
        v=v.view(B,T,self.h,self.hd).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(self.hd)
        a=a.masked_fill(self.mask[:,:,:T,:T]==0,float("-inf"))
        a=F.softmax(a,-1); self.saved_attn=a.detach().cpu()
        return self.proj((a@v).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self,d,h,T):
        super().__init__()
        self.ln1=RMSNorm(d); self.attn=Head(d,h,T)
        self.ln2=RMSNorm(d)
        self.fc1=nn.Linear(d,4*d,bias=False); self.fc2=nn.Linear(4*d,d,bias=False)
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        return x+self.fc2(F.relu(self.fc1(self.ln2(x))))

class GPT(nn.Module):
    def __init__(self,d,h,L):
        super().__init__()
        self.d=d; self.h_per_layer=h; self.n_layer=L
        self.wte=nn.Embedding(VOCAB,d); self.wpe=nn.Embedding(BLOCK,d)
        self.blocks=nn.ModuleList([Block(d,h,BLOCK) for _ in range(L)])
        self.ln_f=RMSNorm(d); self.head=nn.Linear(d,VOCAB,bias=False)
        self.wte.weight=self.head.weight
        self.apply(lambda m: nn.init.normal_(m.weight,0,0.02)
                   if isinstance(m,(nn.Linear,nn.Embedding)) else None)
    def forward(self,idx,targets=None):
        B,t=idx.shape
        x=self.wte(idx)+self.wpe(torch.arange(t,device=idx.device))
        for b in self.blocks: x=b(x)
        logits=self.head(self.ln_f(x))
        loss=F.cross_entropy(logits.view(-1,VOCAB),targets.view(-1)) if targets is not None else None
        return logits,loss
    def gen(self,temp=0.8,top_k=10,prefix=None,max_len=20):
        self.eval()
        start=[[BOS]] if prefix is None else [[BOS]+[c2i[c] for c in prefix if c in c2i]]
        idx=torch.tensor(start,dtype=torch.long)
        for sn in range(max_len):
            lo,_=self(idx[:,-BLOCK:]); lo=lo[:,-1,:]/temp
            if sn==0 and prefix is None: lo[:,BOS]=float("-inf")
            v,_=torch.topk(lo,min(top_k,VOCAB)); lo[lo<v[:,[-1]]]=float("-inf")
            nxt=torch.multinomial(F.softmax(lo,-1),1)
            if sn>0 and nxt.item()==BOS: break
            if nxt.item()!=BOS: idx=torch.cat([idx,nxt],dim=1)
        return "".join(i2c[i] for i in idx[0].tolist() if i!=BOS)

def train(model, iters, lr=3e-3, device=DEVICE):
    model.to(device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.99),weight_decay=0.1)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iters,eta_min=3e-4)
    for step in range(iters):
        xb,yb=get_batch("train",device)
        _,loss=model(xb,yb); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step()
    return model

@torch.no_grad()
def val_loss(model, device=DEVICE, n=40):
    model.eval()
    v=sum(model(*get_batch("val",device))[1].item() for _ in range(n))/n
    model.train(); return v

# ── probes ────────────────────────────────────────────────────────────────────
def bigram_accuracy(model):
    """Top-1 next-char accuracy on val names using model (vs bigram baseline)."""
    # build bigram table
    bg = collections.defaultdict(collections.Counter)
    for n in train_names:
        for a,b in zip(encode(n),encode(n)[1:]): bg[a][b]+=1
    def bg_pred(ctx): return max(bg.get(ctx[-1],{BOS:1}),key=bg.get(ctx[-1],{BOS:1}).get)
    def gpt_pred(ctx):
        idx=torch.tensor([ctx[-BLOCK:]],dtype=torch.long)
        lo,_=model(idx); return lo[0,-1].argmax().item()
    model.eval(); model.cpu()
    bg_ok=gpt_ok=total=0
    for name in val_names[:200]:
        toks=encode(name)
        for i in range(len(toks)-1):
            ctx=toks[:i+1]; target=toks[i+1]
            bg_ok  +=(bg_pred(ctx)==target); gpt_ok+=(gpt_pred(ctx)==target); total+=1
    return gpt_ok/total, bg_ok/total

def length_calibration(model, n=200):
    """KL between generated length dist and training length dist."""
    gen_names=[model.gen(temp=0.8) for _ in range(n)]
    gen_names=[x for x in gen_names if x]
    tr_lens=collections.Counter(len(x) for x in train_names)
    gn_lens=collections.Counter(len(x) for x in gen_names)
    all_lens=sorted(set(tr_lens)|set(gn_lens))
    tr_tot=sum(tr_lens.values()); gn_tot=max(sum(gn_lens.values()),1)
    kl=0.0
    for l in all_lens:
        p=tr_lens.get(l,0)/tr_tot; q=gn_lens.get(l,1e-9)/gn_tot
        if p>0: kl+=p*math.log(p/q)
    mean_tr=sum(l*c for l,c in tr_lens.items())/tr_tot
    mean_gn=sum(l*c for l,c in gn_lens.items())/gn_tot
    return kl, mean_tr, mean_gn

def long_name_probe(model, n_gen=300):
    """Do generated long names (>8) have proper internal structure?
    Measures: prefix-suffix vowel balance, repeat-char rate, valid-completion rate."""
    all_gen=[model.gen(temp=0.9,top_k=20) for _ in range(n_gen)]
    long=[x for x in all_gen if len(x)>8]
    short=[x for x in all_gen if 4<=len(x)<=6]
    def vowel_rate(names): return sum(c in "aeiou" for n in names for c in n)/max(sum(len(n) for n in names),1)
    def repeat_rate(names): return sum(len(n)-len(set(n)) for n in names)/max(sum(len(n) for n in names),1)
    def alternation(names):
        "fraction of adjacent pairs that alternate vowel/consonant"
        pairs=[(a in "aeiou",b in "aeiou") for n in names for a,b in zip(n,n[1:])]
        return sum(1 for a,b in pairs if a!=b)/max(len(pairs),1)
    vr_long=vowel_rate(long); vr_short=vowel_rate(short)
    alt_long=alternation(long); alt_short=alternation(short)
    rr_long=repeat_rate(long); rr_short=repeat_rate(short)
    return {
        "n_long":len(long),"n_short":len(short),
        "vowel_rate_long":vr_long,"vowel_rate_short":vr_short,
        "alternation_long":alt_long,"alternation_short":alt_short,
        "repeat_rate_long":rr_long,"repeat_rate_short":rr_short,
        "long_examples":long[:6],"short_examples":short[:4],
    }

def attention_range(model, n_names=60):
    """Average attention distance per head per layer. Higher = longer-range."""
    samples=[model.gen(temp=0.8) for _ in range(n_names)]
    samples=[s for s in samples if len(s)>=4][:50]
    model.eval().cpu()
    results={}  # (layer,head) -> list of avg offsets
    for name in samples:
        toks=torch.tensor([encode(name)[:-1]],dtype=torch.long)
        with torch.no_grad(): model(toks)
        for L,block in enumerate(model.blocks):
            attn=block.attn.saved_attn  # (1,n_head,T,T)
            if attn is None: continue
            T=attn.shape[-1]
            for h in range(model.h_per_layer):
                # for each query pos, expected attended distance
                for q in range(T):
                    dist=sum(attn[0,h,q,k].item()*(q-k) for k in range(q+1))
                    results.setdefault((L,h),[]).append(dist)
    return {k:sum(v)/len(v) for k,v in results.items()}

def completion_coherence(model, prefix_len=4, n=100):
    """Given first `prefix_len` chars of a real val name, complete it.
    Score: fraction of completions that are real names OR phonologically valid."""
    score=0
    for name in random.sample(val_names, min(n,len(val_names))):
        if len(name)<prefix_len+2: continue
        prefix=name[:prefix_len]; completed=model.gen(temp=0.7,prefix=prefix)
        # count as "good" if: in any name list OR ends like training names (ends with vowel common)
        if completed in train_set or completed in set(val_names): score+=1
        elif completed and completed[-1] in "aeiou": score+=0.5  # soft credit
    return score/n

# ══════════════════════════════════════════════════════════════════════════════
# DEFINE 3 MODEL SIZES
# ══════════════════════════════════════════════════════════════════════════════
SIZES = [
    dict(name="Small",  d=16,  h=2, L=3, iters=5000, lr=3e-3),   #   ~9K
    dict(name="Medium", d=64,  h=4, L=4, iters=5000, lr=3e-3),   # ~200K
    dict(name="Large",  d=128, h=8, L=6, iters=5000, lr=1e-3),   #  ~1.2M
]

all_results = []

for cfg in SIZES:
    nm,d,h,L,iters,lr = cfg["name"],cfg["d"],cfg["h"],cfg["L"],cfg["iters"],cfg["lr"]
    model = GPT(d,h,L)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {nm} — {n_params/1e3:.1f}K params  d={d} h={h} L={L}  iters={iters}")
    print(f"{'='*60}")

    t0=time.time()
    train(model, iters, lr=lr, device=DEVICE)
    vl=val_loss(model, device=DEVICE)
    train_t=time.time()-t0
    print(f"  val_loss={vl:.4f}  ppl={math.exp(vl):.3f}  train={train_t:.0f}s")

    model=model.cpu().eval()

    # A. bigram accuracy
    gpt_acc, bg_acc = bigram_accuracy(model)
    print(f"\n  A. Bigram accuracy:  GPT={gpt_acc*100:.1f}%  Bigram={bg_acc*100:.1f}%  Δ=+{(gpt_acc-bg_acc)*100:.1f}pp")

    # B. length calibration
    kl, mean_tr, mean_gn = length_calibration(model)
    print(f"  B. Length KL={kl:.4f}  train_mean={mean_tr:.2f}  gen_mean={mean_gn:.2f}")

    # C. long-name probe
    lp = long_name_probe(model)
    print(f"  C. Long-name probe ({lp['n_long']} long / {lp['n_short']} short):")
    print(f"     vowel_rate:  long={lp['vowel_rate_long']:.3f}  short={lp['vowel_rate_short']:.3f}  (train≈0.38)")
    print(f"     alternation: long={lp['alternation_long']:.3f}  short={lp['alternation_short']:.3f}")
    print(f"     repeat_rate: long={lp['repeat_rate_long']:.3f}  short={lp['repeat_rate_short']:.3f}")
    print(f"     long examples:  {lp['long_examples']}")
    print(f"     short examples: {lp['short_examples']}")

    # D. attention range
    ar = attention_range(model)
    print(f"  D. Attention range (avg distance, higher=longer):")
    for L_i in range(L):
        row="     L{}: ".format(L_i)
        for h_i in range(h):
            row+=f"H{h_i}={ar.get((L_i,h_i),0):.2f}  "
        print(row)
    all_avg = sum(ar.values())/max(len(ar),1)
    print(f"     Overall mean distance: {all_avg:.3f}")

    # E. completion coherence
    coh = completion_coherence(model)
    print(f"  E. Completion coherence (4-char prefix): {coh*100:.1f}%")

    # 5 generated names
    gen5=[model.gen(temp=0.7) for _ in range(8)]
    gen5=[g for g in gen5 if g][:5]
    print(f"\n  Generated (T=0.7): {' | '.join(gen5)}")

    all_results.append(dict(name=nm,n_params=n_params,val_loss=vl,
        gpt_acc=gpt_acc,bg_acc=bg_acc,kl=kl,mean_gn=mean_gn,
        attn_range=all_avg,coherence=coh,lp=lp))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n"+"="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Model':8s} {'Params':8s} {'ValLoss':8s} {'PPL':6s} {'Top1':6s} {'ΔBigram':8s} {'LenKL':7s} {'AttnDist':9s} {'LongVowel':10s}")
print("-"*70)
for r in all_results:
    print(f"{r['name']:8s} {r['n_params']/1e3:6.0f}K  "
          f"{r['val_loss']:7.4f}  {math.exp(r['val_loss']):5.3f}  "
          f"{r['gpt_acc']*100:5.1f}%  "
          f"+{(r['gpt_acc']-r['bg_acc'])*100:4.1f}pp  "
          f"{r['kl']:6.4f}  "
          f"{r['attn_range']:8.3f}  "
          f"{r['lp']['vowel_rate_long']:.3f}")
print("="*70)

print("\nKEY QUESTIONS:")
for r in all_results:
    lp=r['lp']
    long_diff=lp['vowel_rate_long']-lp['vowel_rate_short']
    print(f"  {r['name']:8s}: knows long-name rules? "
          f"vowel_rate_long vs short = {long_diff:+.3f}  "
          f"alt_diff={lp['alternation_long']-lp['alternation_short']:+.3f}  "
          f"→ {'YES' if abs(long_diff)>0.02 else 'WEAK'}")
