"""
names_gpt.py — PyTorch nanoGPT on names.txt (char-level, vocab=27).

Same architecture as infineon_gpt.py but for Karpathy's original names task.
Runs on MPS in ~1 min. Then explores:
  A. Temperature sweep    (0.2 / 0.5 / 0.8 / 1.2)
  B. Novelty analysis     (% generated names not in training set)
  C. Length distribution  (generated vs training)
  D. Bigram baseline      (Markov chain, same metrics)
  E. Loss curve snapshot  (quality at step 100/300/600/1000)

Usage:
  python names_gpt.py [--iters N] [--device mps|cpu]
"""
from __future__ import annotations
import argparse, math, random, time, collections
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",   default=str(Path(__file__).parent / "input.txt"))
parser.add_argument("--iters",  default=5000, type=int)
parser.add_argument("--device", default=("mps" if torch.backends.mps.is_available()
                                         else "cuda" if torch.cuda.is_available()
                                         else "cpu"))
parser.add_argument("--seed",   default=42, type=int)
args = parser.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed)
DEVICE = args.device
print(f"device={DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
names = [l.strip().lower() for l in open(args.data) if l.strip()]
random.shuffle(names)
train_names = names[:int(0.9 * len(names))]
val_names   = names[int(0.9 * len(names)):]
train_set   = set(train_names)

CHARS = sorted(set("".join(names)))
BOS   = len(CHARS)          # 26 (a-z only)
VOCAB = len(CHARS) + 1      # 27
c2i   = {c: i for i, c in enumerate(CHARS)}
i2c   = CHARS

print(f"names={len(names)}  vocab={VOCAB}  train={len(train_names)}  val={len(val_names)}")
print(f"chars: {''.join(CHARS)}  BOS={BOS}")

def encode(name: str) -> list[int]:
    return [BOS] + [c2i[c] for c in name] + [BOS]

def decode(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i == BOS and out:
            break
        if i != BOS:
            out.append(i2c[i])
    return "".join(out)

# Pack flat buffer
flat_train = []
for n in train_names:
    flat_train.extend(encode(n))
flat_val = []
for n in val_names:
    flat_val.extend(encode(n))

BLOCK = 16   # max name length ~15 chars
BATCH = 256  # small vocab → bigger batch fits fine

# Use separator token distinct from BOS so model doesn't confuse start vs end.
# Trick: use BOS only as START, use a separate EOS=BOS+1, adjust vocab.
# Simpler fix: pack without BOS-BOS adjacency — separate sequences with padding.
# Actually simplest: just pack names separated by a single newline token
# and only use BOS as START, stop generation at newline token.

def get_batch(split: str):
    data = torch.tensor(flat_train if split == "train" else flat_val, dtype=torch.long)
    ix = torch.randint(len(data) - BLOCK, (BATCH,))
    x = torch.stack([data[i:i+BLOCK]   for i in ix]).to(DEVICE)
    y = torch.stack([data[i+1:i+BLOCK+1] for i in ix]).to(DEVICE)
    return x, y

# ── Model (same as infineon_gpt.py, scaled for tiny vocab) ───────────────────
class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x): return self.w * x * x.pow(2).mean(-1, keepdim=True).add(1e-5).rsqrt()

class CausalAttn(nn.Module):
    def __init__(self, d, h, T):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv  = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1,1,T,T))
    def forward(self, x):
        B, T, C = x.shape
        q,k,v = self.qkv(x).split(C, dim=2)
        q = q.view(B,T,self.h,self.hd).transpose(1,2)
        k = k.view(B,T,self.h,self.hd).transpose(1,2)
        v = v.view(B,T,self.h,self.hd).transpose(1,2)
        a = (q @ k.transpose(-2,-1)) / math.sqrt(self.hd)
        a = a.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        return self.proj((F.softmax(a,-1) @ v).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self, d, h, T):
        super().__init__()
        self.ln1 = RMSNorm(d); self.attn = CausalAttn(d, h, T)
        self.ln2 = RMSNorm(d)
        self.fc1 = nn.Linear(d, 4*d, bias=False); self.fc2 = nn.Linear(4*d, d, bias=False)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))

class GPT(nn.Module):
    def __init__(self, V, T, d, h, L):
        super().__init__()
        self.wte = nn.Embedding(V, d); self.wpe = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h, T) for _ in range(L)])
        self.ln_f = RMSNorm(d); self.head = nn.Linear(d, V, bias=False)
        self.wte.weight = self.head.weight
        self.apply(lambda m: nn.init.normal_(m.weight, 0, 0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)
        self.T = T
    def forward(self, idx, targets=None):
        B, t = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    @torch.no_grad()
    def generate_name(self, temp=0.8, top_k=10, device="cpu") -> str:
        self.eval()
        idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
        for step_n in range(self.T - 1):
            logits, _ = self(idx[:, -self.T:])
            logits = logits[:, -1, :] / temp
            if step_n == 0:
                logits[:, BOS] = float("-inf")  # force at least one char
            v, _ = torch.topk(logits, min(top_k, VOCAB))
            logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            if step_n > 0 and nxt.item() == BOS:
                break
            if nxt.item() != BOS:
                idx = torch.cat([idx, nxt], dim=1)
        return decode(idx[0].tolist())

# ── Training ──────────────────────────────────────────────────────────────────
# Tiny vocab → n_embd=64, 4 layers fits in memory, trains fast
N_EMBD, N_HEAD, N_LAYER = 64, 4, 4
model = GPT(VOCAB, BLOCK, N_EMBD, N_HEAD, N_LAYER).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"model: {n_params:,} params  n_layer={N_LAYER} n_embd={N_EMBD}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=1e-4)

EVAL_EVERY = max(50, args.iters // 10)

@torch.no_grad()
def eval_loss(n=30):
    model.eval()
    losses = {s: sum(model(xb,yb)[1].item() for xb,yb in (get_batch(s) for _ in range(n)))/n
              for s in ("train","val")}
    model.train()
    return losses

# Snapshot generators at different training stages
snapshots: dict[int, list[str]] = {}

print(f"\nTraining {args.iters} iters...")
t0 = time.time()
for step in range(args.iters):
    # capture snapshot at 10%, 30%, 60%, 100% of training
    if step in {args.iters//10, args.iters*3//10, args.iters*6//10, args.iters-1}:
        gen_model = model.cpu()
        snap = [gen_model.generate_name(temp=0.8, top_k=10) for _ in range(20)]
        snapshots[step] = snap
        model.to(DEVICE)

    if step % EVAL_EVERY == 0:
        L = eval_loss()
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (args.iters - step)
        print(f"  step {step:5d} | train={L['train']:.4f} val={L['val']:.4f} | {elapsed:.0f}s ETA={eta:.0f}s")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()

total = time.time() - t0
print(f"\nDone. {total:.1f}s")

# ── Move model to CPU for all generation/analysis ─────────────────────────────
model = model.cpu()
model.eval()

# ── A. Temperature sweep ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("A. TEMPERATURE SWEEP (10 names each)")
print("="*60)
for temp in (0.2, 0.5, 0.8, 1.2):
    gen = [model.generate_name(temp=temp, top_k=10) for _ in range(10)]
    novel = sum(1 for n in gen if n not in train_set)
    print(f"  T={temp:.1f}: {' | '.join(gen)}")
    print(f"         novelty={novel}/10  unique={len(set(gen))}/10")

# ── B. Novelty + quality analysis ────────────────────────────────────────────
print("\n" + "="*60)
print("B. NOVELTY ANALYSIS (100 names, T=0.8)")
print("="*60)
gen100 = [model.generate_name(temp=0.8, top_k=10) for _ in range(100)]
novel100 = [n for n in gen100 if n not in train_set]
in_train = [n for n in gen100 if n in train_set]
empty    = [n for n in gen100 if len(n) == 0]
print(f"  total=100  novel={len(novel100)}  memorized={len(in_train)}  empty={len(empty)}")
print(f"  novel examples:     {', '.join(novel100[:10])}")
print(f"  memorized examples: {', '.join(in_train[:5])}")

# ── C. Length distribution ────────────────────────────────────────────────────
print("\n" + "="*60)
print("C. LENGTH DISTRIBUTION")
print("="*60)
train_lens = [len(n) for n in train_names]
gen_lens   = [len(n) for n in gen100 if n]
def hist(lens, label):
    c = collections.Counter(lens)
    total = sum(c.values())
    print(f"  {label}: mean={sum(lens)/len(lens):.1f}  "
          f"dist={dict(sorted({k: round(v/total,2) for k,v in c.items() if k<=12}.items()))}")
hist(train_lens, "train")
hist(gen_lens,   "gen  ")

# ── D. Bigram Markov baseline ─────────────────────────────────────────────────
print("\n" + "="*60)
print("D. BIGRAM MARKOV BASELINE")
print("="*60)

# Build bigram transition counts from training names
bigram: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
for name in train_names:
    tokens = encode(name)
    for a, b in zip(tokens, tokens[1:]):
        bigram[a][b] += 1

def bigram_sample(temp=1.0) -> str:
    token = BOS
    out = []
    for _ in range(20):
        counts = bigram.get(token, {BOS: 1})
        tokens_list = list(counts.keys())
        weights = [counts[t] ** (1/temp) for t in tokens_list]
        token = random.choices(tokens_list, weights=weights)[0]
        if token == BOS:
            break
        out.append(i2c[token])
    return "".join(out)

def bigram_log_prob(name: str) -> float:
    tokens = encode(name)
    lp = 0.0
    total_chars = len(tokens)
    for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
        counts = bigram.get(a, {})
        total = sum(counts.values()) or 1
        prob = (counts.get(b, 0) + 1) / (total + VOCAB)  # Laplace smoothing
        lp += math.log(prob)
    return lp / max(total_chars - 1, 1)

def gpt_log_prob(name: str) -> float:
    tokens = torch.tensor([encode(name)], dtype=torch.long)
    if tokens.size(1) < 2:
        return 0.0
    logits, _ = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    lp = -F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1)).item()
    return lp

# Perplexity on val set
bigram_lps = [bigram_log_prob(n) for n in val_names[:200]]
gpt_lps    = [gpt_log_prob(n)    for n in val_names[:200]]
bigram_ppl = math.exp(-sum(bigram_lps)/len(bigram_lps))
gpt_ppl    = math.exp(-sum(gpt_lps)/len(gpt_lps))
print(f"  Val perplexity → Bigram: {bigram_ppl:.2f}  GPT: {gpt_ppl:.2f}  (lower=better)")
print(f"  GPT/Bigram perplexity ratio: {gpt_ppl/bigram_ppl:.3f}x")

# Next-char top-1 accuracy on val names
def top1_acc(model_fn, names_sample):
    correct = total = 0
    for name in names_sample:
        tokens = encode(name)
        for pos in range(len(tokens) - 1):
            pred = model_fn(tokens[:pos+1])
            if pred == tokens[pos+1]:
                correct += 1
            total += 1
    return correct / max(total, 1)

def bigram_top1(context):
    a = context[-1]
    counts = bigram.get(a, {BOS: 1})
    return max(counts, key=counts.get)

def gpt_top1(context):
    idx = torch.tensor([context[-BLOCK:]], dtype=torch.long)
    logits, _ = model(idx)
    return logits[0, -1].argmax().item()

val_sample = val_names[:100]
bg_acc = top1_acc(bigram_top1, val_sample)
gpt_acc = top1_acc(gpt_top1, val_sample)
print(f"  Next-char top-1 accuracy → Bigram: {bg_acc:.3f}  GPT: {gpt_acc:.3f}")
print(f"  GPT improvement over bigram: +{(gpt_acc-bg_acc)*100:.1f}pp")

bg_names = [bigram_sample(temp=0.8) for _ in range(10)]
print(f"  Bigram samples: {' | '.join(bg_names)}")

# ── E. Training snapshots (quality over time) ─────────────────────────────────
print("\n" + "="*60)
print("E. QUALITY OVER TRAINING")
print("="*60)
for step_n, names_snap in sorted(snapshots.items()):
    pct = int((step_n+1) / args.iters * 100)
    novel_n = sum(1 for n in names_snap if n and n not in train_set)
    examples = " | ".join(n for n in names_snap[:6] if n)
    print(f"  step {step_n+1:5d} ({pct:3d}%): {examples}  [novelty {novel_n}/20]")

# ── Final: 5 train + 5 generated ─────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY: 5 TRAIN vs 5 GENERATED")
print("="*60)
print("  Train:    ", " | ".join(train_names[:5]))
gen5 = [model.generate_name(temp=0.7, top_k=10) for _ in range(5)]
print("  Generated:", " | ".join(gen5))
print("="*60)
print(f"\nTotal wall time: {total:.1f}s  ({total/60:.1f} min)")
