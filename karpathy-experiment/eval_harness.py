#!/usr/bin/env python3
"""
eval_harness.py — Experiment A: nanoGPT (nano) vs Bigram on Infineon fab sequences.

Generates synthetic variants from canonical IC/IGBT sequences,
trains both models on 600 sequences, evaluates 200 held-out on 3 tasks,
prints comparison table, saves results_nano.json.

Usage:
  python karpathy-experiment/eval_harness.py [--checkpoint PATH] [--device mps|cpu|cuda]
                                              [--iters N] [--batch_size B]
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Seeds ────────────────────────────────────────────────────────────────────
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
    default="/tmp/zero_one_hack_01/tracks/industrial-infineon/training_data")
parser.add_argument("--checkpoint", default=None,
    help="Skip GPT training — load this .pt checkpoint instead")
parser.add_argument("--device", default=None)
parser.add_argument("--iters", default=1000, type=int)
parser.add_argument("--batch_size", default=64, type=int)
args = parser.parse_args()

DATA_DIR = Path(args.data_dir)

if args.device is None:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
else:
    DEVICE = args.device

print(f"device={DEVICE}  iters={args.iters}  batch={args.batch_size}")

# ════════════════════════════════════════════════════════════════════════════
# § 1  DATA LOADING & VARIANT GENERATION
# ════════════════════════════════════════════════════════════════════════════

# Steps containing these keywords are "optional" (metrology / inspection).
# They are randomly dropped to create sequence variants.
OPTIONAL_KW = (
    "MEASURE", "INSPECT", "INSPECTION", "CHECK", "ANALYSIS",
    "YIELD", "PARAMETRIC", "IDENTIFICATION", "PARTICLE", "ROUGHNESS",
    "PLANARITY", "GEOMETRY", "THICKNESS", "RESISTANCE", "CD ",
    "LINE WIDTH", "VIA CD", "PAD OPENING", "THINNING CHECK",
    "OXIDE CHECK", "FINAL CD", "FINAL PARTICLE", "FINAL ELECTRICAL",
    "SURFACE DEFECT", "SURFACE PARTICLES", "INITIAL THICKNESS",
)

def is_optional(step: str) -> bool:
    su = step.upper()
    return any(kw in su for kw in OPTIONAL_KW)

def load_canonical(path: Path) -> list[str]:
    steps: list[str] = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["STEP"].strip().strip('"')
            if s:
                steps.append(s)
    return steps

def generate_variants(
    canonical: list[str],
    n: int,
    seed: int,
    p_keep_optional: float = 0.70,
) -> list[list[str]]:
    """Create n variants by randomly dropping optional steps."""
    rng = random.Random(seed)
    variants: list[list[str]] = []
    for _ in range(n):
        variant = [s for s in canonical if not is_optional(s) or rng.random() < p_keep_optional]
        if len(variant) >= 10:
            variants.append(variant)
    return variants

print("\n[1] Loading canonical sequences …")
ic_canon   = load_canonical(DATA_DIR / "syntheticIC.csv")
igbt_canon = load_canonical(DATA_DIR / "syntheticIGBT.csv")
print(f"  IC:   {len(ic_canon)} steps  "
      f"({sum(1 for s in ic_canon if is_optional(s))} optional)")
print(f"  IGBT: {len(igbt_canon)} steps  "
      f"({sum(1 for s in igbt_canon if is_optional(s))} optional)")

print("\n[2] Generating synthetic variants …")
ic_vars   = generate_variants(ic_canon,   500, seed=SEED)
igbt_vars = generate_variants(igbt_canon, 500, seed=SEED+1)
print(f"  IC   variants: {len(ic_vars)}  avg len {sum(len(s) for s in ic_vars)/len(ic_vars):.1f}")
print(f"  IGBT variants: {len(igbt_vars)}  avg len {sum(len(s) for s in igbt_vars)/len(igbt_vars):.1f}")

# Stratified train / holdout split (100 per family = 200 held-out)
ic_train,   ic_hold   = ic_vars[:400],   ic_vars[400:]
igbt_train, igbt_hold = igbt_vars[:400], igbt_vars[400:]
train_seqs   = ic_train + igbt_train        # 800
holdout_seqs = ic_hold + igbt_hold          # 200
random.shuffle(train_seqs)
random.shuffle(holdout_seqs)
print(f"\n[3] Split: {len(train_seqs)} train / {len(holdout_seqs)} holdout")

# ── Vocabulary ────────────────────────────────────────────────────────────
all_steps  = sorted(set(s for seq in train_seqs + holdout_seqs for s in seq))
step_to_id = {s: i for i, s in enumerate(all_steps)}
id_to_step = all_steps
BOS_ID     = len(all_steps)
EOS_ID     = len(all_steps) + 1
VOCAB_SIZE = len(all_steps) + 2
BLOCK_SIZE = 256
print(f"[4] Vocab: {len(all_steps)} steps + BOS/EOS = {VOCAB_SIZE} tokens")

def encode(seq: list[str]) -> list[int]:
    return [BOS_ID] + [step_to_id[s] for s in seq] + [EOS_ID]

def decode(ids: list[int]) -> list[str]:
    out: list[str] = []
    for i in ids:
        if i == BOS_ID:
            continue
        if i == EOS_ID:
            break
        if 0 <= i < len(id_to_step):
            out.append(id_to_step[i])
    return out

# ════════════════════════════════════════════════════════════════════════════
# § 2  ANOMALY GENERATION  (50 clean + 50 violated for Task 3)
# ════════════════════════════════════════════════════════════════════════════

def _inject_clean_skip(seq: list[str]) -> list[str] | None:
    """Remove a CLEAN step immediately before a DEPOSIT step (violates pre-clean rule)."""
    s = list(seq)
    for i in range(1, len(s)):
        if "DEPOSIT" in s[i].upper():
            for j in range(max(0, i-5), i):
                if "CLEAN" in s[j].upper() or "RINSE" in s[j].upper():
                    s.pop(j)
                    return s
    return None

def _inject_ship_early(seq: list[str]) -> list[str] | None:
    """Move SHIP LOT to before WAFER SORT TEST (ship before test)."""
    s = list(seq)
    ship = next((i for i, x in enumerate(s) if "SHIP" in x.upper()), None)
    test = next((i for i, x in enumerate(s) if "SORT TEST" in x.upper() or
                 "WAFER SORT" in x.upper()), None)
    if ship is not None and test is not None and ship > test + 1:
        item = s.pop(ship)
        s.insert(test, item)
        return s
    return None

def _inject_anneal_skip(seq: list[str]) -> list[str] | None:
    """Remove ANNEAL step after IMPLANT (violates implant-anneal ordering)."""
    s = list(seq)
    for i in range(len(s)):
        if "IMPLANT" in s[i].upper():
            for j in range(i+1, min(i+6, len(s))):
                if "ANNEAL" in s[j].upper():
                    s.pop(j)
                    return s
    return None

def _inject_swap(seq: list[str], rng: random.Random) -> list[str]:
    """Swap two distant mandatory steps (structural disorder)."""
    s = list(seq)
    mandatory = [i for i, x in enumerate(s) if not is_optional(x)]
    if len(mandatory) >= 6:
        idx = rng.sample(mandatory, 2)
        i, j = sorted(idx)
        if j - i >= 5:  # must be non-adjacent
            s[i], s[j] = s[j], s[i]
    return s

def inject_violation(seq: list[str], rng: random.Random) -> list[str]:
    """Apply a random rule violation. Tries rules in random order, falls back to swap."""
    rules = [_inject_clean_skip, _inject_ship_early, _inject_anneal_skip]
    rng.shuffle(rules)
    for fn in rules:
        r = fn(seq)
        if r is not None:
            return r
    return _inject_swap(seq, rng)

print("\n[5] Building anomaly test set …")
_vrng = random.Random(SEED + 99)
task3_clean    = holdout_seqs[150:200]          # 50 clean
task3_violated = [inject_violation(s, _vrng) for s in task3_clean]  # 50 violated
task12_holdout = holdout_seqs[:150]             # 150 for Task 1 & 2
print(f"  Task1/2: {len(task12_holdout)} seqs  "
      f"| Task3 clean: {len(task3_clean)}  violated: {len(task3_violated)}")

# ════════════════════════════════════════════════════════════════════════════
# § 3  BIGRAM BASELINE
# ════════════════════════════════════════════════════════════════════════════

class BigramModel:
    def __init__(self) -> None:
        self.counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def fit(self, seqs: list[list[str]]) -> None:
        for seq in seqs:
            enc = encode(seq)
            for a, b in zip(enc, enc[1:]):
                self.counts[a][b] += 1

    def top_k(self, ctx: list[int], k: int) -> list[int]:
        """Return top-k token IDs for next step given context (uses last token only)."""
        last = ctx[-1] if ctx else BOS_ID
        raw  = self.counts.get(last, {})
        # +1 Laplace over all non-BOS tokens
        candidates = {tok: raw.get(tok, 0) + 1 for tok in range(VOCAB_SIZE)}
        return sorted(candidates, key=lambda t: -candidates[t])[:k]

    def seq_logprob(self, seq: list[str]) -> float:
        enc = encode(seq)
        lp: list[float] = []
        for a, b in zip(enc, enc[1:]):
            raw   = self.counts.get(a, {})
            total = sum(raw.values()) + VOCAB_SIZE  # Laplace
            prob  = (raw.get(b, 0) + 1) / total
            lp.append(math.log(prob + 1e-12))
        return float(np.mean(lp)) if lp else -999.0

    def greedy_complete(self, prefix: list[int], max_steps: int = 200) -> list[int]:
        ctx = list(prefix)
        for _ in range(max_steps):
            nxt = self.top_k(ctx, 1)[0]
            if nxt == EOS_ID:
                break
            ctx.append(nxt)
        return ctx

print("\n[6] Training Bigram …")
t0_bg = time.time()
bigram = BigramModel()
bigram.fit(train_seqs)
bg_train_time = time.time() - t0_bg
n_transitions = sum(len(v) for v in bigram.counts.values())
print(f"  {n_transitions} transitions  trained in {bg_train_time:.3f}s")

# ════════════════════════════════════════════════════════════════════════════
# § 4  NANOGPT MODEL
# ════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.n_head   = n_head
        self.n_embd   = n_embd
        self.head_dim = n_embd // n_head
        self.qkv  = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd,   bias=False)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4*n_embd, bias=False)
        self.fc2 = nn.Linear(4*n_embd, n_embd, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

class GPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1  = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2  = RMSNorm(n_embd)
        self.mlp  = MLP(n_embd)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer:    int = 6
    n_embd:     int = 128
    n_head:     int = 8

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.wte     = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe     = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks  = nn.ModuleList(
            [GPTBlock(cfg.n_embd, cfg.n_head, cfg.block_size) for _ in range(cfg.n_layer)]
        )
        self.ln_f    = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        pos    = torch.arange(T, device=idx.device)
        x      = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        x      = self.ln_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ── Training data packing ─────────────────────────────────────────────────

def pack_to_tensor(seqs: list[list[str]]) -> torch.Tensor:
    flat: list[int] = []
    for seq in seqs:
        flat.extend(encode(seq))
    return torch.tensor(flat, dtype=torch.long)

BATCH_SIZE  = args.batch_size
MAX_ITERS   = args.iters
LR          = 3e-4
EVAL_EVERY  = max(100, MAX_ITERS // 10)

split_n     = int(0.9 * len(train_seqs))
gpt_train   = pack_to_tensor(train_seqs[:split_n])
gpt_val     = pack_to_tensor(train_seqs[split_n:])
print(f"\n[7] Token buffer: train={len(gpt_train):,}  val={len(gpt_val):,}")

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = gpt_train if split == "train" else gpt_val
    ix   = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x    = torch.stack([data[i:i+BLOCK_SIZE]     for i in ix])
    y    = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(n: int = 40) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        ls = [model(*get_batch(split))[1].item() for _ in range(n)]
        out[split] = float(np.mean(ls))
    model.train()
    return out

# ── Train or load ─────────────────────────────────────────────────────────
gpt_cfg = GPTConfig(vocab_size=VOCAB_SIZE)
model   = GPT(gpt_cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"[8] Model: {n_params/1e6:.2f}M params on {DEVICE}")

gpt_train_time = 0.0
best_val_loss  = float("inf")

if args.checkpoint:
    print(f"  Loading checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    best_val_loss = ck.get("val_loss", float("inf"))
    print(f"  val_loss={best_val_loss:.4f}")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   betas=(0.9, 0.99), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=MAX_ITERS
    )
    print(f"\n[8b] Training GPT for {MAX_ITERS} iters …")
    t0_gpt = time.time()
    _iter_times: list[float] = []
    _t = time.time()

    for step in range(MAX_ITERS):
        if step % EVAL_EVERY == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0_gpt
            ms_it   = (sum(_iter_times[-EVAL_EVERY:]) /
                       max(len(_iter_times[-EVAL_EVERY:]), 1) * 1000)
            eta     = ms_it / 1000 * (MAX_ITERS - step)
            print(f"  step {step:5d} | train={losses['train']:.4f}  val={losses['val']:.4f} "
                  f"| {ms_it:.0f}ms/it | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        _iter_times.append(time.time() - _t)
        _t = time.time()

    gpt_train_time = time.time() - t0_gpt
    print(f"  GPT training: {gpt_train_time:.1f}s  best_val_loss={best_val_loss:.4f}")

# Save checkpoint
ck_dir  = Path(__file__).parent / "checkpoints"
ck_dir.mkdir(exist_ok=True)
ck_path = ck_dir / "gpt_nano_eval.pt"
torch.save({
    "model":       model.state_dict(),
    "val_loss":    best_val_loss,
    "step_to_id":  step_to_id,
    "id_to_step":  id_to_step,
    "config":      gpt_cfg.__dict__,
}, ck_path)
print(f"  Checkpoint saved: {ck_path}")

# ── GPT inference helpers ─────────────────────────────────────────────────

# Keep model on DEVICE for all inference; generation is also run on DEVICE.
# (MPS per-token generation is ~600ms/step but manageable for N_TASK2=30 seqs.)
GEN_DEVICE = DEVICE
gen_model  = model

@torch.no_grad()
def gpt_top_k_ids(ctx_ids: list[int], k: int) -> list[int]:
    """Return top-k next-token IDs for a context."""
    model.eval()
    x = torch.tensor([ctx_ids[-BLOCK_SIZE:]], dtype=torch.long, device=model.wte.weight.device)
    logits, _ = model(x)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return probs.topk(k).indices.tolist()

@torch.no_grad()
def gpt_seq_logprob(seq: list[str]) -> float:
    """Mean log-prob of a sequence under GPT (higher = more normal)."""
    model.eval()
    enc = encode(seq)
    if len(enc) < 2:
        return -999.0
    dev   = model.wte.weight.device
    enc_t = torch.tensor([enc[:BLOCK_SIZE]], dtype=torch.long, device=dev)
    logits, _ = model(enc_t)
    log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V)
    T       = min(len(enc)-1, BLOCK_SIZE)
    targets = torch.tensor(enc[1:T+1], dtype=torch.long, device=dev)
    per_tok = log_probs[:T].gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(per_tok.mean().item())

@torch.no_grad()
def gpt_greedy_complete(prefix_ids: list[int], max_steps: int = 200) -> list[int]:
    """Greedy autoregressive completion."""
    dev = model.wte.weight.device
    model.eval()
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=dev)
    for _ in range(max_steps):
        cond  = idx[:, -BLOCK_SIZE:]
        logits, _ = model(cond)
        nxt   = logits[0, -1, :].argmax().item()
        idx   = torch.cat([idx, torch.tensor([[nxt]], device=dev)], dim=1)
        if nxt == EOS_ID:
            break
    return idx[0].tolist()

# ════════════════════════════════════════════════════════════════════════════
# § 5  EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def levenshtein(a: list[int], b: list[int]) -> int:
    """Token-level Levenshtein edit distance (DP)."""
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev, dp[0] = dp[0], i
        for j in range(1, n+1):
            old   = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev  = old
    return dp[n]

# ── Task 1: Next-step prediction ─────────────────────────────────────────

print("\n" + "="*70)
print("TASK 1  Next-step prediction (top-1 / top-3 / top-5)")
print("="*70)

POSITIONS = [10, 30, 50, 80]
task1_results: dict[str, dict] = {"gpt": {}, "bigram": {}}

for pos in POSITIONS:
    gpt_hits   = {1: 0, 3: 0, 5: 0}
    bg_hits    = {1: 0, 3: 0, 5: 0}
    total      = 0

    for seq in task12_holdout:
        enc = encode(seq)
        # enc[0] = BOS, enc[1..] = steps; position p means predicting enc[p+1]
        # We want the p-th step in the sequence (0-indexed)
        # context = enc[0..p]  (length p+1)
        # target  = enc[p+1]
        if pos + 1 >= len(enc):
            continue  # sequence too short

        ctx    = enc[:pos+1]
        target = enc[pos+1]
        total += 1

        # GPT
        gpt_k5 = gpt_top_k_ids(ctx, 5)
        for k in (1, 3, 5):
            if target in gpt_k5[:k]:
                gpt_hits[k] += 1

        # Bigram
        bg_k5 = bigram.top_k(ctx, 5)
        for k in (1, 3, 5):
            if target in bg_k5[:k]:
                bg_hits[k] += 1

    gpt_row = {k: gpt_hits[k]/total*100 if total else 0 for k in (1,3,5)}
    bg_row  = {k: bg_hits[k]/total*100  if total else 0 for k in (1,3,5)}
    task1_results["gpt"][pos]    = gpt_row
    task1_results["bigram"][pos] = bg_row
    print(f"  pos={pos:3d}  n={total:4d} | "
          f"GPT top1={gpt_row[1]:.1f}% top3={gpt_row[3]:.1f}% top5={gpt_row[5]:.1f}% | "
          f"Bigram top1={bg_row[1]:.1f}% top3={bg_row[3]:.1f}% top5={bg_row[5]:.1f}%")

# Aggregate across positions
def _agg(model_key: str, k: int) -> float:
    vals = [task1_results[model_key][p][k] for p in POSITIONS
            if p in task1_results[model_key]]
    return float(np.mean(vals)) if vals else 0.0

gpt_top1_avg = _agg("gpt", 1)
bg_top1_avg  = _agg("bigram", 1)
print(f"\n  Overall top-1: GPT={gpt_top1_avg:.1f}%  Bigram={bg_top1_avg:.1f}%")

# ── Task 2: Sequence completion ───────────────────────────────────────────

print("\n" + "="*70)
print("TASK 2  Sequence completion (prefix=50 steps, greedy decoding)")
print("="*70)

CUTPOINT  = 50
N_TASK2   = 30   # limit for wall-clock (generation is slow)
task2_seqs = [s for s in task12_holdout if len(s) > CUTPOINT + 5][:N_TASK2]

gpt_edit_dists, bg_edit_dists   = [], []
gpt_tok_accs,   bg_tok_accs     = [], []
gpt_overlaps,   bg_overlaps     = [], []

t2_start = time.time()
for i, seq in enumerate(task2_seqs):
    enc        = encode(seq)
    prefix_enc = enc[:CUTPOINT+1]           # BOS + first 50 steps
    gold_enc   = enc[CUTPOINT+1:]           # remaining steps + EOS
    gold_ids   = [t for t in gold_enc if t != EOS_ID]

    # GPT greedy completion
    gpt_full = gpt_greedy_complete(prefix_enc, max_steps=len(gold_ids)+30)
    gpt_ids  = [t for t in gpt_full[CUTPOINT+1:] if t not in (EOS_ID, BOS_ID)]

    # Bigram greedy completion
    bg_full  = bigram.greedy_complete(prefix_enc, max_steps=len(gold_ids)+30)
    bg_ids   = [t for t in bg_full[CUTPOINT+1:] if t not in (EOS_ID, BOS_ID)]

    max_len  = max(len(gold_ids), 1)
    gpt_ed   = levenshtein(gpt_ids, gold_ids)
    bg_ed    = levenshtein(bg_ids,  gold_ids)
    gpt_edit_dists.append(gpt_ed / max_len)
    bg_edit_dists.append(bg_ed   / max_len)

    # Token accuracy (fraction of positions that match, up to min length)
    def tok_acc(pred: list[int], gold: list[int]) -> float:
        n = min(len(pred), len(gold))
        if n == 0:
            return 0.0
        return sum(p == g for p, g in zip(pred, gold)) / n

    gpt_tok_accs.append(tok_acc(gpt_ids, gold_ids))
    bg_tok_accs.append(tok_acc(bg_ids,   gold_ids))

    # Step overlap (F1-style: % of gold steps that appear in pred at all)
    gpt_overlap = len(set(gpt_ids) & set(gold_ids)) / max(len(set(gold_ids)), 1)
    bg_overlap  = len(set(bg_ids)  & set(gold_ids)) / max(len(set(gold_ids)), 1)
    gpt_overlaps.append(gpt_overlap)
    bg_overlaps.append(bg_overlap)

t2_elapsed = time.time() - t2_start

task2_results = {
    "gpt":    {"norm_edit":  float(np.mean(gpt_edit_dists)),
               "tok_acc":    float(np.mean(gpt_tok_accs)),
               "step_overlap": float(np.mean(gpt_overlaps))},
    "bigram": {"norm_edit":  float(np.mean(bg_edit_dists)),
               "tok_acc":    float(np.mean(bg_tok_accs)),
               "step_overlap": float(np.mean(bg_overlaps))},
}
print(f"  n={len(task2_seqs)} sequences  ({t2_elapsed:.1f}s on {GEN_DEVICE if DEVICE=='mps' else DEVICE})")
print(f"  GPT:    norm_edit={task2_results['gpt']['norm_edit']:.3f}  "
      f"tok_acc={task2_results['gpt']['tok_acc']:.3f}  "
      f"step_overlap={task2_results['gpt']['step_overlap']:.3f}")
print(f"  Bigram: norm_edit={task2_results['bigram']['norm_edit']:.3f}  "
      f"tok_acc={task2_results['bigram']['tok_acc']:.3f}  "
      f"step_overlap={task2_results['bigram']['step_overlap']:.3f}")
print("  (lower norm_edit = better;  higher tok_acc / step_overlap = better)")

# ── Task 3: Anomaly detection ─────────────────────────────────────────────

print("\n" + "="*70)
print("TASK 3  Anomaly detection (50 clean + 50 violated sequences)")
print("="*70)

def compute_f1(scores: list[float], labels: list[int],
               threshold: float) -> dict[str, float]:
    """Binary F1: label=1 means anomalous. Predict 1 if score < threshold."""
    preds = [1 if s < threshold else 0 for s in scores]
    tp = sum(p==1 and g==1 for p,g in zip(preds, labels))
    fp = sum(p==1 and g==0 for p,g in zip(preds, labels))
    fn = sum(p==0 and g==1 for p,g in zip(preds, labels))
    tn = sum(p==0 and g==0 for p,g in zip(preds, labels))
    prec = tp / (tp+fp+1e-9)
    rec  = tp / (tp+fn+1e-9)
    f1   = 2*prec*rec / (prec+rec+1e-9)
    acc  = (tp+tn) / len(labels)
    return {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def best_f1_threshold(scores: list[float], labels: list[int]) -> tuple[float, dict]:
    """Find threshold that maximises F1 on this set."""
    best_f1, best_t, best_metrics = 0.0, 0.0, {}
    thresholds = sorted(set(scores))
    for t in thresholds:
        m = compute_f1(scores, labels, t)
        if m["f1"] > best_f1:
            best_f1, best_t, best_metrics = m["f1"], t, m
    return best_t, best_metrics

# GPT scores
gpt_clean_scores    = [gpt_seq_logprob(s) for s in task3_clean]
gpt_violated_scores = [gpt_seq_logprob(s) for s in task3_violated]
gpt_all_scores      = gpt_clean_scores + gpt_violated_scores
labels_all          = [0]*50 + [1]*50   # 0=normal, 1=anomaly

# Bigram scores
bg_clean_scores    = [bigram.seq_logprob(s) for s in task3_clean]
bg_violated_scores = [bigram.seq_logprob(s) for s in task3_violated]
bg_all_scores      = bg_clean_scores + bg_violated_scores

gpt_thresh, gpt_t3  = best_f1_threshold(gpt_all_scores, labels_all)
bg_thresh,  bg_t3   = best_f1_threshold(bg_all_scores,  labels_all)

task3_results = {
    "gpt":    {**gpt_t3, "threshold": gpt_thresh,
               "clean_mean_logp":    float(np.mean(gpt_clean_scores)),
               "violated_mean_logp": float(np.mean(gpt_violated_scores))},
    "bigram": {**bg_t3,  "threshold": bg_thresh,
               "clean_mean_logp":    float(np.mean(bg_clean_scores)),
               "violated_mean_logp": float(np.mean(bg_violated_scores))},
}

print(f"  GPT    clean logP={np.mean(gpt_clean_scores):.3f}  "
      f"violated logP={np.mean(gpt_violated_scores):.3f}  "
      f"→ F1={gpt_t3['f1']:.3f}  acc={gpt_t3['accuracy']:.3f}")
print(f"  Bigram clean logP={np.mean(bg_clean_scores):.3f}  "
      f"violated logP={np.mean(bg_violated_scores):.3f}  "
      f"→ F1={bg_t3['f1']:.3f}  acc={bg_t3['accuracy']:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# § 6  COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print(f"{'Model':<10} | {'Train(s)':>8} | "
      f"{'T1 top1':>7} {'T1 top3':>7} {'T1 top5':>7} | "
      f"{'T2 edit':>7} {'T2 tacc':>7} | "
      f"{'T3 F1':>6} {'T3 acc':>6}")
print("-"*85)

for model_key, train_t in [("gpt", gpt_train_time), ("bigram", bg_train_time)]:
    t1 = task1_results[model_key]
    t2 = task2_results[model_key]
    t3 = task3_results[model_key]
    top1 = _agg(model_key, 1)
    top3 = _agg(model_key, 3)
    top5 = _agg(model_key, 5)
    print(f"{model_key:<10} | {train_t:>8.1f} | "
          f"{top1:>6.1f}% {top3:>6.1f}% {top5:>6.1f}% | "
          f"{t2['norm_edit']:>7.3f} {t2['tok_acc']:>7.3f} | "
          f"{t3['f1']:>6.3f} {t3['accuracy']:>6.3f}")

print("="*70)
print(f"GPT val_loss={best_val_loss:.4f}  (ppl={math.exp(best_val_loss):.2f})  "
      f"params={n_params/1e6:.2f}M")

# ════════════════════════════════════════════════════════════════════════════
# § 7  SAVE JSON
# ════════════════════════════════════════════════════════════════════════════

results = {
    "meta": {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device":      DEVICE,
        "gpt_iters":   args.iters,
        "batch_size":  args.batch_size,
        "n_params":    n_params,
        "vocab_size":  VOCAB_SIZE,
        "train_seqs":  len(train_seqs),
        "holdout_seqs": len(holdout_seqs),
    },
    "models": {
        "gpt": {
            "train_time_s": gpt_train_time,
            "best_val_loss": best_val_loss,
            "val_ppl":       math.exp(best_val_loss),
            "task1":         {str(k): v for k, v in task1_results["gpt"].items()},
            "task1_top1_avg": gpt_top1_avg,
            "task2":         task2_results["gpt"],
            "task3":         task3_results["gpt"],
        },
        "bigram": {
            "train_time_s": bg_train_time,
            "task1":         {str(k): v for k, v in task1_results["bigram"].items()},
            "task1_top1_avg": bg_top1_avg,
            "task2":         task2_results["bigram"],
            "task3":         task3_results["bigram"],
        },
    },
}

out_path = Path(__file__).parent / "results_nano.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved → {out_path}")
print("DONE.")
