#!/usr/bin/env python3
"""
anomaly_detection.py — Task 3 anomaly detection for Infineon semiconductor fab sequences.

Trains nanoGPT-nano (6L/128d, 500 iters) on 2700 synthetic sequence variants,
then compares GPT log-prob vs bigram log-prob for detecting violated sequences.

Mutations applied to 300 held-out sequences:
  (a) swap two adjacent steps at a random position
  (b) remove a CLEAN step immediately before a DEPOSIT or ETCH step
  (c) insert a wrong-fab-phase step (e.g. SHIP LOT at position 10)

Output: karpathy-experiment/results_anomaly.json

Usage:
  python karpathy-experiment/anomaly_detection.py [--device mps|cpu|cuda] [--iters N]
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
parser.add_argument("--device", default=None)
parser.add_argument("--iters", default=500, type=int)
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

print("\n[2] Generating 3000 synthetic variants (1500 IC + 1500 IGBT) …")
ic_vars   = generate_variants(ic_canon,   1500, seed=SEED)
igbt_vars = generate_variants(igbt_canon, 1500, seed=SEED+1)
print(f"  IC   variants: {len(ic_vars)}  avg len {sum(len(s) for s in ic_vars)/len(ic_vars):.1f}")
print(f"  IGBT variants: {len(igbt_vars)}  avg len {sum(len(s) for s in igbt_vars)/len(igbt_vars):.1f}")

# Stratified split: 150 holdout per family = 300 held-out, rest = ~2700 train
ic_train,   ic_hold   = ic_vars[:1350],   ic_vars[1350:]
igbt_train, igbt_hold = igbt_vars[:1350], igbt_vars[1350:]
train_seqs   = ic_train + igbt_train        # ~2700
holdout_seqs = ic_hold + igbt_hold          # 300
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
# § 2  MUTATION FUNCTIONS  (Task 3 violation injection)
# ════════════════════════════════════════════════════════════════════════════
#
# Three mutation types as per task specification:
#   (a) swap_adjacent  — swap two adjacent steps at a random position
#   (b) remove_clean   — remove a CLEAN step before DEPOSIT or ETCH
#   (c) insert_wrong   — insert a wrong-fab-phase step (SHIP LOT) at position 10

# Steps that belong to "late logistics" phase — inserting early = violation.
# Restricted to known vocab tokens (populated after vocab build).
_WRONG_PHASE_STEPS_CANDIDATES = [
    "SHIP LOT",
    "LOT RELEASE",
    "FINAL LOT RELEASE",
    "WAFER SORT TEST",
    "YIELD ANALYSIS",
    "PARAMETRIC TEST",
    "PACKAGE PREPARATION",
]
_WRONG_PHASE_STEPS: list[str] = [s for s in _WRONG_PHASE_STEPS_CANDIDATES if s in step_to_id]
print(f"  Wrong-phase steps in vocab: {_WRONG_PHASE_STEPS}")

def _mut_swap_adjacent(seq: list[str], rng: random.Random) -> list[str]:
    """Swap two adjacent steps at a random interior position."""
    s = list(seq)
    if len(s) < 4:
        return s
    # pick a random position (not first or last)
    pos = rng.randint(1, len(s) - 3)
    s[pos], s[pos + 1] = s[pos + 1], s[pos]
    return s

def _mut_remove_clean(seq: list[str]) -> list[str] | None:
    """Remove a CLEAN/RINSE step that appears immediately before DEPOSIT or ETCH.
    Returns None if no such pattern found."""
    s = list(seq)
    for i in range(1, len(s)):
        su = s[i].upper()
        if "DEPOSIT" in su or "ETCH" in su or "DEPOSITION" in su:
            # look for CLEAN or RINSE in window [i-5, i)
            for j in range(max(0, i - 5), i):
                if "CLEAN" in s[j].upper() or "RINSE" in s[j].upper():
                    s.pop(j)
                    return s
    return None

def _mut_insert_wrong_phase(seq: list[str], rng: random.Random) -> list[str]:
    """Insert a wrong-fab-phase step (logistics/end-of-line) at position 10."""
    s = list(seq)
    step = rng.choice(_WRONG_PHASE_STEPS)
    # Make sure step not already present
    if step not in s:
        insert_pos = min(10, len(s) - 1)
        s.insert(insert_pos, step)
    return s

def inject_violation(seq: list[str], rng: random.Random) -> tuple[list[str], str]:
    """Apply a random rule violation. Returns (mutated_seq, mutation_name).
    Tries remove_clean first (most semantically meaningful), falls back to others."""
    # Prioritize semantically rich violations
    order = ["remove_clean", "insert_wrong", "swap_adjacent"]
    rng.shuffle(order)
    for mut in order:
        if mut == "remove_clean":
            r = _mut_remove_clean(seq)
            if r is not None:
                return r, "remove_clean"
        elif mut == "insert_wrong":
            r = _mut_insert_wrong_phase(seq, rng)
            if r != seq:  # only if it changed
                return r, "insert_wrong"
        elif mut == "swap_adjacent":
            return _mut_swap_adjacent(seq, rng), "swap_adjacent"
    # absolute fallback
    return _mut_swap_adjacent(seq, rng), "swap_adjacent"

print("\n[5] Building anomaly test set …")
_vrng = random.Random(SEED + 99)

# All 300 held-out sequences are "clean" (label=0)
# Generate 300 violated sequences (label=1) from copies of the held-out
task3_clean    = holdout_seqs                    # 300 clean
task3_violated = []
task3_mut_types: list[str] = []
for seq in task3_clean:
    mutated, mut_type = inject_violation(seq, _vrng)
    task3_violated.append(mutated)
    task3_mut_types.append(mut_type)

mut_counts: dict[str, int] = defaultdict(int)
for t in task3_mut_types:
    mut_counts[t] += 1
print(f"  Clean: {len(task3_clean)}  Violated: {len(task3_violated)}")
print(f"  Mutation breakdown: " + "  ".join(f"{k}={v}" for k, v in sorted(mut_counts.items())))

# Labels: 0=normal/clean, 1=anomaly/violated
labels_all = [0] * len(task3_clean) + [1] * len(task3_violated)

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
        last = ctx[-1] if ctx else BOS_ID
        raw  = self.counts.get(last, {})
        candidates = {tok: raw.get(tok, 0) + 1 for tok in range(VOCAB_SIZE)}
        return sorted(candidates, key=lambda t: -candidates[t])[:k]

    def seq_logprob(self, seq: list[str]) -> float:
        enc = encode(seq)
        lp: list[float] = []
        for a, b in zip(enc, enc[1:]):
            raw   = self.counts.get(a, {})
            total = sum(raw.values()) + VOCAB_SIZE  # Laplace smoothing
            prob  = (raw.get(b, 0) + 1) / total
            lp.append(math.log(prob + 1e-12))
        return float(np.mean(lp)) if lp else -999.0

print("\n[6] Training Bigram …")
t0_bg = time.time()
bigram = BigramModel()
bigram.fit(train_seqs)
bg_train_time = time.time() - t0_bg
n_transitions = sum(len(v) for v in bigram.counts.values())
print(f"  {n_transitions} transitions  trained in {bg_train_time:.3f}s")

# ════════════════════════════════════════════════════════════════════════════
# § 4  NANOGPT MODEL  (6L / 128d / 8H — ~1.24M params)
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
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
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
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

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


# ── Token buffer packing ──────────────────────────────────────────────────

def pack_to_tensor(seqs: list[list[str]]) -> torch.Tensor:
    flat: list[int] = []
    for seq in seqs:
        flat.extend(encode(seq))
    return torch.tensor(flat, dtype=torch.long)

BATCH_SIZE  = args.batch_size
MAX_ITERS   = args.iters
LR          = 3e-4
EVAL_EVERY  = max(50, MAX_ITERS // 10)

_split_n  = int(0.9 * len(train_seqs))
gpt_train = pack_to_tensor(train_seqs[:_split_n])
gpt_val   = pack_to_tensor(train_seqs[_split_n:])
print(f"\n[7] Token buffer: train={len(gpt_train):,}  val={len(gpt_val):,}")

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = gpt_train if split == "train" else gpt_val
    ix   = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x    = torch.stack([data[i:i + BLOCK_SIZE]     for i in ix])
    y    = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
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

# ── Instantiate model ─────────────────────────────────────────────────────
gpt_cfg = GPTConfig(vocab_size=VOCAB_SIZE)
model   = GPT(gpt_cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"[8] Model: {n_params / 1e6:.2f}M params on {DEVICE}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               betas=(0.9, 0.99), weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=MAX_ITERS
)

# ── Training loop ──────────────────────────────────────────────────────────
print(f"\n[8b] Training GPT for {MAX_ITERS} iters …")
t0_gpt = time.time()
_iter_times: list[float] = []
_t = time.time()
best_val_loss = float("inf")

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
print(f"  GPT training: {gpt_train_time:.1f}s  best_val_loss={best_val_loss:.4f}  "
      f"ppl={math.exp(best_val_loss):.3f}")

# ── Save checkpoint ────────────────────────────────────────────────────────
ck_dir  = Path(__file__).parent / "checkpoints"
ck_dir.mkdir(exist_ok=True)
ck_path = ck_dir / "gpt_anomaly.pt"
torch.save({
    "model":      model.state_dict(),
    "val_loss":   best_val_loss,
    "step_to_id": step_to_id,
    "id_to_step": id_to_step,
    "config":     gpt_cfg.__dict__,
}, ck_path)
print(f"  Checkpoint: {ck_path}")

# ── GPT scoring helper ────────────────────────────────────────────────────
@torch.no_grad()
def gpt_seq_logprob(seq: list[str]) -> float:
    """Mean log-prob per token of a sequence under GPT (higher = more normal)."""
    model.eval()
    enc = encode(seq)
    if len(enc) < 2:
        return -999.0
    dev   = model.wte.weight.device
    enc_t = torch.tensor([enc[:BLOCK_SIZE]], dtype=torch.long, device=dev)
    logits, _ = model(enc_t)
    log_probs = F.log_softmax(logits[0], dim=-1)   # (T, V)
    T       = min(len(enc) - 1, BLOCK_SIZE)
    targets = torch.tensor(enc[1:T + 1], dtype=torch.long, device=dev)
    per_tok = log_probs[:T].gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(per_tok.mean().item())

# ════════════════════════════════════════════════════════════════════════════
# § 5  ANOMALY SCORING
# ════════════════════════════════════════════════════════════════════════════

print("\n[9] Scoring sequences (GPT log-prob) …")
t_score = time.time()
gpt_clean_scores    = [gpt_seq_logprob(s) for s in task3_clean]
gpt_violated_scores = [gpt_seq_logprob(s) for s in task3_violated]
print(f"  GPT scoring: {time.time()-t_score:.1f}s")

print("[10] Scoring sequences (Bigram log-prob) …")
bg_clean_scores    = [bigram.seq_logprob(s) for s in task3_clean]
bg_violated_scores = [bigram.seq_logprob(s) for s in task3_violated]

gpt_all_scores = gpt_clean_scores + gpt_violated_scores
bg_all_scores  = bg_clean_scores  + bg_violated_scores

print(f"\n  GPT   — clean: {np.mean(gpt_clean_scores):.4f} ± {np.std(gpt_clean_scores):.4f}  "
      f"violated: {np.mean(gpt_violated_scores):.4f} ± {np.std(gpt_violated_scores):.4f}  "
      f"gap: {np.mean(gpt_clean_scores) - np.mean(gpt_violated_scores):.4f}")
print(f"  Bigram — clean: {np.mean(bg_clean_scores):.4f} ± {np.std(bg_clean_scores):.4f}  "
      f"violated: {np.mean(bg_violated_scores):.4f} ± {np.std(bg_violated_scores):.4f}  "
      f"gap: {np.mean(bg_clean_scores) - np.mean(bg_violated_scores):.4f}")

# ════════════════════════════════════════════════════════════════════════════
# § 6  ROC CURVE + BEST F1 THRESHOLD
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores: list[float], labels: list[int],
                    threshold: float) -> dict[str, float]:
    """Anomaly if score < threshold (lower log-prob = more anomalous)."""
    preds = [1 if s < threshold else 0 for s in scores]
    tp = sum(p == 1 and g == 1 for p, g in zip(preds, labels))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds, labels))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds, labels))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds, labels))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / len(labels)
    return {
        "f1": float(f1), "precision": float(prec), "recall": float(rec),
        "accuracy": float(acc),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

def roc_sweep(scores: list[float], labels: list[int]) -> dict:
    """Sweep all score thresholds, compute TPR/FPR/F1, find best-F1 point."""
    # Sort unique thresholds (include boundary outside score range)
    thresholds = sorted(set(scores + [min(scores) - 1e-6, max(scores) + 1e-6]))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    roc_points: list[dict] = []
    best_f1, best_thresh, best_metrics = 0.0, thresholds[0], {}

    for t in thresholds:
        m = compute_metrics(scores, labels, t)
        tpr = m["tp"] / (n_pos + 1e-9)
        fpr = m["fp"] / (n_neg + 1e-9)
        roc_points.append({"threshold": float(t), "tpr": float(tpr),
                           "fpr": float(fpr), "f1": m["f1"]})
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = t
            best_metrics = m

    # AUC-ROC (trapezoid rule over sorted FPR)
    roc_sorted = sorted(roc_points, key=lambda p: p["fpr"])
    auc = 0.0
    for i in range(1, len(roc_sorted)):
        dx = roc_sorted[i]["fpr"] - roc_sorted[i-1]["fpr"]
        auc += dx * (roc_sorted[i]["tpr"] + roc_sorted[i-1]["tpr"]) / 2

    return {
        "best_threshold": float(best_thresh),
        "best_f1":        float(best_f1),
        "auc_roc":        float(auc),
        "best_metrics":   best_metrics,
        "roc_curve":      roc_sorted,   # full curve (omit from JSON for brevity if needed)
    }

print("\n[11] ROC sweep …")
gpt_roc = roc_sweep(gpt_all_scores, labels_all)
bg_roc  = roc_sweep(bg_all_scores,  labels_all)

# Per-mutation-type breakdown (GPT)
print("\n[12] Per-mutation-type breakdown (GPT) …")
mutation_breakdown: dict[str, dict] = {}
for mut in sorted(set(task3_mut_types)):
    idxs = [i for i, t in enumerate(task3_mut_types) if t == mut]
    c_scores = [gpt_clean_scores[i] for i in idxs]
    v_scores = [gpt_violated_scores[i] for i in idxs]
    n = len(idxs)
    sub_labels = [0] * n + [1] * n
    sub_scores = c_scores + v_scores
    sub_roc = roc_sweep(sub_scores, sub_labels)
    mutation_breakdown[mut] = {
        "n": n,
        "clean_mean_logp":    float(np.mean(c_scores)),
        "violated_mean_logp": float(np.mean(v_scores)),
        "gap":                float(np.mean(c_scores) - np.mean(v_scores)),
        "best_f1":            sub_roc["best_f1"],
        "auc_roc":            sub_roc["auc_roc"],
    }
    print(f"  {mut:20s}  n={n:3d}  gap={mutation_breakdown[mut]['gap']:+.4f}  "
          f"F1={sub_roc['best_f1']:.3f}  AUC={sub_roc['auc_roc']:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# § 7  PRINT RESULTS TABLE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 3  ANOMALY DETECTION  (300 clean + 300 violated)")
print("=" * 70)
print(f"\n{'Model':<12} {'Clean logP':>11} {'Viol logP':>10} {'Gap':>8} "
      f"{'F1':>7} {'Prec':>7} {'Recall':>7} {'AUC-ROC':>8}")
print("-" * 70)

for name, scores_c, scores_v, roc in [
    ("GPT-nano",  gpt_clean_scores, gpt_violated_scores, gpt_roc),
    ("Bigram",    bg_clean_scores,  bg_violated_scores,  bg_roc),
]:
    gap = np.mean(scores_c) - np.mean(scores_v)
    m   = roc["best_metrics"]
    print(f"{name:<12} {np.mean(scores_c):>11.4f} {np.mean(scores_v):>10.4f} "
          f"{gap:>+8.4f} {roc['best_f1']:>7.3f} {m['precision']:>7.3f} "
          f"{m['recall']:>7.3f} {roc['auc_roc']:>8.3f}")

print("=" * 70)
print(f"\nGPT: val_loss={best_val_loss:.4f}  ppl={math.exp(best_val_loss):.3f}  "
      f"params={n_params/1e6:.2f}M  train_time={gpt_train_time:.1f}s")
print(f"Bigram: train_time={bg_train_time:.3f}s  transitions={n_transitions}")

gpt_m = gpt_roc["best_metrics"]
bg_m  = bg_roc["best_metrics"]
print(f"\nBest-F1 threshold sweep:")
print(f"  GPT    thresh={gpt_roc['best_threshold']:.4f}  "
      f"TP={gpt_m['tp']} FP={gpt_m['fp']} FN={gpt_m['fn']} TN={gpt_m['tn']}")
print(f"  Bigram thresh={bg_roc['best_threshold']:.4f}   "
      f"TP={bg_m['tp']} FP={bg_m['fp']} FN={bg_m['fn']} TN={bg_m['tn']}")

# ════════════════════════════════════════════════════════════════════════════
# § 8  SAVE JSON
# ════════════════════════════════════════════════════════════════════════════

# Drop full ROC curve from JSON output (keep summary only)
gpt_roc_summary = {k: v for k, v in gpt_roc.items() if k != "roc_curve"}
bg_roc_summary  = {k: v for k, v in bg_roc.items()  if k != "roc_curve"}

results = {
    "meta": {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device":        DEVICE,
        "gpt_iters":     args.iters,
        "batch_size":    args.batch_size,
        "n_params":      n_params,
        "vocab_size":    VOCAB_SIZE,
        "train_seqs":    len(train_seqs),
        "holdout_seqs":  len(holdout_seqs),
        "n_clean":       len(task3_clean),
        "n_violated":    len(task3_violated),
        "mutation_breakdown_counts": dict(mut_counts),
    },
    "gpt": {
        "train_time_s":    gpt_train_time,
        "best_val_loss":   best_val_loss,
        "val_ppl":         float(math.exp(best_val_loss)),
        "clean_mean_logp": float(np.mean(gpt_clean_scores)),
        "clean_std_logp":  float(np.std(gpt_clean_scores)),
        "violated_mean_logp": float(np.mean(gpt_violated_scores)),
        "violated_std_logp":  float(np.std(gpt_violated_scores)),
        "logp_gap":        float(np.mean(gpt_clean_scores) - np.mean(gpt_violated_scores)),
        "roc": gpt_roc_summary,
        "per_mutation": mutation_breakdown,
    },
    "bigram": {
        "train_time_s":    bg_train_time,
        "n_transitions":   n_transitions,
        "clean_mean_logp": float(np.mean(bg_clean_scores)),
        "clean_std_logp":  float(np.std(bg_clean_scores)),
        "violated_mean_logp": float(np.mean(bg_violated_scores)),
        "violated_std_logp":  float(np.std(bg_violated_scores)),
        "logp_gap":        float(np.mean(bg_clean_scores) - np.mean(bg_violated_scores)),
        "roc": bg_roc_summary,
    },
}

out_path = Path(__file__).parent / "results_anomaly.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nResults saved → {out_path}")
print("\nDONE.")
