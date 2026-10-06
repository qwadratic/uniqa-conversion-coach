"""
infineon_gpt.py — microgpt.py × Infineon semiconductor data, PyTorch edition.

Karpathy's microgpt.py architecture (decoder-only GPT, RMSNorm, ReLU, no biases,
causal attention) ported to PyTorch — runs on CUDA, MPS (Apple Silicon), or CPU.

Step-level tokenisation: each STEP string = one token (~200-vocab total).
Context window 256 covers the longest sequence (155 steps + BOS/EOS + margin).

Usage:
  python infineon_gpt.py [--data_dir DIR] [--config mac|nano|small] [--iters N]
                         [--batch_size B] [--device cuda|mps|cpu]

Configs:
  mac   — 1.24M params, batch=64,  1000 iters  — ~4 min MPS | ~17 min CPU
  nano  — 1.24M params, batch=128, 2000 iters  — ~15 min CPU | <1 min A100
  small — 9.6M  params, batch=256, 10000 iters — ~90 min CPU | ~7 min A100

Output: 5 random training sequences  +  5 generated sequences.
"""
from __future__ import annotations
import argparse
import csv
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
    default="/tmp/zero_one_hack_01/tracks/industrial-infineon/training_data")
parser.add_argument("--config", default="mac", choices=["mac", "nano", "small"])
parser.add_argument("--iters",  default=None, type=int)
parser.add_argument("--batch_size", default=None, type=int)
_default_device = ("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available()
                   else "cpu")
parser.add_argument("--device", default=_default_device)
parser.add_argument("--seed",   default=42, type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
DEVICE = args.device
print(f"device={DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif DEVICE == "mps":
    print(f"  Apple Silicon MPS (Metal Performance Shaders)")

# ── Hyperparameter configs ────────────────────────────────────────────────────
CONFIGS = {
    # ~1.24M params — local Mac: ~4 min MPS | ~17 min CPU
    "mac":   dict(n_layer=6,  n_embd=128, n_head=8,  block_size=256,
                  batch_size=64,  max_iters=1000,  lr=3e-4),
    # ~1.24M params — A100 smoke test: <1 min | CPU: ~50 min
    "nano":  dict(n_layer=6,  n_embd=128, n_head=8,  block_size=256,
                  batch_size=128, max_iters=2000,  lr=3e-4),
    # ~9.6M params — A100 proper run: ~7 min | CPU: ~90 min
    "small": dict(n_layer=12, n_embd=256, n_head=8,  block_size=256,
                  batch_size=256, max_iters=10000, lr=3e-4),
}
cfg = CONFIGS[args.config]
if args.iters is not None:  cfg["max_iters"]  = args.iters
if args.batch_size:         cfg["batch_size"] = args.batch_size

BLOCK_SIZE  = cfg["block_size"]
BATCH_SIZE  = cfg["batch_size"]
MAX_ITERS   = cfg["max_iters"]
LR          = cfg["lr"]
N_LAYER     = cfg["n_layer"]
N_EMBD      = cfg["n_embd"]
N_HEAD      = cfg["n_head"]
EVAL_EVERY  = max(100, MAX_ITERS // 20)

print(f"\nConfig '{args.config}': n_layer={N_LAYER} n_embd={N_EMBD} n_head={N_HEAD} "
      f"block_size={BLOCK_SIZE} batch={BATCH_SIZE} iters={MAX_ITERS}")

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = Path(args.data_dir)
CSV_FILES = ["MOSFET_variants.csv", "IGBT_variants.csv", "IC_variants.csv"]

print("\nLoading sequences...")
all_sequences: list[list[str]] = []
family_labels: list[str] = []

for fname in CSV_FILES:
    fpath = DATA_DIR / fname
    if not fpath.exists():
        print(f"  WARNING: {fpath} not found, skipping")
        continue
    family = fname.split("_")[0]
    seqs: dict[str, list[str]] = defaultdict(list)
    with open(fpath) as f:
        for row in csv.DictReader(f):
            seqs[row["SEQUENCE_ID"]].append(row["STEP"].strip())
    n = len(seqs)
    all_sequences.extend(seqs.values())
    family_labels.extend([family] * n)
    print(f"  {fname}: {n} sequences, vocab scan...")

print(f"  Total: {len(all_sequences)} sequences")

# ── Step-level tokeniser ──────────────────────────────────────────────────────
all_steps = sorted(set(s for seq in all_sequences for s in seq))
step_to_id = {s: i for i, s in enumerate(all_steps)}
id_to_step = all_steps  # index → step string
BOS_ID     = len(all_steps)
EOS_ID     = len(all_steps) + 1
VOCAB_SIZE = len(all_steps) + 2  # steps + BOS + EOS

print(f"Vocabulary: {len(all_steps)} unique steps + BOS/EOS = {VOCAB_SIZE} tokens")

def encode(seq: list[str]) -> list[int]:
    return [BOS_ID] + [step_to_id[s] for s in seq] + [EOS_ID]

def decode(ids: list[int]) -> list[str]:
    out = []
    for i in ids:
        if i == BOS_ID:
            continue
        if i == EOS_ID:
            break
        if 0 <= i < len(id_to_step):
            out.append(id_to_step[i])
    return out

# ── 5 random training examples (before training) ──────────────────────────────
print("\n" + "="*70)
print("5 RANDOM TRAINING SEQUENCES (first 8 steps shown)")
print("="*70)
sample_indices = random.sample(range(len(all_sequences)), 5)
for i, idx in enumerate(sample_indices):
    seq = all_sequences[idx]
    preview = " → ".join(seq[:8])
    if len(seq) > 8:
        preview += f" ... ({len(seq)} steps total)"
    print(f"  [{i+1}] {family_labels[idx]:6s}  {preview}")
print("="*70)

# ── Dataset packing ───────────────────────────────────────────────────────────
# Encode all sequences and concatenate into one big flat token buffer.
# This is the standard nanoGPT approach: sample random windows for training.
random.shuffle(all_sequences)
split_idx = int(0.9 * len(all_sequences))
train_seqs = all_sequences[:split_idx]
val_seqs   = all_sequences[split_idx:]

def pack_to_tensor(seqs: list[list[str]]) -> torch.Tensor:
    flat: list[int] = []
    for seq in seqs:
        flat.extend(encode(seq))
    return torch.tensor(flat, dtype=torch.long)

print("\nEncoding and packing token buffer...")
train_data = pack_to_tensor(train_seqs)
val_data   = pack_to_tensor(val_seqs)
print(f"  train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix   = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x    = torch.stack([data[i:i+BLOCK_SIZE]     for i in ix])
    y    = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ── Model ─────────────────────────────────────────────────────────────────────
# Faithful to microgpt.py: RMSNorm, ReLU MLP, no biases, causal self-attn.

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.head_dim = n_embd // n_head
        # no bias — matches microgpt
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd,     bias=False)
        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y   = att @ v  # (B, n_head, T, head_dim)
        y   = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """ReLU MLP (no biases) — matches microgpt's fc1/fc2 with ReLU activation."""
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd,     4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = RMSNorm(n_embd)
        self.mlp  = MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer:    int
    n_embd:     int
    n_head:     int


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte    = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe    = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [Block(cfg.n_embd, cfg.n_head, cfg.block_size) for _ in range(cfg.n_layer)]
        )
        self.ln_f   = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying
        self.wte.weight = self.lm_head.weight

        # init
        self.apply(self._init_weights)
        # scale residual projections like GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x      = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.9, top_k: int = 40) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            # stop at EOS
            if next_id.item() == EOS_ID:
                break
        return idx


# ── Instantiate + parameter count ─────────────────────────────────────────────
model_cfg = GPTConfig(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_embd=N_EMBD,
    n_head=N_HEAD,
)
model = GPT(model_cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params/1e6:.2f}M parameters on {DEVICE}")

if DEVICE == "cuda":
    # Check VRAM usage after model init
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM after init: {mem_alloc:.2f} / {mem_total:.1f} GB")

# ── Optimizer (Adam, matches microgpt lr=0.01 but we use 3e-4 for stability) ──
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99),
                               weight_decay=0.1)

# Linear LR decay (matches microgpt's lr_t = lr * (1 - step/num_steps))
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=MAX_ITERS
)

# Use bf16 on A100 (much faster, no precision loss for this task)
USE_BF16 = (DEVICE == "cuda" and
            torch.cuda.is_bf16_supported() and
            torch.cuda.get_device_capability()[0] >= 8)
AUTOCAST_ON = USE_BF16   # MPS: fp32 only (bfloat16 not in MPS autocast)
print(f"  precision: {'bf16 autocast' if AUTOCAST_ON else 'fp32'}")

@torch.no_grad()
def estimate_loss(iters: int = 50) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(iters):
            xb, yb = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AUTOCAST_ON):
                _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ── Training loop ──────────────────────────────────────────────────────────────
print(f"\nTraining for {MAX_ITERS} iterations...")
t0 = time.time()
best_val_loss = float("inf")

_iter_times: list[float] = []
_t_iter = time.time()

for step in range(MAX_ITERS):
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        ms_iter = (sum(_iter_times[-EVAL_EVERY:]) / max(len(_iter_times[-EVAL_EVERY:]), 1) * 1000
                   if _iter_times else 0.0)
        eta = ms_iter / 1000 * (MAX_ITERS - step)
        mem = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
        print(f"  step {step:5d} | train={losses['train']:.4f} val={losses['val']:.4f} "
              f"| {ms_iter:.0f}ms/it | elapsed={elapsed:.0f}s ETA={eta:.0f}s"
              + (f" | vram={mem:.2f}GB" if DEVICE == "cuda" else ""))
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

    xb, yb = get_batch("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AUTOCAST_ON):
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    _iter_times.append(time.time() - _t_iter)
    _t_iter = time.time()

total_time = time.time() - t0
print(f"\nTraining done. {total_time:.1f}s  best_val_loss={best_val_loss:.4f}")

# ── Generate 5 sequences (batched BOS start, CPU fallback for MPS speed) ─────
print("\n" + "="*70)
print("5 GENERATED SEQUENCES")
print("="*70)
# Token-by-token autoregressive on MPS is ~670ms/fwd — move to CPU for generation.
gen_device = "cpu" if DEVICE == "mps" else DEVICE
gen_model = model.cpu() if DEVICE == "mps" else model
gen_model.eval()
# Update bias buffers (registered buffers stay on old device after .cpu())
t_gen = time.time()
# Generate all 5 in one batched pass: start col = BOS, grow right
N_GEN = 5
MAX_GEN = 160  # max steps to generate
idx = torch.full((N_GEN, 1), BOS_ID, dtype=torch.long, device=gen_device)
active = [True] * N_GEN  # tracks which sequences haven't hit EOS yet
with torch.no_grad():
    for _ in range(MAX_GEN):
        if not any(active):
            break
        idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
        logits, _ = gen_model(idx_cond)
        logits = logits[:, -1, :] / 0.9   # temperature
        # top-k
        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, 1)
        for s in range(N_GEN):
            if not active[s]:
                next_ids[s, 0] = EOS_ID
            elif next_ids[s, 0].item() == EOS_ID:
                active[s] = False
        idx = torch.cat([idx, next_ids], dim=1)

for i in range(N_GEN):
    steps = decode(idx[i].tolist())
    preview = " → ".join(steps[:8])
    if len(steps) > 8:
        preview += f" ... ({len(steps)} steps total)"
    print(f"  [{i+1}] {preview}")

print(f"  (generation: {time.time()-t_gen:.1f}s on {gen_device})")
print("="*70)
print("\nDONE.")
