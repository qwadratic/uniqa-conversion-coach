"""
lstm_baseline.py — Tax-style LSTM for Infineon next-step prediction.

Unidirectional LSTM (causal — correct for online next-step prediction).
Same flat-buffer sliding-window training as infineon_gpt.py → fair comparison.

Usage:
  python karpathy-experiment/lstm_baseline.py [--device mps|cpu|cuda] [--iters N]

Outputs:
  karpathy-experiment/results_lstm.json  — eval metrics
"""
from __future__ import annotations
import argparse, csv, json, math, random, time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
    default="/tmp/zero_one_hack_01/tracks/industrial-infineon/training_data")
parser.add_argument("--device", default=(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"))
parser.add_argument("--iters",      default=2000, type=int)
parser.add_argument("--batch_size", default=64,   type=int)
parser.add_argument("--hidden",     default=256,  type=int)
parser.add_argument("--n_layers",   default=2,    type=int)
parser.add_argument("--d_embd",     default=64,   type=int)
parser.add_argument("--dropout",    default=0.3,  type=float)
parser.add_argument("--lr",         default=1e-3, type=float)
parser.add_argument("--block_size", default=256,  type=int)
parser.add_argument("--seed",       default=1337, type=int)
parser.add_argument("--out", default="karpathy-experiment/results_lstm.json")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

DEVICE     = args.device
BATCH_SIZE = args.batch_size
BLOCK_SIZE = args.block_size
MAX_ITERS  = args.iters
EVAL_EVERY = max(100, MAX_ITERS // 20)

print(f"device={DEVICE}  iters={MAX_ITERS}  batch={BATCH_SIZE}  "
      f"hidden={args.hidden}  layers={args.n_layers}")

# ── Data: load templates + generate variants ──────────────────────────────────
DATA_DIR = Path(args.data_dir)

def load_template_csv(fpath: Path) -> list[str]:
    """Load single-sequence template CSV (header = STEP column)."""
    steps = []
    with open(fpath) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                s = row[0].strip().strip('"')
                if s:
                    steps.append(s)
    return steps

# Template files (single sequences)
TEMPLATES = {
    "IC":   [("IC",   DATA_DIR / "syntheticIC.csv"),
             ("IC",   DATA_DIR / "IC_Longdescr.csv")],
    "IGBT": [("IGBT", DATA_DIR / "syntheticIGBT.csv"),
             ("IGBT", DATA_DIR / "IGBT_Longdescr.csv")],
}

# Optional-step keywords: these steps can be dropped without breaking the sequence
OPTIONAL_KW  = {"MEASURE", "INSPECT", "MONITOR", "SCAN", "VERIFY", "CHECK",
                 "REVIEW", "AUDIT"}
CRITICAL_KW  = {"ETCH", "DEPOSIT", "IMPLANT", "ANNEAL", "SHIP", "RECEIVE",
                 "THERMAL", "OXIDATION", "DOPE", "PATTERN", "EXPOSE", "DEVELOP",
                 "CMP", "METAL", "VIA", "CONTACT", "TEST", "CLEAN"}

def is_optional(step: str) -> bool:
    words = set(step.upper().split())
    return bool(words & OPTIONAL_KW) and not bool(words & CRITICAL_KW)

def generate_variants(template: list[str], family: str,
                      n: int = 500, seed: int = 0) -> list[tuple[str, list[str]]]:
    """Generate n variants of a template by randomly dropping optional steps."""
    rng = random.Random(seed)
    variants: list[tuple[str, list[str]]] = [(family, list(template))]
    for i in range(n - 1):
        variant = []
        for step in template:
            if is_optional(step):
                # Drop with probability sampled per-variant
                drop_p = rng.uniform(0.0, 0.20)
                if rng.random() < drop_p:
                    continue
            variant.append(step)
        if len(variant) >= 20:
            variants.append((family, variant))
    return variants

print("\nLoading templates and generating variants...")
all_sequences: list[list[str]] = []
family_labels: list[str]       = []
loaded_templates: list[tuple[str, list[str]]] = []

for fam, tmpl_files in TEMPLATES.items():
    for fam_label, fpath in tmpl_files:
        if fpath.exists():
            tmpl = load_template_csv(fpath)
            loaded_templates.append((fam_label, tmpl))
            print(f"  {fpath.name}: {len(tmpl)} steps")

if not loaded_templates:
    raise RuntimeError(f"No template CSVs found in {DATA_DIR}")

# Also check for pre-existing variant CSVs (e.g. MOSFET_variants.csv)
for fname in ["MOSFET_variants.csv", "IGBT_variants.csv", "IC_variants.csv"]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        family = fname.split("_")[0]
        seqs: dict[str, list[str]] = defaultdict(list)
        with open(fpath) as f:
            for row in csv.DictReader(f):
                seqs[row["SEQUENCE_ID"]].append(row["STEP"].strip())
        n = len(seqs)
        all_sequences.extend(seqs.values())
        family_labels.extend([family] * n)
        print(f"  {fname}: {n} sequences (pre-existing variants)")

if not all_sequences:
    # Generate variants from templates
    print("  No pre-built variants found — generating synthetic variants...")
    N_VARIANTS = 600  # per template
    for i, (fam, tmpl) in enumerate(loaded_templates):
        variants = generate_variants(tmpl, fam, n=N_VARIANTS, seed=args.seed + i * 13)
        all_sequences.extend([v for _, v in variants])
        family_labels.extend([fam] * len(variants))
        print(f"    {fam}: generated {len(variants)} variants from {len(tmpl)}-step template")
else:
    print(f"  Using pre-existing variants ({len(all_sequences)} sequences total)")
    # Also augment with template variants if few sequences
    if len(all_sequences) < 500:
        for i, (fam, tmpl) in enumerate(loaded_templates):
            variants = generate_variants(tmpl, fam, n=200, seed=args.seed + i * 7)
            all_sequences.extend([v for _, v in variants])
            family_labels.extend([fam] * len(variants))

print(f"\nTotal sequences: {len(all_sequences)}")
lens = [len(s) for s in all_sequences]
print(f"  Length distribution: min={min(lens)} mean={sum(lens)//len(lens)} max={max(lens)}")

# ── Tokeniser ─────────────────────────────────────────────────────────────────
all_steps  = sorted(set(s for seq in all_sequences for s in seq))
step_to_id = {s: i + 1 for i, s in enumerate(all_steps)}  # 0 = PAD
id_to_step = ["<PAD>"] + all_steps
BOS_ID     = len(all_steps) + 1
EOS_ID     = len(all_steps) + 2
VOCAB_SIZE = len(all_steps) + 3  # PAD + steps + BOS + EOS

print(f"\nVocab: {len(all_steps)} unique steps + PAD/BOS/EOS = {VOCAB_SIZE} tokens")

def encode(seq: list[str]) -> list[int]:
    return [BOS_ID] + [step_to_id[s] for s in seq] + [EOS_ID]

def decode(ids: list[int]) -> list[str]:
    out = []
    for i in ids:
        if i in (BOS_ID, 0):
            continue
        if i == EOS_ID:
            break
        if 1 <= i < len(id_to_step):
            out.append(id_to_step[i])
    return out

# ── Train/val split (seed=1337, 90/10) ───────────────────────────────────────
random.shuffle(all_sequences)
split = int(0.9 * len(all_sequences))
train_seqs = all_sequences[:split]
val_seqs   = all_sequences[split:]

# Hold out 10% as "held-out" for Task-1 eval (top-k accuracy)
val_split    = int(0.5 * len(val_seqs))
eval_seqs    = val_seqs[val_split:]    # strictly for eval, never trained on
val_seqs_lm  = val_seqs[:val_split]   # used for val loss estimate

print(f"Split: {len(train_seqs)} train / {len(val_seqs_lm)} val / {len(eval_seqs)} eval")

def pack(seqs: list[list[str]]) -> torch.Tensor:
    flat: list[int] = []
    for seq in seqs:
        flat.extend(encode(seq))
    return torch.tensor(flat, dtype=torch.long)

train_data = pack(train_seqs)
val_data   = pack(val_seqs_lm)
print(f"  Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix   = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x    = torch.stack([data[i : i + BLOCK_SIZE]     for i in ix])
    y    = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ── Model: causal (unidirectional) LSTM ──────────────────────────────────────
# Unidirectional is correct for online next-step prediction.
# BiLSTM would cheat by seeing future tokens during training.

class InfineonLSTM(nn.Module):
    def __init__(self, vocab_size: int, d_embd: int, d_hidden: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.embd   = nn.Embedding(vocab_size, d_embd, padding_idx=0)
        self.drop   = nn.Dropout(dropout)
        self.lstm   = nn.LSTM(d_embd, d_hidden, n_layers,
                              dropout=(dropout if n_layers > 1 else 0.0),
                              batch_first=True)
        self.norm   = nn.LayerNorm(d_hidden)
        self.head   = nn.Linear(d_hidden, vocab_size, bias=False)

    def forward(self, x: torch.Tensor,
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb   = self.drop(self.embd(x))            # (B, T, d_embd)
        h, _  = self.lstm(emb)                     # (B, T, d_hidden)
        h     = self.norm(self.drop(h))
        logits = self.head(h)                      # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=0)
        return logits, loss

model = InfineonLSTM(VOCAB_SIZE, args.d_embd, args.hidden, args.n_layers, args.dropout)
model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params/1e3:.1f}K parameters on {DEVICE}")
print(f"  d_embd={args.d_embd}  d_hidden={args.hidden}  "
      f"n_layers={args.n_layers}  dropout={args.dropout}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# Cosine annealing (more stable than linear for LSTMs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=1e-5)

# ── Eval loss helper ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(iters: int = 40) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ── Training loop ──────────────────────────────────────────────────────────────
print(f"\nTraining for {MAX_ITERS} iters...")
t0 = time.time()
best_val = float("inf")
train_losses: list[float] = []

for step in range(MAX_ITERS):
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (MAX_ITERS - step)
        print(f"  step {step:5d} | train={losses['train']:.4f} val={losses['val']:.4f} "
              f"| elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        if losses["val"] < best_val:
            best_val = losses["val"]

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

total_time = time.time() - t0
print(f"\nDone. {total_time:.1f}s  best_val_loss={best_val:.4f}  "
      f"perplexity={math.exp(best_val):.3f}")

# ── Task-1 Eval: top-k next-step accuracy ─────────────────────────────────────
# At each position in eval sequences, predict next step → top-1/3/5 accuracy.
print("\n── EVAL: Task 1 — Next-step prediction ──────────────────────────────")

model.eval()
# Evaluate at multiple cut points: 20%, 40%, 60%, 80% of each sequence
CUT_POINTS = [0.20, 0.40, 0.60, 0.80]

top1_hits = {cp: 0 for cp in CUT_POINTS}
top3_hits = {cp: 0 for cp in CUT_POINTS}
top5_hits = {cp: 0 for cp in CUT_POINTS}
mrr_sum   = {cp: 0.0 for cp in CUT_POINTS}
total_preds = {cp: 0 for cp in CUT_POINTS}

# Per-position accuracy (averaged over all sequences)
pos_top1_all: list[int] = []
pos_top1_n:   list[int] = []

# Also collect all positions for overall accuracy
all_top1 = 0
all_top3 = 0
all_top5 = 0
all_n    = 0

with torch.no_grad():
    for seq in eval_seqs:
        enc = encode(seq)  # [BOS, s1, s2, ..., sN, EOS]
        if len(enc) < 4:
            continue
        t_enc = torch.tensor(enc, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Forward the full prefix once → logits for every position
        logits, _ = model(t_enc[:, :-1])  # (1, T-1, V)
        logits = logits[0]                 # (T-1, V)
        targets = t_enc[0, 1:]             # (T-1,)  — shifted by 1

        # Overall per-token accuracy (all positions)
        for pos in range(len(logits)):
            gold = targets[pos].item()
            if gold == 0:  # PAD
                continue
            topk = logits[pos].topk(5).indices.tolist()
            all_top1 += int(gold == topk[0])
            all_top3 += int(gold in topk[:3])
            all_top5 += int(gold in topk[:5])
            all_n += 1

        # Cut-point accuracy
        seq_len = len(seq)  # original sequence length (no BOS/EOS)
        for cp in CUT_POINTS:
            cut_idx = max(1, int(cp * seq_len))  # number of input steps (after BOS)
            # Predict step at position cut_idx (0-indexed in original seq)
            if cut_idx >= len(logits):
                continue
            gold = targets[cut_idx].item()
            if gold == 0:
                continue
            logit_at_cut = logits[cut_idx]
            topk = logit_at_cut.topk(min(5, VOCAB_SIZE)).indices.tolist()

            top1_hits[cp]    += int(gold == topk[0])
            top3_hits[cp]    += int(gold in topk[:3])
            top5_hits[cp]    += int(gold in topk[:5])
            # MRR
            rank = next((r+1 for r, t in enumerate(topk) if t == gold), 0)
            mrr_sum[cp]      += (1.0 / rank) if rank > 0 else 0.0
            total_preds[cp]  += 1

# Print per-cut-point results
print(f"\n  {'Cut':>6} | {'Top-1':>6} {'Top-3':>6} {'Top-5':>6} {'MRR':>6} | n")
print(f"  {'------':>6}-+-{'------':>6}-{'------':>6}-{'------':>6}-{'------':>6}-+---")
cut_metrics = {}
for cp in CUT_POINTS:
    n = total_preds[cp]
    if n == 0:
        continue
    t1  = top1_hits[cp] / n
    t3  = top3_hits[cp] / n
    t5  = top5_hits[cp] / n
    mrr = mrr_sum[cp] / n
    print(f"  {cp*100:5.0f}% | {t1:6.3f} {t3:6.3f} {t5:6.3f} {mrr:6.3f} | {n}")
    cut_metrics[f"cut_{int(cp*100)}"] = dict(top1=t1, top3=t3, top5=t5, mrr=mrr, n=n)

# Overall per-token accuracy
if all_n > 0:
    t1o = all_top1 / all_n
    t3o = all_top3 / all_n
    t5o = all_top5 / all_n
    print(f"\n  Overall per-token:  top-1={t1o:.3f}  top-3={t3o:.3f}  "
          f"top-5={t5o:.3f}  (n={all_n})")

# ── Task-2 Eval: sequence completion from 60% and 80% cut ────────────────────
print("\n── EVAL: Task 2 — Sequence completion (greedy decode) ──────────────")
CUT2 = [0.60, 0.80]

def greedy_complete(prefix_ids: list[int], max_new: int = 80) -> list[int]:
    """Greedy autoregressive completion from a prefix."""
    t = torch.tensor(prefix_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            inp = t[:, -BLOCK_SIZE:]
            logits, _ = model(inp)
            next_id = logits[0, -1].argmax().item()
            if next_id == EOS_ID or next_id == 0:
                break
            generated.append(next_id)
            t = torch.cat([t, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return generated

def token_accuracy(pred: list[int], gold: list[int]) -> float:
    n = min(len(pred), len(gold))
    if n == 0:
        return 0.0
    return sum(p == g for p, g in zip(pred, gold)) / max(len(pred), len(gold))

comp_metrics: dict[str, dict] = {}
for cp in CUT2:
    tok_accs = []
    for seq in eval_seqs[:50]:  # limit for speed
        enc = encode(seq)
        if len(enc) < 6:
            continue
        cut = max(2, int(cp * len(seq)))
        prefix = enc[:cut + 1]  # +1 for BOS
        gold_cont = enc[cut + 1:]  # remainder
        pred_cont = greedy_complete(prefix, max_new=len(gold_cont) + 10)
        tok_accs.append(token_accuracy(pred_cont, gold_cont))
    avg_tok = sum(tok_accs) / max(len(tok_accs), 1)
    print(f"  cut={cp:.0%}  token_acc={avg_tok:.3f}  (n={len(tok_accs)})")
    comp_metrics[f"cut_{int(cp*100)}"] = dict(token_accuracy=avg_tok, n=len(tok_accs))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"  Model:       LSTM  {n_params/1e3:.1f}K params")
print(f"  Train time:  {total_time:.1f}s  ({total_time/60:.1f} min)")
print(f"  Best val loss: {best_val:.4f}  perplexity={math.exp(best_val):.2f}")
if all_n:
    print(f"  Overall top-1: {t1o:.1%}  top-3: {t3o:.1%}  top-5: {t5o:.1%}")

cut60 = cut_metrics.get("cut_60", {})
if cut60:
    print(f"  @60% cut  top-1={cut60['top1']:.1%}  top-5={cut60['top5']:.1%}")
cut80 = cut_metrics.get("cut_80", {})
if cut80:
    print(f"  @80% cut  top-1={cut80['top1']:.1%}  top-5={cut80['top5']:.1%}")
print("="*60)

# ── Save checkpoint ───────────────────────────────────────────────────────────
ckpt_dir = Path("karpathy-experiment/checkpoints")
ckpt_dir.mkdir(exist_ok=True)
ckpt_path = ckpt_dir / "lstm.pt"
torch.save({
    "model_state": model.state_dict(),
    "step_to_id":  step_to_id,
    "id_to_step":  id_to_step,
    "BOS_ID":      BOS_ID,
    "EOS_ID":      EOS_ID,
    "VOCAB_SIZE":  VOCAB_SIZE,
    "cfg": dict(d_embd=args.d_embd, d_hidden=args.hidden,
                n_layers=args.n_layers, dropout=args.dropout),
    "best_val_loss": best_val,
    "train_time_s":  total_time,
}, ckpt_path)
print(f"\nCheckpoint saved: {ckpt_path}")

# ── Save JSON results ─────────────────────────────────────────────────────────
results = {
    "model":          "LSTM",
    "params_k":       round(n_params / 1e3, 1),
    "d_embd":         args.d_embd,
    "d_hidden":       args.hidden,
    "n_layers":       args.n_layers,
    "dropout":        args.dropout,
    "train_iters":    MAX_ITERS,
    "train_time_s":   round(total_time, 1),
    "best_val_loss":  round(best_val, 4),
    "perplexity":     round(math.exp(best_val), 3),
    "task1_overall":  dict(
        top1=round(t1o, 4) if all_n else None,
        top3=round(t3o, 4) if all_n else None,
        top5=round(t5o, 4) if all_n else None,
    ) if all_n else {},
    "task1_cut_points": {k: {m: round(v, 4) if isinstance(v, float) else v
                             for m, v in vm.items()}
                         for k, vm in cut_metrics.items()},
    "task2_completion": {k: {m: round(v, 4) if isinstance(v, float) else v
                              for m, v in vm.items()}
                         for k, vm in comp_metrics.items()},
    "vocab_size": VOCAB_SIZE,
    "n_train_seqs": len(train_seqs),
    "n_eval_seqs":  len(eval_seqs),
}

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {out_path}")
print("\nDONE.")
