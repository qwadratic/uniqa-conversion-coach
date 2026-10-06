# Infineon Karpathy — Experiment Plan

> Branch: `industrial-experiments`  
> Author: planner subagent, 2026-06-03  
> Audience: ML worker subagent implementing experiments A/B/C

---

## 0. Context Recap (one-screen)

- **Track:** Infineon Industrial AI — predict next step / complete sequence / detect anomalies on semiconductor fab process traces.
- **Data:** `/tmp/zero_one_hack_01/tracks/industrial-infineon/training_data/{MOSFET,IGBT,IC}_variants.csv` — 3000 sequences, ~198-step vocab, avg 115–148 steps/seq.
- **Baseline already shipped:** `infineon_gpt.py --config mac` (1.24M params, MPS, ~4–6 min, **val_loss=0.345**). That's perplexity ≈ 1.41 over a 200-token vocab — already extremely sharp.
- **Eval files NOT yet released** (`eval_input_valid.csv`, `eval_input_anomaly.csv`, `eval_metrics.py`) — organizers release on event day. We therefore build our own held-out + synthetic-violation eval that *mirrors* the published submission format and metric definitions in `scout_repo.md`.
- **10 forbidden rules** are documented verbatim in `scout_repo.md §9` — sufficient to synthesize Task-3 violations ourselves.
- **Block-level categories** for Task 2 are listed (`LITHO, ETCH, DOPING_THERMAL, DEPOSITION, PLANARIZATION, VIA, PASSIVATION, BACKSIDE, METROLOGY_TEST, LOGISTICS, OTHER`).

### Decisions up front (decided here, not by worker)

| Decision | Choice | Why |
|---|---|---|
| SOTA PoC | **Tax-style 2-layer LSTM with learned embeddings** (`d_embd=64`, `d_hidden=128`) | Most data-efficient at 3k examples per `sota_research.md`. Fits in <150 LOC. Beats bigram by 10–15pp in published BPI benchmarks. Different inductive bias from transformer → meaningful comparison. |
| Reject | GPT-2 fine-tune, pm4py hybrid, full ProcessTransformer port | Overkill / overlapping with what nanoGPT already does / wall-clock blows past 15 min budget. |
| Scaling sweep | nano (1.24M ✅ done) + **mini (~0.6M, 2L)** + **small (9.6M)** *only if* mini→nano shows undersaturation | Lit says 3k examples → ≤3 layers optimal. Small at 12L likely overfits, but user explicitly asked. Run as stretch, with early-stopping on val. |
| Internal eval split | Hold out **300 sequences** (100 per family) **before any training**, seed=1337, frozen across all experiments | Same split for nano/mini/small/LSTM/bigram → apples-to-apples. |
| Internal Task-1 metric | Sample 5 cut points per held-out sequence (10%, 30%, 50%, 70%, 90%) → 1500 prediction events | Mirrors event format (0.6/0.8 cut) but denser → tighter confidence intervals. |
| Internal Task-2 metric | Cut at 60% **and** 80% (exact event format) | Direct comparability. |
| Internal Task-3 metric | Use 300 held-out as **positives**; synthesize 300 negatives by injecting 1 of 10 rule violations into copies of positives. | Mirrors event 600 valid + 387 invalid balance. |

### Expected outcomes (point estimates)

| Model | Params | Wall (MPS) | Top-1 next-step | Top-5 | Anomaly F1 |
|---|---|---|---|---|---|
| Bigram | <1k | <5s | **62–68%** (floor) | ~85% | ~0.55 (perplexity-thresh) |
| Trigram | ~50k | <10s | 70–76% | ~90% | ~0.65 |
| LSTM (Tax-style, SOTA) | ~250k | ~3–5 min | **84–88%** | 96–98% | ~0.85 |
| nanoGPT mini (2L 128d) | ~600k | ~3 min | 86–90% | 98% | ~0.88 |
| nanoGPT nano (6L 128d) ✅ | 1.24M | ~6 min | **88–92%** | 98–99% | ~0.90 |
| nanoGPT small (12L 256d) | 9.6M | ~45–60 min | 88–92% (overfit risk) | 99% | ~0.91 |

Headline message we want to land in the demo: **"4× smaller LSTM gets within 4pp of nanoGPT; nanoGPT beats bigram by 25+pp; small model overfits, validating literature."**

---

## 1. Repo Layout (what worker will create)

```
karpathy-experiment/
├── infineon_gpt.py             # ALREADY EXISTS — do NOT rewrite
├── EXPERIMENT_PLAN.md          # THIS FILE
├── eval/
│   ├── __init__.py
│   ├── split.py                # frozen train/held-out split (seed=1337)
│   ├── metrics.py              # top-k acc, MRR, edit dist, block acc, anomaly F1+AUC
│   ├── synth_anomalies.py      # inject 1 of 10 rule violations into a sequence
│   ├── block_categories.py     # step → category mapping (LITHO/ETCH/…)
│   └── run_eval.py             # CLI: --model {bigram,trigram,lstm,gpt} --ckpt PATH
├── baselines/
│   ├── bigram.py               # transition matrix; pickle to .pkl
│   └── trigram.py              # (2,1)-gram with backoff to bigram
├── sota/
│   └── tax_lstm.py             # 2-layer LSTM, learned embeddings, ~150 LOC
├── checkpoints/                # *.pt for trained models
└── results/
    ├── leaderboard.md          # auto-appended table
    └── runs/                   # JSON per-run: {model, params, time, metrics}
```

Worker MUST commit each file with a focused message on `industrial-experiments`.

---

## 2. Experiment A — Eval Harness  (PRIORITY 1, blocks B+C)

### A.1  Frozen split  (`eval/split.py`)

```python
# Pseudocode — worker fills in
SEED = 1337
def make_split(data_dir, n_holdout_per_family=100):
    """Returns (train_seqs, holdout_seqs) where holdout is stratified by family.
    Writes JSON manifest to results/split_manifest.json with sequence IDs so all
    later runs read the *same* held-out set."""
```

**Acceptance:** Re-running prints identical SHA256 of the held-out ID list.

### A.2  Metrics module  (`eval/metrics.py`)

Implement exactly the metrics from `scout_repo.md §2`:

| Task | Function | Returns |
|---|---|---|
| 1 | `next_step_metrics(preds: List[List[int]], gold: List[int]) -> dict` | `{top1, top3, top5, mrr}` |
| 2 | `completion_metrics(pred_seqs, gold_seqs) -> dict` | `{exact_match, norm_edit_dist, token_acc, block_acc}` |
| 3 | `anomaly_metrics(scores: List[float], labels: List[int]) -> dict` | `{accuracy, precision, recall, f1, roc_auc, confusion}` |

Block-level accuracy uses `eval/block_categories.py` (a hand-written dict mapping each of the ~198 step strings to one of 11 categories — worker derives this by keyword matching on step names, e.g. `"DEPOSIT" in step → DEPOSITION`).

**Edit distance:** Levenshtein on token IDs (write a 20-line DP — no `python-Levenshtein` dep).

### A.3  Synthetic anomalies  (`eval/synth_anomalies.py`)

For each of the 10 rules in `scout_repo.md §9`, write a function `inject_RULE_<name>(seq) -> seq'` that returns a mutated copy with that single rule violated. Examples:

- `RULE_DEP_NO_CLEAN`: find a `DEPOSIT *` step; delete the most recent cleaning step within the prior 12.
- `RULE_LITHO_LEVEL_SKIP`: find first `ALIGN MASK LEVEL 2`; rename to `ALIGN MASK LEVEL 3` (and matching EXPOSE).
- `RULE_SHIP_BEFORE_TEST`: swap `SHIP LOT` with `WAFER SORT TEST`.
- …(7 more — worker implements all 10 from the verbatim rule text)

Build the negative set: for each of 300 held-out positives, pick a random rule (uniform), inject, and label it `IS_VALID=0` with `PREDICTED_RULE=<rule_id>`. If injection is impossible for that sequence (e.g. no `DEPOSIT *` step) → skip and resample rule.

### A.4  Scoring driver  (`eval/run_eval.py`)

CLI:
```bash
python eval/run_eval.py --model bigram   --ckpt checkpoints/bigram.pkl
python eval/run_eval.py --model trigram  --ckpt checkpoints/trigram.pkl
python eval/run_eval.py --model lstm     --ckpt checkpoints/lstm.pt
python eval/run_eval.py --model gpt      --ckpt checkpoints/gpt_nano.pt
```

Each call:
1. Loads frozen held-out via `split.py`.
2. Runs all three tasks.
3. Appends one row to `results/leaderboard.md`:
   ```
   | model | params | wall_train_s | top1 | top3 | top5 | mrr | exact | edit | tok_acc | blk_acc | f1 | auc |
   ```
4. Dumps full JSON to `results/runs/<model>_<timestamp>.json`.

**Acceptance for Experiment A:** Bigram baseline scores show up in the leaderboard with all 13 columns populated, and re-running gives byte-identical numbers.

### A.5  Bigram + trigram baselines  (`baselines/{bigram,trigram}.py`)

Each <50 LOC. Bigram = `count(a→b) / count(a→*)`, smoothed with `+1` Laplace. Trigram backs off to bigram when unseen.

For Task 3, anomaly score = `mean(-log P(s_t | s_{t-1}))` over the full sequence. Threshold tuned on a 20% slice of held-out positives + injected negatives, then applied to remaining 80% for the reported F1/AUC.

---

## 3. Experiment B — Model Scaling

### B.0  Re-run nano under the eval harness

```bash
python infineon_gpt.py --config nano --device mps   # already trained, but save ckpt
# Worker patches infineon_gpt.py to torch.save({model, cfg, step_to_id}) at end → checkpoints/gpt_nano.pt
python eval/run_eval.py --model gpt --ckpt checkpoints/gpt_nano.pt
```

### B.1  Add "mini" config (PRIORITY)

Add to `CONFIGS` dict in `infineon_gpt.py`:
```python
"mini": dict(n_layer=2, n_embd=128, n_head=4, block_size=256,
             batch_size=64, max_iters=1000, lr=3e-4),
# ~600k params, expected ~3 min MPS
```

This directly tests the literature claim ("dataset_size/1000 ≈ optimal depth → 3 layers max"). If mini ≈ nano, the marginal value of depth is confirmed-low.

```bash
python infineon_gpt.py --config mini --device mps
python eval/run_eval.py --model gpt --ckpt checkpoints/gpt_mini.pt
```

### B.2  Small (stretch, conditional)

**Run small only if** `val_loss(mini) > val_loss(nano) + 0.05` (i.e. depth still helping). Otherwise skip — small will overfit at 3k examples and waste 45+ min.

If we do run:
```bash
# Sized to fit our 15-min budget per CLAUDE.md: cap iters at 3000, not 10000
python infineon_gpt.py --config small --device mps --iters 3000
```

Wall estimate: 9.6M / 1.24M × 6 min × (3000/1000 iters scaling) ≈ ~45 min. **Exceeds the 15-min budget in CLAUDE.md** — therefore default = **DO NOT RUN unless mini-vs-nano gap justifies it**, and if run, launch via `nohup ... &` with PID polling (never foreground).

### B.3  Minimum-model-that-beats-bigram answer

Worker fills in a sentence in `results/leaderboard.md`:
> "The smallest GPT that beats trigram's top-1 of XX% is **<config>** with <P> params."

If even mini beats trigram, that's the answer. If mini < trigram (unlikely), we add an even smaller "micro" config (1 layer, 64d, ~150k params) and try.

---

## 4. Experiment C — SOTA PoC (Tax LSTM)

### C.1  Why Tax LSTM over alternatives

| Candidate | LOC | Wall | Expected vs bigram | Data-efficient at 3k? | Verdict |
|---|---|---|---|---|---|
| **Tax 2-layer LSTM + embedding** | ~140 | ~3–5 min MPS | +15–20pp | **YES** (paper run on 3k–13k cases) | ✅ **PICK** |
| ProcessTransformer (encoder-only) | ~300 | ~5 min MPS | +18–22pp | Yes | ❌ too close to nanoGPT design → boring comparison |
| GPT-2 fine-tune | ~200+deps | hours | +15pp | NO (overfits) | ❌ wrong scale |
| pm4py hybrid | ~400+heavy dep | unclear | +2–4pp on LSTM | n/a | ❌ dependency hell |
| LSTM autoencoder for anomaly | ~250 | ~5 min | only addresses Task 3 | yes | ❌ partial coverage |

**Pick = Tax LSTM.** Reasons recap: (1) different inductive bias (sequential recurrence vs full self-attention) makes the comparison meaningful, not a "nanoGPT vs slightly different nanoGPT"; (2) most data-efficient family at our scale per literature; (3) fits in 150 LOC clean PyTorch; (4) the **same** trained LSTM can score all three tasks (perplexity for anomaly = free lunch, same as GPT).

### C.2  Implementation  (`sota/tax_lstm.py`, ≤150 LOC)

```python
class TaxLSTM(nn.Module):
    def __init__(self, vocab_size, d_embd=64, d_hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, d_embd)
        self.lstm = nn.LSTM(d_embd, d_hidden, n_layers,
                            dropout=dropout, batch_first=True)
        self.head = nn.Linear(d_hidden, vocab_size)
    def forward(self, x, targets=None):
        h, _ = self.lstm(self.embd(x))
        logits = self.head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.view(-1)) if targets is not None else None
        return logits, loss
```

Training loop reuses **the same `get_batch` and packed-token-buffer** from `infineon_gpt.py` (refactor those into `eval/data.py` so both `infineon_gpt.py` and `sota/tax_lstm.py` import them — same data, same split, same batching → fair comparison).

Hyperparams:
- `batch_size=64`, `block_size=256` (same as nano), `max_iters=1500`, `lr=1e-3`, AdamW
- ~250k params, expected ~3–5 min on MPS

```bash
python sota/tax_lstm.py --device mps
python eval/run_eval.py --model lstm --ckpt checkpoints/lstm.pt
```

### C.3  Comparison artifact

After all runs, write `results/comparison.md` containing:
1. The leaderboard table (auto-generated).
2. A markdown plot (ASCII bar chart or matplotlib PNG saved to `results/comparison.png`) of top-1 accuracy per model.
3. A 1-paragraph "what we learned":
   - Did nanoGPT beat LSTM? By how much?
   - Did small overfit (compare train vs val)?
   - Did mini hit the same accuracy as nano with half the params?
   - Anomaly F1: did the "free perplexity" approach work, or do we need a dedicated detector?

---

## 5. Execution Order (worker MUST follow this sequence)

1. **A.1 + A.2 + A.5 (bigram)** — get a number on the leaderboard *first*. Sanity check: bigram top-1 ∈ [60%, 70%]. If outside, the eval harness is wrong, not the model.
2. **A.5 (trigram)** — second number on the leaderboard.
3. **A.3 + A.4 anomaly piece** — verify on bigram: F1 should be ≥0.55 (perplexity thresholding works at all).
4. **B.0** — patch `infineon_gpt.py` to save checkpoint, eval the already-trained nano.
5. **C.2** — train Tax LSTM, eval.
6. **B.1** — train mini GPT, eval.
7. **B.2** — IF condition met, launch small in background per CLAUDE.md rules. Otherwise skip and document why.
8. **C.3** — write comparison.md + leaderboard summary.

After each step: `git add -A && git commit -m "<step>: <result>"` on `industrial-experiments`.

**Time budget (M-chip MPS, sequential):**
- Steps 1–3: ~15 min (mostly coding, training is seconds)
- Step 4: ~6 min train (already done — just eval, ~30s)
- Step 5: ~5 min train + 30s eval
- Step 6: ~3 min train + 30s eval
- Step 7 (if run): 45 min background
- Step 8: ~5 min writeup

**Total foreground: ~35 min. Total with small in background: ~80 min wall.**

---

## 6. Risk Register

| Risk | Mitigation |
|---|---|
| Held-out split leaks into training (worker forgets) | `split.py` writes a manifest; `infineon_gpt.py` must `assert` no held-out ID in training set before first step. |
| Synthetic anomalies are "too easy" (one obvious deletion) → inflated F1 | Generate a *second* harder negative set: 2-step swaps within a block. Report both numbers. |
| MPS non-determinism causes flaky leaderboard | Seed everything (`torch.manual_seed`, `random`, `np.random`); use `torch.use_deterministic_algorithms(False)` on MPS (true would crash); accept ±0.5pp jitter. |
| `eval_metrics.py` released on event day disagrees with our implementation | Our metric definitions match `scout_repo.md` verbatim. Day-of, drop in the real `eval_metrics.py` and re-score from saved per-example predictions (worker writes `predictions_*.csv` in submission format alongside JSON dump → 2-min swap). |
| Small overfits and looks bad in demo | That IS the story. Plot train-vs-val divergence — empirical confirmation of the SOTA-research scaling-law claim. |

---

## 7. Deliverable Checklist (for demo)

- [ ] `results/leaderboard.md` with all 6 models scored on 3 tasks
- [ ] `results/comparison.png` — bar chart top-1 accuracy
- [ ] `results/runs/*.json` — full per-run dumps
- [ ] One-slide summary: "Bigram 65% → LSTM 86% → nanoGPT-nano 90% → small (overfits)"
- [ ] All checkpoints in `checkpoints/` so reviewers can re-score
- [ ] `predictions_nextstep.csv`, `predictions_completion.csv`, `predictions_anomaly.csv` for the best model, in the **exact** submission format from `scout_repo.md §4` — ready to drop on event day.
