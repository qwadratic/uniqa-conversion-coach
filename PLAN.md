# Last Iteration Plan — UNIQA Persona Distillation

> Written after hackathon Day 2 eval runs. Covers: tuning experience summary,
> key architectural shifts, exploration ideas, concrete next steps, Leonardo plan.

---

## 1. Tuning Experience Summary

### What worked

| Thing | Status |
|---|---|
| Teacher (sim_loop LLM) funnel simulation | ✅ Produces believable 8-step sessions |
| K-sampled per-step atomic dataset (v4, 500/step/persona) | ✅ Fixes class-imbalance collapse |
| `train_persona_lora.py` — per-persona LoRA training | ✅ franz + judith adapters trained |
| Widget state machine | ✅ Deterministic, well-tested |
| Per-step dropoff ground truth | ✅ S4≈66.7%, addon≈24%, S6+S7≈78%, conv≈5.6% |

### What failed

| Thing | Failure mode |
|---|---|
| Hard-label SFT on whole sessions (v1) | Always-convert collapse (class imbalance + exposure bias) |
| Single unified LoRA for all 3 personas (v5) | Judith starved at late funnel (5 rows for S7/S8) |
| Coach LoRA | Trained on wrong distribution — 421 observational / 1 shim-eligible pair |
| Unified v5 data | Imbalanced: judith 143, franz 693, peter 520 rows |

### Key numbers from current Leonardo run (job 43930729)

```
off/judith  success=0.34  convert=0.28  abandon=0.66  advisor=0.06
off/franz   running...
```

judith advisor=0.06 (should be ~0.30) → symptom of late-funnel starvation.

---

## 2. Key Architectural Shifts

### 2a. Separate LoRA per persona (not unified)
**Old:** one model, `system: "persona: judith"` (14 chars).
**New:** 3 separate LoRA adapters, one per persona. `data_sim/` (per-persona split of v4) already exists.

Rationale:
- Judith needs her own distribution at S5–S8 — no crosstalk from franz.
- r=16 on 1B with 128 judith rows cannot separate late-funnel behavior.
- Separate adapters: each has the full model capacity for one persona.

### 2b. Simpler prompts
**Old:** ~12K-char user JSON with `cognitive_model`, `output_schema`, `rules` paragraphs baked into every prompt. The model learns to follow rules in the context, not to internalize behavior.

**New target prompt format:**

```
SYSTEM:
You are a health insurance funnel user.
Dials: budget_pressure=0.56 value_orientation=0.89 complexity_overwhelm=0.15 advisor_lean=0.08 patience=0.45

USER:
step: S4_TARIFF_SELECT
state: attention=0.72 satisfaction=0.60 effort_left=0.55 grasp=0.65 effort_vs_reward=0.60
history: S1:continue S2:continue S3:continue
instance: age=38 visit_goal=price_check price_expectation=firm_budget familiarity=first_time
screen: tariff_table start=38.74 optimal=68.14 provisional=true

ASSISTANT:
{"events":[...], "decision":"leave|continue", "feeling":"...", "state":{...}}
```

Benefits:
- Prompt size ~200 chars vs ~12K — faster inference, less padding waste
- Model must internalize behavior, not just pattern-match instructions
- Configurable: swap dials to create arbitrary persona blend

### 2c. Configurable/composable persona environment

Define a `PersonaConfig` as a parameter vector:
```python
@dataclass
class PersonaConfig:
    name: str
    dials: dict          # 10 floats from prompts/personas/*.params.json
    session_sampler: callable  # samples session_instance from a pool
```

Then `eval_dropoff.py` accepts a list of `PersonaConfig` objects → runs N sessions each → produces per-config statistical signature: `{churn_at_step: {S1..S8}, feeling_dist, state_evolution}`.

This enables:
- Interpolating between existing personas: `blend(franz, 0.7, judith, 0.3)`
- Finding which dials drive S4 churn vs S7 churn
- Synthetic persona exploration without LLM teacher calls

---

## 3. New Ideas to Build

### 3a. Non-persona system prompt experiments
The model trained on `persona: X` tags. But the dials and state are richer than a name.
**Experiment:** replace `system: "persona: judith"` with one of:
- `system: "budget_pressure=0.67 advisor_lean=0.82"` (raw dials)
- `system: "segment: service_affine patience=0.25 complexity_overwhelm=0.70"` (semantic dials)
- `system: "customer_id=4291"` (opaque ID, forces implicit learning)

Generate new SFT data where the system prompt is the dials JSON (not the name), train, compare.
Expected result: a model that responds to continuous parameters, enabling persona interpolation.

### 3b. Teacher vs student comparison framework
Run the same session_instance pool through:
1. Teacher (sim_loop LLM via OpenRouter)
2. Student LoRA (per-persona adapter)
3. Unified LoRA (current v6 experiment)

Measure:
- Per-step churn distribution (JS divergence teacher ↔ student)
- Feeling distribution (how often `cant_grasp` vs `too_much_effort` etc.)
- State trajectory (mean attention/satisfaction/effort_left over steps)
- ε = TV(φ_sim, φ_real) vs UNIQA anchors

### 3c. Statistical persona comparison tool
Given two persona configs, compute a "distance" in behavioral space:
```python
JSD(churn_judith, churn_franz)  # per-step JS divergence
```
Visualize as a radar chart: 8 steps × 2 personas × churn rate.
Lets you answer: "how different are judith and franz really at S4 vs S7?"

### 3d. Persona population simulator
Instead of 3 fixed personas, simulate a population drawn from a parameter distribution.
`N(μ_franz, Σ)` where Σ is calibrated from the real UNIQA segmentation data (n=4004).
Then the "funnel dropoff chart" is a population-level emergent property, not a persona average.

---

## 4. Main Eval Target

**Primary metric:** per-step dropout chart for a 30/50/20 persona mix (judith/franz/peter),
no coach, must match UNIQA ground truth:

| Step | UNIQA anchor | Target |
|---|---|---|
| S4 (tariff) | 66.7% leave | ±8pp |
| S5 (addon) | 24% leave | ±8pp |
| S6+S7 (data+health) | 78% combined leave | ±10pp |
| Final conversion | 5.6% | ±2pp |

`eval_dropoff.py` computes ε = mean absolute deviation vs these anchors.

---

## 5. Last Iteration — Concrete Steps

### Step 1: Measure teacher baseline (local, no GPU, ~1h)

```bash
# from zero-one/ with OPENROUTER_API_KEY in .env
python slurm/eval_dropoff.py \
    --backend teacher \
    --n 50 \
    --proportions 0.30,0.50,0.20 \
    --out eval_dropoff_teacher.json
```

Expected: ε ≤ 0.08 (teacher is calibrated). This is the reference.

### Step 2: Prepare per-persona SFT from v4 balanced data (local, 5 min)

```bash
# splits data_v4_unified into per-persona files (500/step balanced)
python slurm/prepare_sft_per_persona.py \
    --src slurm/data_v4_unified \
    --out slurm/data_per_persona
# → slurm/data_per_persona/{judith,franz,peter}.{train,val}.jsonl
```

### Step 3: Train 3 per-persona LoRA adapters on Leonardo

On Leonardo (tmux session `leo`):

```bash
# [inside leo tmux, $HOME/zero-one]

# Ensure v4 data is staged
ls ~/zero-one/slurm/data_per_persona/

# Submit 3 sequential jobs (one A100, ~3h total)
OUTROOT="$HOME/zero-one/leonardo/out_per_persona" \
DATA="$HOME/zero-one/slurm/data_per_persona" \
BASE="$HOME/models/minicpm5-1b" \
  sbatch ~/zero-one/slurm/slurm_per_persona_train.sh

# Watch
squeue --me
tail -f ~/zero-one/slurm-per-persona-*.out
```

### Step 4: Eval per-persona dropoff on Leonardo

```bash
# [inside leo tmux, after Step 3 completes]

pixi run --manifest-path ~/zero-one/pixi.toml \
  python3 ~/zero-one/slurm/eval_dropoff.py \
    --backend lora \
    --base ~/models/minicpm5-1b \
    --adapter_dir ~/zero-one/leonardo/out_per_persona \
    --n 100 \
    --proportions 0.30,0.50,0.20 \
    --out ~/zero-one/leonardo/dropoff_lora_per_persona.json

cat ~/zero-one/leonardo/dropoff_lora_per_persona.json
```

### Step 5: Simple prompt experiment (new datagen + retrain, ~4h)

**Datagen (local, teacher LLM):**
```bash
# Generate SFT with dial-based system prompts
python slurm/simple_prompt_datagen.py \
    --n_per_persona 500 \
    --out slurm/data_simple_prompt
```

**Transfer + train on Leonardo:**
```bash
# [inside leo tmux]
# Upload
# (use datamover, see README)

# Train with simple prompts
OUTROOT="$HOME/zero-one/leonardo/out_simple_prompt" \
DATA="$HOME/zero-one/slurm/data_simple_prompt" \
  sbatch ~/zero-one/slurm/slurm_per_persona_train.sh
```

### Step 6: Compare all variants

Run `eval_dropoff.py` for each:
- `teacher` (reference)
- `lora_unified_v6` (current job, minicpm5-1b)
- `lora_per_persona_v4` (Step 3-4)
- `lora_simple_prompt` (Step 5, if time)

Produce a comparison table:
```
Model               | ε    | S4_churn | S7_churn | conv  | advisor
--------------------|------|----------|----------|-------|--------
teacher             | 0.06 | 0.64     | 0.81     | 0.056 | 0.28
lora_unified_v6     | 0.18 | 0.72     | 0.66     | 0.28  | 0.06  ← current
lora_per_persona_v4 | ?    | ?        | ?        | ?     | ?
```

---

## 6. Leonardo Utilization Plan

### Active jobs
- `43930729` (full-loop-eval) — finishes in ~40 min. Read results with:
  ```bash
  # [in leo tmux]
  cat ~/zero-one/leonardo/eval_full_loop.json
  ```

### Next submission sequence

```bash
# [in leo tmux]

# 0. Stage per-persona data (if not already there)
ls ~/zero-one/slurm/data_per_persona/ || echo "need to upload"

# 1. Submit per-persona training
sbatch ~/zero-one/slurm/slurm_per_persona_train.sh
# Expected: ~3h, 3x sequential (judith ~50m, franz ~70m, peter ~60m)

# 2. After training, run dropoff eval (submit with dependency)
JOB1=$(sbatch --parsable ~/zero-one/slurm/slurm_per_persona_train.sh)
sbatch --dependency=afterok:$JOB1 ~/zero-one/slurm/slurm_eval_dropoff.sh

# 3. Poll
squeue --me
watch -n 30 'squeue --me; echo "---"; tail -3 ~/zero-one/slurm-*.out'
```

### GPU budget estimate

| Job | GPUs | Time | Rows |
|---|---|---|---|
| per-persona train (3× sequential) | 1 A100 | 3h | 500/step × 7 steps × 3 personas |
| dropoff eval | 1 A100 | 30m | 300 sessions |
| simple prompt train (optional) | 1 A100 | 3h | same as above |

---

## 7. Repository Structure (target after this commit)

```
sim_loop/              teacher simulation engine (LLM, pure stdlib)
  run.py               turn loop: persona ↔ widget ↔ coach
  widget.py            funnel state machine + screen renderer
  persona.py           LLM persona
  coach.py             coach policy + 32-effector library

slurm/                 Leonardo HPC — distillation + eval
  eval_dropoff.py      ★ parametrizable eval: teacher|lora → per-step dropoff + ε
  prepare_sft_per_persona.py  ★ split any dataset by persona
  train_persona_lora.py       per-persona LoRA trainer (1 A100)
  train_unified_lora.py       unified trainer (4 GPU DDP) — archived
  eval_full_loop.py    coach+persona dual eval (current run)
  slurm_eval_dropoff.sh  ★ SLURM job for dropoff eval
  slurm_per_persona_train.sh  ★ SLURM job for per-persona training
  data_sim/            per-persona SFT from v1 (legacy)
  data_v4_unified/     K-sampled balanced dataset (500/step/persona) ← USE THIS
  data_per_persona/    (to generate) per-persona split of v4

prompts/personas/      persona definitions
  *.md                 segment description
  *.params.json        behavioural dials (10 floats) ← the composable unit

PLAN.md               this file
```

★ = new files in this commit
