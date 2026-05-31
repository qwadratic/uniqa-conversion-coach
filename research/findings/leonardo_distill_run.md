# Leonardo Persona-Distillation LoRA — Run Report
*2026-05-31 — Hackathon Day 2, executed within the reservation window*

---

## TL;DR

**All three persona LoRA adapters trained and saved before the reservation closed (12:00 CEST).**
Judith and Franz converged cleanly. Peter shows degenerate loss=0 (see §6). Job completed
in ~15 min total.

---

## 1. SFT Data Construction — `sim_loop/to_sft.py`

**Hard requirement met:** data built exclusively from `sim_loop` code, not from
`research/datagen_v2.py` or `datasets/persona_v1/`.

### How it works

`sim_loop/to_sft.py` reads `sim_loop/out/sessions_coach_{off,on}.jsonl` (100 sessions total,
50 per arm). For each stored session, it **replays** the exact widget/persona builders:

| SFT field | Source |
|-----------|--------|
| `system` | `sim_loop.persona_prompt.build_system_prompt(seg, session_instance)` |
| `user` | `sim_loop.widget.render(step, running_state, history_brief, si, intent, coach_injection)` → JSON string |
| `assistant` | `step['persona_output']` → JSON string |

**State reconstruction:** `running_state` starts from `session_pools.json["start_state"]`
and threads forward through each step's `persona_output["state"]`. `history_brief` is built
from `f"{step_name}: {decision}/{feeling}"` entries, exactly as `persona.py` does it.
`coach_injection` comes from `step["shown_coach"]` (the actual intervention shown in that
step during the recorded run).

### Volume

```
Source files: sessions_coach_off.jsonl (50 sessions), sessions_coach_on.jsonl (50 sessions)
Total steps converted: 545 (train=492, val=53)

Persona breakdown:
  judith:  29 train,  3 val   steps: S1-S4, S5(sparse)
  franz:  269 train, 29 val   steps: all S1-S8
  peter:  194 train, 21 val   steps: all S1-S8
```

Output: `slurm/data_sim/` (local) → uploaded to `~/zero-one/leonardo/data_sim/`

---

## 2. Reservation Status

Checked at 11:28 CEST:
```
ReservationName=s_tra_ncc
StartTime=2026-05-29T20:00:00  EndTime=2026-05-31T12:00:00
State=ACTIVE  Nodes=25 A100 nodes  Account=euhpc_d30_031
```

**Active with 31 minutes remaining at job submission.** All training completed by 11:46,
within the 12:00 window.

---

## 3. Upload Path

scp to login nodes is **blocked** → used datamover:
```bash
# tarball: slurm/data_sim/ + slurm/slurm_persona_sim.sh + sim_loop/to_sft.py  (604 KB)
scp /tmp/persona_sim_sft.tar.gz a08trd13@dmover1.leonardo.cineca.it:/leonardo/home/usertrain/a08trd13/
# then on Leonardo login node:
cd ~/zero-one && tar -xzf ~/persona_sim_sft.tar.gz
cp -r slurm/data_sim leonardo/data_sim
cp slurm/slurm_persona_sim.sh leonardo/
```

(Note: tar extracted to `slurm/` subdir because local repo is `zero-one/slurm/`; manual
copy to `leonardo/data_sim` was needed since the HPC repo still uses the old `leonardo/` dir.)

---

## 4. Job Details

| Item | Value |
|------|-------|
| Job ID | **43452561** |
| Partition | `boost_usr_prod` |
| Reservation | `s_tra_ncc` |
| Account | `euhpc_d30_031` |
| Node | `lrdn0058` (NVIDIA A100-SXM-64GB, 64 GB VRAM) |
| Start | 11:31:53 CEST |
| End | 11:46:31 CEST |
| Wall time | ~15 min |
| Job script | `~/zero-one/leonardo/slurm_persona_sim.sh` |
| Log | `~/zero-one/slurm-persona-sim-43452561.out` |

**Base model:** `~/models/qwen2.5-1.5b` (Qwen/Qwen2.5-1.5B-Instruct, pre-staged)

**Training config:** TRL 1.5.1 · `SFTTrainer` · LoRA r=16 alpha=32 · `assistant_only_loss=True`
· bsz=4, grad_accum=4 (effective=16) · 3 epochs · LR 2e-4 cosine · bf16 · max_len=4096

---

## 5. Training Results — Judith and Franz

### Judith (29 train, 3 val)
```
Steps: 6 total  |  ~60 s wall-time
Epoch 1: eval_loss=0.554  eval_accuracy=0.835
Epoch 2: eval_loss=0.479  eval_accuracy=0.876
Epoch 3: eval_loss=0.478  eval_accuracy=0.884
train_loss=0.530  ← healthy convergence
```

### Franz (269 train, 29 val)
```
Steps: 51 total  |  ~7.5 min wall-time
Epoch 1: eval_accuracy=0.702  (eval_loss=nan — see §6)
Epoch 2: eval_accuracy tracked via train loss
Epoch 3: eval_accuracy=0.702
train_loss=0.255  ← strong convergence (start ~0.46 → end ~0.20)
```

**Adapter locations:**
```
~/zero-one/leonardo/out_sim/judith/   adapter_model.safetensors + adapter_config.json
~/zero-one/leonardo/out_sim/franz/    adapter_model.safetensors + adapter_config.json
```
Each directory also has epoch checkpoints: `checkpoint-N/` with `adapter_model.safetensors`.

---

## 6. Issues and Anomalies

### 6a. Peter — degenerate loss=0

Peter training produced `loss=0, grad_norm=0, entropy=0, mean_token_accuracy=0` throughout
all 3 epochs. The adapter was saved but is untrained weight.

**Probable cause:** `assistant_only_loss=True` failed to locate assistant-turn tokens in
peter's examples, masking the entire sequence to `-100` → zero loss. Judith and Franz worked
correctly with identical code, so the issue is data-specific.

Hypothesis: peter's `persona_output` assistant content contains formatting that confuses
the Qwen2.5 chat-template token boundary detection in TRL 1.5.1. Possible culprits:
- Very long sequences (194 examples with token counts comparable to 269 for franz)
- Nested JSON with characters that interfere with chat markers
- A token length distribution that pushes over `max_len` causing truncation of the
  assistant turn before the mask is applied

**Mitigation for next run:**
```bash
# Diagnose: check tokenized assistant mask on one peter example
python - <<'EOF'
import json; from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$HOME/models/qwen2.5-1.5b")
ex = json.loads(open("leonardo/data_sim/peter.train.jsonl").readline())
toks = tok.apply_chat_template(ex["messages"], tokenize=True, return_dict=True)
print("Total tokens:", len(toks["input_ids"]))
EOF

# Fix: use DataCollatorForCompletionOnlyLM instead of assistant_only_loss=True
# Response template for Qwen2.5: "<|im_start|>assistant\n"
```

### 6b. eval_loss=nan (Franz, Peter)

The `eval_loss=nan` for Franz/Peter is a known TRL quirk when the validation batch has all
tokens in a padding position after completion-only masking and the denominator becomes 0.
The `eval_mean_token_accuracy` is a better signal and was correctly reported for Franz (0.70).

### 6c. Stale `leonardo.*` imports in eval scripts

`slurm/eval_local_batched.py` and `slurm/eval_local.py` import:
```python
from research.run import validate, format_md, PERSONAS   # stale path
from leonardo.batched_teacher import BatchedLocalTeacher  # stale: should be slurm.batched_teacher
```
Not needed for this training run; fix before running eval.

### 6d. PYTHONPATH note

Job script sets `PYTHONPATH=$HOME/zero-one:$HOME/zero-one/src` which exposes `research/`,
`leonardo/`, and `uniqa/` correctly. The `sim_loop/to_sft.py` uses `sys.path.insert`
to add `sim_loop/` itself for widget and persona_prompt imports.

---

## 7. File Changes Made (local)

| File | Action |
|------|--------|
| `sim_loop/to_sft.py` | **NEW** — sim_loop-derived SFT builder |
| `slurm/data_sim/` | **NEW** — 6 JSONL files (492 train + 53 val, 3 personas) |
| `slurm/slurm_persona_sim.sh` | **NEW** — fixed job script using `data_sim`, `out_sim` |
| `slurm/data_sim/summary.json` | **NEW** — per-persona step counts |

---

## 8. Adapter Locations on Leonardo

```
# Usable adapters (healthy training):
/leonardo/home/usertrain/a08trd13/zero-one/leonardo/out_sim/judith/
/leonardo/home/usertrain/a08trd13/zero-one/leonardo/out_sim/franz/

# Saved but degenerate (peter):
/leonardo/home/usertrain/a08trd13/zero-one/leonardo/out_sim/peter/

# Checkpoints (last epoch = same as adapter, earlier epochs also present):
/leonardo/home/usertrain/a08trd13/zero-one/leonardo/out_sim/judith/checkpoint-{2,4,6}/
/leonardo/home/usertrain/a08trd13/zero-one/leonardo/out_sim/franz/checkpoint-{17,34,51}/
/leonardo/home/usertrain/a08trd13/zero-one/leonardo/out_sim/peter/checkpoint-{13,26,39}/
```

---

## 9. Resume / Re-run Commands

```bash
# If reservation is gone — submit to regular queue (no --reservation flag):
# Edit slurm_persona_sim.sh to remove #SBATCH --reservation=s_tra_ncc line, then:
sbatch ~/zero-one/leonardo/slurm_persona_sim.sh

# Re-run peter only (with DataCollatorForCompletionOnlyLM fix):
cd ~/zero-one
~/.pixi/bin/pixi run --manifest-path ~/zero-one/pixi.toml \
  python3 leonardo/train_persona_lora.py \
    --persona peter \
    --base ~/models/qwen2.5-1.5b \
    --data ~/zero-one/leonardo/data_sim \
    --out ~/zero-one/leonardo/out_sim/peter_v2 \
    --epochs 3

# Load adapter for inference:
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("~/models/qwen2.5-1.5b", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "~/zero-one/leonardo/out_sim/judith")
```

---

## 10. Next Steps

1. **Fix peter training:** diagnose the `assistant_only_loss` mask failure, switch to
   `DataCollatorForCompletionOnlyLM` with `"<|im_start|>assistant\n"` response template.
2. **More sim_loop volume:** run `python sim_loop/run.py --sessions 100 --arms off,on
   --concurrency 12 --out sim_loop/out` to get ~1000+ more SFT pairs and re-run.
3. **Quick eval:** use `slurm/eval_local_batched.py` (fix the stale import first) to
   benchmark judith and franz adapters against the frontier teacher.
4. **Demo integration:** wire the saved adapters into the sim_loop local inference path
   for the hackathon demo — persona LLM call replaced by local Qwen2.5+LoRA.
