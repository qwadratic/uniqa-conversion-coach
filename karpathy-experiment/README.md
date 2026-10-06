# Infineon track — sequence models for fab process traces

Side experiment from Zero One Hack (track: Infineon Industrial AI). Small models trained
from scratch on semiconductor process sequences: predict the next step, complete a
sequence, flag sequences that break process rules.

## Data

The dataset provided by the hackathon organizers was used as is:
`tracks/industrial-infineon/training_data` in
[Lumos-Data/zero_one_hack_01](https://github.com/Lumos-Data/zero_one_hack_01)
(`MOSFET_variants.csv`, `IGBT_variants.csv`, `IC_variants.csv` — 3,000 sequences,
198 step types). It is not copied into this repo; the scripts read it from
`--data_dir` (default `/tmp/zero_one_hack_01/tracks/industrial-infineon/training_data`):

```bash
git clone https://github.com/Lumos-Data/zero_one_hack_01 /tmp/zero_one_hack_01
```

`input.txt` is `names.txt` from karpathy/makemore, used by the warm-up scripts
(`microgpt.py`, `names_*.py`).

## Results

Numbers are from the `results_*.json` files in this folder. Evaluation is on our own
held-out split and on synthetic rule violations, not on the organizers' official eval set.

| Task | Model | Result |
|---|---|---|
| Next step, top-1 | bigram baseline | 73.2% |
| Next step, top-1 | nanoGPT, 6 layers, 1.24M params (`infineon_gpt.py`, `eval_harness.py`) | 92.5% |
| Next step, top-1 | 2-layer LSTM, 921k params (`lstm_baseline.py`) | 97.9% |
| Sequence completion, token accuracy | LSTM, cut at 60% / 80% | 83.4% / 76.7% |
| Anomaly detection (300 clean + 300 violated) | bigram, log-prob threshold | F1 0.789, AUC 0.862 |
| Anomaly detection (300 clean + 300 violated) | nanoGPT, log-prob threshold (`anomaly_detection.py`) | F1 0.902, AUC 0.947 |

The GPT and LSTM runs use different train/held-out splits (see the json files), so the
two top-1 numbers are not a strict head-to-head.

Trained on a MacBook (PyTorch, MPS). `slurm_infineon_test.sh` is the job script for
CINECA Leonardo. `EXPERIMENT_PLAN.md` is the plan the runs followed; the code was written
with coding agents under my direction.
