#!/bin/bash
# Step 1: Generate persona traces on GPU — BATCHED (NO internet needed).
#
# Submit:  sbatch ~/zero-one/slurm/slurm_persona_traces.sh
#
#SBATCH --job-name=persona-traces
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm-persona-traces-%j.out

set -euo pipefail
echo "node=$(hostname) gpus=${SLURM_GPUS_PER_TASK:-?} job=$SLURM_JOB_ID  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HOME/zero-one"
export PYTHONPATH="$HOME/zero-one:$HOME/zero-one/sim_loop"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PIXI="$HOME/.pixi/bin/pixi run --manifest-path $HOME/zero-one/pixi.toml"

BASE="${BASE:-$HOME/models/minicpm5-1b}"
ADAPTER="${ADAPTER:-$HOME/zero-one/leonardo/out_v6_unified}"
N="${N:-80}"
OUT="${OUT:-$HOME/zero-one/leonardo/coach_data}"

echo "BASE=$BASE  ADAPTER=$ADAPTER  N=$N  OUT=$OUT"

$PIXI python3 slurm/persona_traces.py \
    --base "$BASE" \
    --adapter "$ADAPTER" \
    --n "$N" \
    --batch_size 48 \
    --seed 500 \
    --out "$OUT"

echo "=== TRACES DONE $(date) ==="
ls -la "$OUT/"
