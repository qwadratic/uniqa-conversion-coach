#!/bin/bash
# Eval per-step funnel dropoff for per-persona LoRA adapters.
# No coach arm. Computes ε vs UNIQA anchors.
#
# Submit after per-persona training:
#   sbatch --dependency=afterok:<train_jobid> ~/zero-one/slurm/slurm_eval_dropoff.sh
#
# Or standalone:
#   sbatch ~/zero-one/slurm/slurm_eval_dropoff.sh
#
#SBATCH --job-name=dropoff-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=slurm-dropoff-eval-%j.out

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
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/zero-one/leonardo/out_per_persona}"
N="${N:-100}"
PROPORTIONS="${PROPORTIONS:-0.30,0.50,0.20}"
OUT="${OUT:-$HOME/zero-one/leonardo/dropoff_lora_per_persona.json}"

echo "BASE=$BASE"
echo "ADAPTER_DIR=$ADAPTER_DIR"
echo "N=$N per_persona  PROPORTIONS=$PROPORTIONS"

# Verify at least one adapter exists
found=0
for P in judith franz peter; do
    if [ -f "$ADAPTER_DIR/$P/adapter_config.json" ]; then
        echo "  [ok] $P adapter found"
        found=$((found+1))
    else
        echo "  [warn] $P adapter NOT found at $ADAPTER_DIR/$P"
    fi
done
if [ "$found" -eq 0 ]; then
    echo "[ERROR] No adapters found in $ADAPTER_DIR"
    exit 1
fi

$PIXI python3 slurm/eval_dropoff.py \
    --backend lora \
    --base "$BASE" \
    --adapter_dir "$ADAPTER_DIR" \
    --n "$N" \
    --proportions "$PROPORTIONS" \
    --batch_size 32 \
    --max_new_tokens 512 \
    --out "$OUT"

echo "=== EVAL DONE $(date) ==="
echo "Results: $OUT"
cat "$OUT"
