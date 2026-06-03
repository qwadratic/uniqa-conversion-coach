#!/bin/bash
# Full-loop eval: distilled persona + distilled coach, both arms.
# Depends on coach training output.
#
# Submit:
#   sbatch --dependency=afterok:<train_jobid> ~/zero-one/slurm/slurm_eval_full_loop.sh
#
#SBATCH --job-name=full-loop-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=slurm-full-loop-eval-%j.out

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
PERSONA_ADAPTER="${PERSONA_ADAPTER:-$HOME/zero-one/leonardo/out_v6_unified}"
COACH_ADAPTER="${COACH_ADAPTER:-$HOME/zero-one/leonardo/out_coach_lora}"
N="${N:-100}"
OUT="${OUT:-$HOME/zero-one/leonardo/eval_full_loop.json}"

echo "BASE=$BASE"
echo "PERSONA_ADAPTER=$PERSONA_ADAPTER"
echo "COACH_ADAPTER=$COACH_ADAPTER"
echo "N=$N  OUT=$OUT"

# Verify adapters exist
if [ ! -f "$COACH_ADAPTER/adapter_config.json" ]; then
    echo "[ERROR] Coach adapter not found at $COACH_ADAPTER"
    exit 1
fi

$PIXI python3 slurm/eval_full_loop.py \
    --base "$BASE" \
    --persona_adapter "$PERSONA_ADAPTER" \
    --coach_adapter "$COACH_ADAPTER" \
    --n "$N" \
    --batch_size 48 \
    --coach_budget 2 \
    --seed 800 \
    --out "$OUT"

echo "=== EVAL DONE $(date) ==="
echo "Results: $OUT"
cat "$OUT"
