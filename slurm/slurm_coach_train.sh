#!/bin/bash
# Coach LoRA fine-tune: 4 GPUs, DDP via torchrun.
# Depends on coach_datagen output.
#
# Submit:  sbatch ~/zero-one/slurm/slurm_coach_train.sh
# Or with dependency:
#   sbatch --dependency=afterok:<datagen_jobid> ~/zero-one/slurm/slurm_coach_train.sh
#
#SBATCH --job-name=coach-train
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --output=slurm-coach-train-%j.out

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
DATA="${DATA:-$HOME/zero-one/leonardo/coach_data/coach_sft}"
OUT="${OUT:-$HOME/zero-one/leonardo/out_coach_lora}"

echo "BASE=$BASE  DATA=$DATA  OUT=$OUT"

# Verify data exists
if [ ! -f "$DATA/train.jsonl" ]; then
    echo "[ERROR] No training data at $DATA/train.jsonl — datagen probably failed"
    exit 1
fi
echo "Data check:"
wc -l "$DATA/train.jsonl" "$DATA/val.jsonl"

# 4-GPU DDP training
$PIXI torchrun --nproc_per_node=4 slurm/train_coach_lora.py \
    --base "$BASE" \
    --data "$DATA" \
    --out "$OUT" \
    --epochs 5 \
    --bsz 4 \
    --grad_accum 2 \
    --max_len 4096

echo "=== TRAIN DONE $(date) ==="
echo "Coach adapter saved to: $OUT"
ls -la "$OUT/"
