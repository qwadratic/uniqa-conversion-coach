#!/bin/bash
# Train 3 per-persona LoRA adapters sequentially on 1x A100.
# Uses balanced K-sampled v4 data (500/step/persona).
#
# Prereqs (run on login node before sbatch):
#   python slurm/prepare_sft_per_persona.py \
#       --src slurm/data_v4_unified --out slurm/data_per_persona
#   (then transfer data_per_persona/ to Leonardo)
#
# Submit:
#   sbatch ~/zero-one/slurm/slurm_per_persona_train.sh
#
# Override via env:
#   BASE=$HOME/models/qwen2.5-1.5b sbatch ...
#   DATA=$HOME/zero-one/slurm/data_per_persona sbatch ...
#
#SBATCH --job-name=per-persona-lora
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=slurm-per-persona-%j.out

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
DATA="${DATA:-$HOME/zero-one/slurm/data_per_persona}"
OUTROOT="${OUTROOT:-$HOME/zero-one/leonardo/out_per_persona}"
EPOCHS="${EPOCHS:-3}"

echo "BASE=$BASE  DATA=$DATA  OUTROOT=$OUTROOT  EPOCHS=$EPOCHS"

# Verify data exists
for P in judith franz peter; do
    if [ ! -f "$DATA/$P.train.jsonl" ]; then
        echo "[ERROR] Missing $DATA/$P.train.jsonl"
        echo "  Run: python slurm/prepare_sft_per_persona.py --src slurm/data_v4_unified --out $DATA"
        exit 1
    fi
done

# Train each persona sequentially
for P in judith franz peter; do
    echo ""
    echo "=== Training persona: $P  $(date) ==="
    OUT="$OUTROOT/$P"

    $PIXI python3 slurm/train_persona_lora.py \
        --persona "$P" \
        --base "$BASE" \
        --data "$DATA" \
        --out "$OUT" \
        --epochs "$EPOCHS" \
        --lr 2e-4 \
        --bsz 4 \
        --grad_accum 4 \
        --max_len 4096

    echo "  $P adapter saved to $OUT"
done

echo ""
echo "=== ALL DONE $(date) ==="
echo "Adapters in $OUTROOT/{judith,franz,peter}"
ls -la "$OUTROOT/"
