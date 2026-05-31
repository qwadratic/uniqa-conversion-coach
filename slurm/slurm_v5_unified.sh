#!/bin/bash
# Leonardo: Unified persona LoRA SFT — ONE model, all 3 personas, 4× A100.
# Uses persona_v5 data with short persona tags (no giant system prompts).
# v5 includes both static + coached arms.
#
# Submit:  sbatch ~/zero-one/leonardo/slurm_v5_unified.sh
#
#SBATCH --job-name=v5-unified-lora
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --output=slurm-v5-unified-%j.out

set -euo pipefail
echo "node=$(hostname) gpus=${SLURM_GPUS_PER_TASK:-?} job=$SLURM_JOB_ID  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HOME/zero-one"
export PYTHONPATH="$HOME/zero-one:$HOME/zero-one/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PIXI="$HOME/.pixi/bin/pixi run --manifest-path $HOME/zero-one/pixi.toml"

BASE="${BASE:-$HOME/models/minicpm5-1b}"
DATA="${DATA:-$HOME/zero-one/leonardo/data_v5_unified}"
OUTROOT="${OUTROOT:-$HOME/zero-one/leonardo/out_v5_unified}"
echo "BASE=$BASE  DATA=$DATA  OUTROOT=$OUTROOT"

# Verify data
for SPLIT in train val; do
  f="$DATA/${SPLIT}.jsonl"
  if [ ! -f "$f" ]; then
    echo "[error] missing: $f" && exit 1
  fi
  N=$(wc -l < "$f")
  echo "  data check: $f  ($N rows)"
done

echo "=== training unified model (all personas) on 4× A100 ==="
$PIXI torchrun --nproc_per_node=4 \
    leonardo/train_unified_lora.py \
    --base    "$BASE" \
    --data    "$DATA" \
    --out     "$OUTROOT" \
    --epochs  3 \
    --bsz     4 \
    --grad_accum 4 \
    --max_len 4096

echo "=== DONE.  Unified adapter in $OUTROOT  $(date) ==="
