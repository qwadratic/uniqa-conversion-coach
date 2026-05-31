#!/bin/bash
# Persona LoRA SFT from sim_loop-derived data (sim arms off+on).
# Data prepared locally by sim_loop/to_sft.py, uploaded to leonardo/data_sim/.
#
# Submit:
#   sbatch ~/zero-one/leonardo/slurm_persona_sim.sh
#
#SBATCH --job-name=persona-sim-lora
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --output=slurm-persona-sim-%j.out

set -euo pipefail
echo "node=$(hostname) gpus=${SLURM_GPUS_PER_TASK:-?} job=$SLURM_JOB_ID  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HOME/zero-one"
export PYTHONPATH="$HOME/zero-one:$HOME/zero-one/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run via pixi (has torch, transformers, peft, trl, datasets, accelerate)
PIXI="$HOME/.pixi/bin/pixi run --manifest-path $HOME/zero-one/pixi.toml"

BASE="${BASE:-$HOME/models/qwen2.5-1.5b}"
# data_sim = sim_loop-derived SFT (built by sim_loop/to_sft.py)
DATA="${DATA:-$HOME/zero-one/leonardo/data_sim}"
OUTROOT="${OUTROOT:-$HOME/zero-one/leonardo/out_sim}"
echo "BASE=$BASE  DATA=$DATA  OUTROOT=$OUTROOT"

# Verify data is present
for P in judith franz peter; do
  for SPLIT in train val; do
    f="$DATA/${P}.${SPLIT}.jsonl"
    if [ ! -f "$f" ]; then
      echo "[error] missing: $f" && exit 1
    fi
    N=$(wc -l < "$f")
    echo "  data check: $f  ($N rows)"
  done
done

for P in judith franz peter; do
  echo "=== training persona=$P  base=$BASE  data=$DATA  out=$OUTROOT/$P ==="
  $PIXI python3 leonardo/train_persona_lora.py \
        --persona "$P" \
        --base    "$BASE" \
        --data    "$DATA" \
        --out     "$OUTROOT/$P" \
        --epochs  3 \
        --bsz     4 \
        --grad_accum 4 \
        --max_len 4096
  echo "=== done persona=$P  $(date) ==="
done

echo "=== all done.  adapters in $OUTROOT  $(date) ==="
echo "Resume command (if interrupted):"
echo "  sbatch --export=ALL,OUTROOT=$OUTROOT/resume $HOME/zero-one/leonardo/slurm_persona_sim.sh"
