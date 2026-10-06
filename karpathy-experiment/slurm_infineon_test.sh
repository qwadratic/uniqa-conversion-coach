#!/bin/bash
# Leonardo A100 — Infineon GPT basic test
# Smoke test: loads data, trains 2000 steps (nano config), prints 5 train + 5 generated sequences.
# Submit: sbatch zero-one/karpathy-experiment/slurm_infineon_test.sh
#
# Expected runtime: ~5-10 min on 1×A100 (nano) | ~30 min (small)
#SBATCH --job-name=infineon-gpt-test
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=slurm-infineon-test-%j.out

set -euo pipefail
echo "===== Infineon GPT Test ====="
echo "node=$(hostname)  job=$SLURM_JOB_ID  $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

cd "$HOME/zero-one"
export PYTHONPATH="$HOME/zero-one:$HOME/zero-one/src"

# Data lives in the hackathon data directory (already on $SCRATCH or /tmp on compute nodes)
# Adjust DATA_DIR if the data was copied to SCRATCH:
DATA_DIR="${DATA_DIR:-/tmp/zero_one_hack_01/tracks/industrial-infineon/training_data}"
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR=$DATA_DIR not found."
  echo "Copy data to SCRATCH first (from login node):"
  echo "  cp -r /tmp/zero_one_hack_01/tracks/industrial-infineon \$SCRATCH/zero_one/"
  echo "Then resubmit with: DATA_DIR=\$SCRATCH/zero_one/industrial-infineon/training_data sbatch ..."
  exit 1
fi

RUN="$HOME/.pixi/bin/pixi run --manifest-path $HOME/zero-one/pixi.toml"

# nano config: 1.3M params, 2000 iters — quick GPU smoke test (~5 min)
# Change --config to 'small' for 9.6M / 10000 iters / ~30 min
CONFIG="${CONFIG:-nano}"
ITERS="${ITERS:-2000}"

echo ""
echo "Config: $CONFIG  |  iters: $ITERS  |  data: $DATA_DIR"
echo ""

$RUN python karpathy-experiment/infineon_gpt.py \
    --data_dir "$DATA_DIR" \
    --config "$CONFIG" \
    --iters "$ITERS" \
    --device cuda

echo ""
echo "===== DONE $(date) ====="
