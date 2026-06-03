#!/bin/bash
# Coach distillation datagen: distilled persona (GPU) + teacher coach (OpenRouter).
# Generates coach SFT training data.
#
# Submit:  sbatch ~/zero-one/slurm/slurm_coach_datagen.sh
#
#SBATCH --job-name=coach-datagen
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=slurm-coach-datagen-%j.out

set -euo pipefail
echo "node=$(hostname) gpus=${SLURM_GPUS_PER_TASK:-?} job=$SLURM_JOB_ID  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HOME/zero-one"
export PYTHONPATH="$HOME/zero-one:$HOME/zero-one/sim_loop"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Proxy for OpenRouter access from compute node
export http_proxy=http://ib0proxy.hpc.cineca.it:3128
export https_proxy=http://ib0proxy.hpc.cineca.it:3128
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

# Source .env for API key
source <(grep -v '^#' .env | sed 's/^/export /')

PIXI="$HOME/.pixi/bin/pixi run --manifest-path $HOME/zero-one/pixi.toml"

BASE="${BASE:-$HOME/models/minicpm5-1b}"
ADAPTER="${ADAPTER:-$HOME/zero-one/leonardo/out_v6_unified}"
N="${N:-80}"
OUT="${OUT:-$HOME/zero-one/leonardo/coach_data}"

echo "BASE=$BASE  ADAPTER=$ADAPTER  N=$N  OUT=$OUT"
echo "Testing OpenRouter connectivity..."
curl -s --max-time 10 -o /dev/null -w "OpenRouter HTTP %{http_code}\n" \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    https://openrouter.ai/api/v1/models || echo "WARNING: OpenRouter unreachable — coach will use retries"

$PIXI python3 slurm/coach_datagen.py \
    --base "$BASE" \
    --adapter "$ADAPTER" \
    --n "$N" \
    --batch_size 24 \
    --coach_budget 2 \
    --seed 500 \
    --out "$OUT"

echo "=== DATAGEN DONE $(date) ==="
echo "Coach SFT data in: $OUT/coach_sft/"
ls -la "$OUT/coach_sft/"
wc -l "$OUT/coach_sft/train.jsonl" "$OUT/coach_sft/val.jsonl"
