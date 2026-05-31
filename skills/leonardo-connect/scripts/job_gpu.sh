#!/bin/bash
# Leonardo SLURM job — hackathon reservation. Set GPUS=1|2|4 below.
# Fair share: mem = 120GB * gpus, cpus = 8 * gpus.
#SBATCH --job-name=zeroone
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1            # ← set 1 / 2 / 4
#SBATCH --mem=120GB                  # ← 120GB * gpus-per-task
#SBATCH --cpus-per-task=8            # ← 8 * gpus-per-task
#SBATCH --time=0:30:00               # up to 24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Compute nodes have NO internet. Uncomment proxy for low-bandwidth fetches only
# (restarts ~every 10 min; always download big files on login nodes instead).
# export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
# export HTTPS_PROXY=$HTTP_PROXY
# export http_proxy=$HTTP_PROXY
# export https_proxy=$HTTP_PROXY

echo "node=$(hostname) gpus=${SLURM_GPUS_PER_TASK:-?} job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Option A: run inside a pixi env ───────────────────────────────────────────
# RUN="$HOME/.pixi/bin/pixi run --manifest-path $HOME/zero-one/pixi.toml"
# $RUN python3 train.py

# ── Option B: run inside a Singularity/Apptainer container ────────────────────
# CONTAINER="singularity exec --nv --bind $SCRATCH:/scratch container.sif"
# $CONTAINER python3 train.py

echo "TODO: replace the placeholder above with the real run command."
