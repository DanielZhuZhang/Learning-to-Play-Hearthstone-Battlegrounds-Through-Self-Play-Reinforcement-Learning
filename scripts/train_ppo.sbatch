#!/bin/bash
# PPO training job for a SLURM cluster (written for Northeastern's Explorer).
#
# One-time setup, on a login node, from the repository root:
#     module load python/3.11            # any Python >= 3.11; check `module avail python`
#     python -m venv .venv
#     .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
#     .venv/bin/python -m pip install -e ".[rl]"
#     mkdir -p logs
#
# Submit from the repository root:
#     sbatch scripts/train_ppo.sbatch                                 # one self-play run
#     RUN_NAME=sp_k5 POOL_SIZE=5 sbatch scripts/train_ppo.sbatch      # override settings
#     sbatch --array=0-3 scripts/train_ppo.sbatch                     # 4 runs, different seeds
#     SELF_PLAY=0 RUN_NAME=vs_random sbatch scripts/train_ppo.sbatch  # train against random bots
#
# Monitor:
#     squeue -u $USER
#     tail -f logs/hsbg-ppo-<jobid>_<task>.out
#     scancel <jobid>
#
# Resubmitting the same command continues the run. If the run's checkpoint
# directory already has ppo_latest.pt, the job resumes from it and trains only
# the iterations still missing to reach TOTAL_ITERATIONS.

#SBATCH --job-name=hsbg-ppo
#SBATCH --partition=short            # check `sinfo` for partitions and time limits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # one rollout worker per core (see WORKERS)
#SBATCH --mem=8G                     # about 0.5 GB per worker
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

set -euo pipefail

# ---- settings (override with VAR=value sbatch ...) -------------------------
RUN_NAME="${RUN_NAME:-cluster_selfplay}"
TOTAL_ITERATIONS="${TOTAL_ITERATIONS:-1000}"
EPISODES_PER_ROLLOUT="${EPISODES_PER_ROLLOUT:-16}"  # keep it >= WORKERS so no worker sits idle
SAVE_EVERY="${SAVE_EVERY:-25}"
SELF_PLAY="${SELF_PLAY:-1}"           # 1 = self-play, 0 = random opponents
POOL_SIZE="${POOL_SIZE:-5}"
RANDOM_OPPONENTS="${RANDOM_OPPONENTS:-2}"
BASE_SEED="${BASE_SEED:-7}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"  # 0 skips the final evaluation
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"   # parallel game-playing processes
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
# ---------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"

# Array jobs get their own seed and checkpoint directory.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_DIR="checkpoints/${RUN_NAME}_s${TASK_ID}"
else
    RUN_DIR="checkpoints/${RUN_NAME}"
fi
# Seeds are spaced far apart so tasks never replay the same games.
SEED=$(( BASE_SEED + TASK_ID * 1000000 ))

module load "${PYTHON_MODULE}" 2>/dev/null || echo "note: module ${PYTHON_MODULE} not loaded; using the venv's python"
PY="${SLURM_SUBMIT_DIR}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "error: ${PY} not found. Run the one-time setup at the top of this script." >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # print each iteration to the log immediately

mkdir -p "${RUN_DIR}"

TRAIN_ARGS=(--checkpoint-dir "${RUN_DIR}"
            --episodes-per-rollout "${EPISODES_PER_ROLLOUT}"
            --save-every "${SAVE_EVERY}"
            --seed "${SEED}"
            --workers "${WORKERS}")
if [[ "${SELF_PLAY}" == "1" ]]; then
    TRAIN_ARGS+=(--self-play --pool-size "${POOL_SIZE}" --random-opponents "${RANDOM_OPPONENTS}")
fi

# Resume if this run already has a checkpoint.
DONE=0
if [[ -f "${RUN_DIR}/ppo_latest.pt" ]]; then
    DONE=$("${PY}" -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu')['iteration']))" "${RUN_DIR}/ppo_latest.pt")
    TRAIN_ARGS+=(--resume "${RUN_DIR}/ppo_latest.pt")
fi
REMAINING=$(( TOTAL_ITERATIONS - DONE ))

echo "job=${SLURM_JOB_ID:-local} task=${TASK_ID} host=$(hostname) start=$(date -Is)"
echo "run_dir=${RUN_DIR} seed=${SEED} workers=${WORKERS} self_play=${SELF_PLAY} pool=${POOL_SIZE} random_opponents=${RANDOM_OPPONENTS}"
echo "iterations: done=${DONE} target=${TOTAL_ITERATIONS} remaining=${REMAINING}"

if (( REMAINING > 0 )); then
    # Round down to a multiple of SAVE_EVERY so the last iteration is always saved.
    if (( REMAINING >= SAVE_EVERY )); then
        REMAINING=$(( REMAINING / SAVE_EVERY * SAVE_EVERY ))
    else
        TRAIN_ARGS+=(--save-every "${REMAINING}")   # later flag wins in argparse
    fi
    "${PY}" rl/train_ppo.py "${TRAIN_ARGS[@]}" --iterations "${REMAINING}"
else
    echo "Target already reached; skipping training."
fi

if (( EVAL_EPISODES > 0 )); then
    echo "evaluating ${RUN_DIR}/ppo_latest.pt on ${EVAL_EPISODES} games vs random bots (random play = placement 4.5)"
    "${PY}" rl/evaluate.py "${RUN_DIR}/ppo_latest.pt" --episodes "${EVAL_EPISODES}" --seed 900000000
fi

echo "end=$(date -Is)"
