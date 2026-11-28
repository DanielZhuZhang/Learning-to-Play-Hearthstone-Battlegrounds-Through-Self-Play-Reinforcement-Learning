#!/bin/bash
# Evaluate Run 2 (the s2 branches, iterations 5000 -> 7000). Submit from the repository root:
#     sbatch scripts/eval_run2.sbatch
#     EPISODES=50 sbatch scripts/eval_run2.sbatch      # quicker, noisier
#
# Step 1, learning curves: for each of the 9 branch runs, scores checkpoints 5025, 5525,
#   6025, 6525 and 7000 against 7 copies of the starting point (s2 ppo_5000). Also reports
#   the padding share, i.e. whether the penalty reduced moves and freezes.
# Step 2, tournament: the 9 ppo_7000 checkpoints plus the starting point, every one
#   against 7 copies of every other one.
#
# Results go to results/run2/ (CSV files) and logs/hsbg-eval-run2-<jobid>.out.
# Expect about 2.5 hours on 16 CPUs.

#SBATCH --job-name=hsbg-eval-run2
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

EPISODES="${EPISODES:-100}"            # tournament games per pairing
CURVE_EPISODES="${CURVE_EPISODES:-50}" # learning-curve games per checkpoint per opponent
START="${START:-checkpoints/cluster_selfplay_s2/ppo_5000.pt}"
VARIANTS=(${VARIANTS:-s2base s2pad0003 s2pad001})
SEEDS=(${SEEDS:-0 1 2})
PYTHON_MODULE="${PYTHON_MODULE:-python/3.13}"
OUT="results/run2"

cd "${SLURM_SUBMIT_DIR}"
module load "${PYTHON_MODULE}" 2>/dev/null || true
PY="${SLURM_SUBMIT_DIR}/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "${OUT}"

# Check everything exists before spending hours.
[[ -f "${START}" ]] || { echo "error: missing ${START}" >&2; exit 1; }
CKPTS=("${START}"); NAMES=(start)
for v in "${VARIANTS[@]}"; do
    for s in "${SEEDS[@]}"; do
        f="checkpoints/${v}_s${s}/ppo_7000.pt"
        [[ -f "${f}" ]] || { echo "error: missing ${f} (has that run finished?)" >&2; exit 1; }
        CKPTS+=("${f}"); NAMES+=("${v#s2}_s${s}")   # e.g. base_s0, pad001_s2
    done
done
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) start=$(date -Is)"

echo; echo "===== Step 1: learning curves vs the starting point (${CURVE_EPISODES} games each) ====="
# One variant at a time, its 3 seeds side by side (5 checkpoints each = 15 processes).
for v in "${VARIANTS[@]}"; do
    for s in "${SEEDS[@]}"; do
        "${PY}" rl/learning_curve.py "checkpoints/${v}_s${s}" --reference "${START}" \
            --every 20 --episodes "${CURVE_EPISODES}" --workers 5 \
            --csv "${OUT}/curve_${v}_s${s}.csv" > "${OUT}/curve_${v}_s${s}.txt" 2>&1 &
    done
    wait
    for s in "${SEEDS[@]}"; do echo "--- ${v}_s${s}"; cat "${OUT}/curve_${v}_s${s}.txt"; done
done

echo; echo "===== Step 2: tournament (${EPISODES} games per pairing) ====="
"${PY}" rl/tournament.py "${CKPTS[@]}" --names "${NAMES[@]}" \
    --episodes "${EPISODES}" --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --csv "${OUT}/tournament_7000.csv" | tee "${OUT}/tournament_7000.txt"

echo "end=$(date -Is)"
