# Experiments

## Run 1: `cluster_selfplay`, 1,000 iterations

5 runs (seeds s0–s4), each trained from scratch on Northeastern Explorer.

```
PYTHON_MODULE=python/3.13 sbatch --array=0-4 scripts/train_ppo.sbatch
```

### Parameters

| Parameter | Value |
|---|---|
| Iterations | 1,000 |
| Games per iteration | 16 (16,000 games per run) |
| Hardware | 8 CPUs, 8 GB memory, no GPU, 8 parallel workers |
| Seeds | 7 + k × 1,000,000 for run k |
| Opponents | 5 frozen copies of the agent (picked from its 5 most recent snapshots) + 2 random bots |
| Reward | Only at the end of the game: (4.5 − place) / 3.5, so +1 for 1st and −1 for 8th |
| Action limit | 64 actions per turn |
| Observation / actions | 1,076 inputs / 112 actions (illegal ones masked) |
| Actor network | 1076 → 256 → 256 → 112 |
| Critic network | 1076 → 128 → 128 → 1 |
| Learning rate | 3 × 10⁻⁴ (Adam) |
| γ / GAE λ | 1.0 / 0.95 |
| PPO clip | 0.2 |
| Epochs / minibatch | 4 / 64 |
| Entropy bonus | 0.01 |
| Value-loss weight | 0.5 |
| Gradient clip | 0.5 |
| KL early stop | 0.02 |
| Checkpoints | Every 25 iterations |

### Results

**Last training iteration (s0):**

| Decisions | Mean return | Policy loss | Value loss | Entropy | Approx. KL |
|---|---|---|---|---|---|
| 11,115 | 0.214 | −0.0010 | 0.0019 | 1.81 | 0.0061 |

**Against 7 random bots** (s0, iteration 1000, 200 games, top-choice actions):

| Mean placement | Mean reward | Decisions per game |
|---|---|---|
| 1.00 (won all 200) | 1.00 | 1,008 |

**Actions per game** (s0, iteration 1000, 10 games, top-choice actions; games lasted 13–17 turns):

| Move minion | Freeze | End turn | Buy | Play | Sell | Upgrade | Reroll |
|---|---|---|---|---|---|---|---|
| 906 (89.9%) | 63 (6.2%) | 16 (1.5%) | 11 (1.1%) | 8 (0.8%) | 3 (0.3%) | 1 (0.1%) | 0 |

**Placement against random bots during training** (s0, 50 games per checkpoint, top-choice actions):

| Iteration | 25 | 125 | 275 | 525 | 725 | 1000 |
|---|---|---|---|---|---|---|
| Mean placement | 1.16 | 1.00 | 1.00 | 1.42 | 1.00 | 1.00 |

### What affects these results

- **Random bots are a weak opponent.** The agent places near 1st from iteration 25 on, so these numbers can't show improvement after that.
- **Top choice vs. sampling.** The agent was trained by sampling its actions. On an older checkpoint, always taking the top choice placed 5.17 against copies of itself, compared with 4.38 when sampling. Top-choice results understate the agent against stronger opponents.
- **Padding.** Moves and freezes don't change the game state, and the reward doesn't penalize them, so the agent uses them until it hits the 64-action limit. This makes the decision counts large.
- **Mean return is measured against the self-play opponents**, which get stronger as the agent does, so it isn't comparable to the random-bot placement.
- **Noise.** With 50 games, differences under about 0.65 places are noise. The 1.42 at iteration 525 is within that range.

---

## Run 1, continued to 5,000 iterations

The same 5 runs, resumed from their iteration-1000 checkpoints with identical settings (job `12166667`).

```
PYTHON_MODULE=python/3.13 TOTAL_ITERATIONS=5000 sbatch --array=0-4 scripts/train_ppo.sbatch
```

| Parameter | Value |
|---|---|
| Iterations | 1,000 → 5,000 (80,000 games per run in total) |
| Everything else | Same as Run 1 |

### Results

**Placement against 7 copies of s0's iteration-1000 checkpoint** (`rl/learning_curve.py`, 200 games per checkpoint, everyone sampling):

| Iteration | 25 | 525 | 1025 | 1525 | 2025 | 2525 | 3025 | Latest |
|---|---|---|---|---|---|---|---|---|
| s0 | 7.24 | 6.21 | 3.16 | 2.17 | 1.93 | 1.78 | 1.37 | 1.34 (3850) |
| s3 | 7.26 | 4.73 | 2.81 | 1.74 | 1.89 | 1.18 | 1.28 | 1.29 (3400) |
| s4 | 7.43 | 4.34 | 2.03 | 1.58 | 1.52 | 1.27 | 1.31 | 1.43 (3850) |

**Padding share** (moves and freezes, sampling): 83–91% at every checkpoint after iteration 500, with no downward trend.

**Fairness check:** s0's iteration-1000 checkpoint against 7 copies of itself placed **4.38** over 400 games (4.5 is even; noise ±0.23).

**Tournament at iteration 5000** (`rl/tournament.py`): each checkpoint against 7 copies of each other checkpoint, 100 games per pairing, everyone sampling. The score is the mean placement against the other four.

| Rank | Checkpoint | Mean placement vs others | First place |
|---|---|---|---|
| 1 | s2 | 3.58 | 33% |
| 2 | s4 | 3.73 | 20% |
| 3 | s3 | 3.92 | 16% |
| 4 | s0 | 4.59 | 10% |
| 5 | s1 | 4.81 | 6% |

Full matrix (row against 7 copies of column):

| | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| s0 | 4.11 | 4.10 | 4.69 | 4.94 | 4.62 |
| s1 | 4.59 | 3.97 | 5.01 | 4.64 | 5.01 |
| s2 | 3.33 | 3.20 | 4.27 | 3.48 | 4.33 |
| s3 | 3.81 | 3.65 | 4.04 | 4.02 | 4.20 |
| s4 | 3.02 | 3.80 | 3.97 | 4.13 | 4.16 |

### What affects these results

- **The benchmark was a weak checkpoint.** s3 and s4 already matched s0's iteration-1000 checkpoint at iteration 525, and s0 ranked 4th of 5 in the tournament. The curves show steady improvement, but the absolute numbers flatter every run.
- **The benchmark is used up.** By iteration 3000 every run places about 1.3 against it, so it can't show further progress. Later comparisons use the tournament instead.
- **Seed matters a lot.** With identical settings, the runs end about 1 place apart (s2 3.58 vs s1 4.81). Results should be reported across seeds, not from one run.
- **Seat advantage.** A checkpoint against copies of itself averaged 4.11 in the tournament instead of 4.5, so the learner's seat (`player-1`) appears to gain about 0.4 places. It applies to every row equally, so the ranking holds, but the absolute placements are slightly optimistic.
- **Noise.** 100 games per pairing gives ±0.46 per cell and ±0.23 per overall score. s2 and s4 (0.15 apart) are not clearly different; the gap to s0 and s1 (about 1 place) is.
- **s1 and s2 curves** were not downloaded, so they are missing from the curve table.

---

## Run 2: branches from s2, iterations 5000 → 7000 (running)

The tournament winner (s2 at iteration 5000) is the starting point for three variants, 3 seeds each. The variant without a penalty is the control.

```
for cfg in "s2base 0" "s2pad0003 0.0003" "s2pad001 0.001"; do
  set -- $cfg
  PYTHON_MODULE=python/3.13 RUN_NAME=$1 ACTION_PENALTY=$2 PENALIZED_ACTIONS=padding \
    TOTAL_ITERATIONS=7000 INIT_FROM='checkpoints/cluster_selfplay_s2/ppo_5000.pt' \
    sbatch --array=0-2 scripts/train_ppo.sbatch
done
```

| Run | Job | Penalty per action | Charged actions | Seeds |
|---|---|---|---|---|
| `s2base` | 12202663 | 0 | none | s0–s2 |
| `s2pad0003` | 12202664 | 0.0003 | moves and freeze toggles | s0–s2 |
| `s2pad001` | 12202665 | 0.001 | moves and freeze toggles | s0–s2 |

All other parameters are the same as Run 1.

**Penalty size.** With γ = 1 the penalties add up in full: 600 padding actions per game (the current level) cost −0.18 at 0.0003 and −0.6 at 0.001. One place is worth 2/7 ≈ 0.29.

**To compare:** run `rl/tournament.py` on the nine iteration-7000 checkpoints plus s2's `ppo_5000.pt`. For each variant, compare placement and padding share.

### Results

*Pending.*
