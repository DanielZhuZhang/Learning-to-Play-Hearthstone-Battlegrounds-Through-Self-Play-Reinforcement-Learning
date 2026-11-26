# Running the Project

How to set up the project, play or watch games, train an agent, evaluate it, and run the tests. Run every command from the repository root.

The training, self-play and checkpoint commands need the `ppo-training-selfplay` branch.

## 1. Setup

You need Python 3.11 or newer.

```powershell
python -m pip install -e ".[gui,rl,dev]"   # pygame, numpy, torch, pytest
git lfs install                            # one time per machine
git lfs pull                               # download the trained checkpoints (*.pt)
```

If `git lfs pull` is skipped, each `checkpoints/**/ppo_latest.pt` is only a small text pointer, and loading it fails.

## 2. Play or watch a game

```powershell
python -m hsbg.play                                   # you are player-1 against 7 random bots
python -m hsbg.play --seed 7 --seat player-3          # choose the seed and your seat
python -m hsbg.play --spectate                        # watch a match with bots in every seat
python -m hsbg.play --spectate --pov player-3         # watch one player's view only
python -m hsbg.play --ui tk                           # older text client (no pygame needed)
```

### Play against a trained agent

```powershell
python -m hsbg.play --agent checkpoints/selfplay1/ppo_latest.pt
python -m hsbg.play --agent checkpoints/selfplay1/ppo_latest.pt --agent-seats player-2,player-3,player-4
python -m hsbg.play --spectate --agent checkpoints/selfplay1/ppo_latest.pt --pov player-1
```

Agent seats show up labeled "(agent)" in the client. `--agent` works only with the Pygame UI.

| Flag | Default | Meaning |
|---|---|---|
| `--seed N` | `0` | Match seed. The same seed and the same moves always give the same game. |
| `--seat player-N` | `player-1` | The seat you control. |
| `--spectate` | off | Watch only. Every seat is a bot. |
| `--pov player-N` | all players | Show only this player's view. |
| `--ui pygame\|tk` | `pygame` | Which client to open. |
| `--profile PATH` | `data/profiles/initial-15.6.json` | Card and rules profile. |
| `--agent PATH` | none | Checkpoint whose actor plays `--agent-seats`. |
| `--agent-seats a,b` | `player-2`, or `player-1` with `--spectate` | Comma-separated seats for the agent. |
| `--agent-greedy` | off | The agent always takes its top choice instead of sampling. |

**Pygame controls**
- **Mouse:** click a shop card to buy it, a hand card to play it, or a board minion to sell or move it.
- **Space:** pause or resume.
- **N:** step forward while paused.
- **+ / −:** change playback speed.
- **Esc:** skip combat.

## 3. Train an agent with PPO

```powershell
# Against random bots
python rl/train_ppo.py --iterations 100 --save-every 10 --checkpoint-dir checkpoints/run2

# Self-play: opponents drawn from the 5 most recent snapshots, 2 seats kept as random bots
python rl/train_ppo.py --self-play --pool-size 5 --random-opponents 2 --iterations 200 --save-every 10 --checkpoint-dir checkpoints/selfplay2

# Continue a run (restores weights, optimizers, iteration count and seed)
python rl/train_ppo.py --resume checkpoints/selfplay2/ppo_latest.pt --self-play --pool-size 5 --iterations 100 --checkpoint-dir checkpoints/selfplay2
```

| Flag | Default | Meaning |
|---|---|---|
| `--iterations N` | `10` | Training iterations. Each one collects games, then runs one PPO update. |
| `--episodes-per-rollout N` | `8` | Games played per iteration. |
| `--seed N` | `7` | First game seed. Each later game uses the next seed. |
| `--actor-lr`, `--critic-lr` | `3e-4` | Adam learning rates. |
| `--gamma`, `--gae-lambda` | `1.0`, `0.95` | Discount factor and GAE λ. |
| `--checkpoint-dir PATH` | `checkpoints/` | Where checkpoints are saved. |
| `--save-every N` | `10` | Save every N iterations. |
| `--resume PATH` | none | Continue from a checkpoint. |
| `--self-play` | off | Opponents are frozen snapshots of the learner instead of random bots. |
| `--pool-size K` | `1` | Number of recent snapshots to keep. 1 means opponents are always the previous iteration. |
| `--random-opponents R` | `0` | With `--self-play`, keep R of the 7 opponent seats as random bots. |
| `--action-penalty X` | `0` | Reward subtracted for each learner action. 0 turns it off. |
| `--penalized-actions all\|padding` | `all` | `all`: every action except end turn. `padding`: only board moves and freeze toggles. |
| `--workers N` | `1` | Processes that play the rollout games in parallel. See "Parallel workers" below. |
| `--device cpu\|cuda` | `cpu` | Use `cuda` to train on a GPU. |

**Output.** Each iteration prints one line with `mean_episode_return`, `policy_loss`, `value_loss`, `entropy` and `approx_kl`. Return runs from −1 (last place) to +1 (first place), so a rising `mean_episode_return` means the agent is improving.

**Checkpoints.** Each save writes `ppo_<iteration>.pt` and `ppo_latest.pt`. Saves happen only on multiples of `--save-every`, so make `--iterations` a multiple of it, or the last iterations won't be saved. Git ignores everything in `checkpoints/` except each run's `ppo_latest.pt`, which is committed through Git LFS.

**Parallel workers.** With `--workers N`, the games in each iteration are split across N processes. Each worker keeps its own copy of the game, gets the current network weights every iteration, plays its share of the games, and sends the decisions back. Training itself still happens in the main process. On a 16-core desktop, 16 games took 5.7s with 1 worker and 1.1s with 8 workers. Starting the workers costs about 5s once per run.

- Set `--episodes-per-rollout` to at least `--workers`. Extra workers with no games to play just sit idle.
- Results are reproducible for a fixed worker count. A different worker count samples different (equally valid) actions.

```powershell
python rl/train_ppo.py --self-play --pool-size 5 --episodes-per-rollout 16 --workers 8 --iterations 200 --checkpoint-dir checkpoints/fast1
```

**Long runs.** On Windows, run training in its own terminal and keep it open. Resume with `--resume` if it stops.

### Action-penalty experiments

The only reward is the final placement, so actions that change nothing (shuffling the board, toggling freeze) cost the agent nothing, and it spends most of its decisions on them. `--action-penalty` makes each action cost a little reward so the agent learns to end its turn when it's done. For scale, one place in the final standings is worth 2/7, about 0.29 reward, and the agent makes about 150 padding moves per game.

Each checkpoint stores the penalty it was trained with under `metadata`. The training log's `mean_episode_return` includes the penalty, so compare runs with `rl/learning_curve.py` or `rl/evaluate.py`, which never apply it.

**Sweep on the cluster, branched from the iteration-1000 baseline.** `INIT_FROM` copies a checkpoint into a new run folder before the first iteration, and `{task}` becomes the array task id. Each branch resumes the baseline's seed, so every penalty value plays the same games as the baseline's own continuation.

```bash
for spec in "pad001 0.001 padding" "pad003 0.003 padding" "all001 0.001 all" "all003 0.003 all"; do
  set -- $spec
  RUN_NAME=$1 ACTION_PENALTY=$2 PENALIZED_ACTIONS=$3 TOTAL_ITERATIONS=2000 PYTHON_MODULE=python/3.13 \
    INIT_FROM='checkpoints/cluster_selfplay_s{task}/ppo_1000.pt' sbatch --array=0-2 scripts/train_ppo.sbatch
done
```

The baseline (penalty 0) is the original `cluster_selfplay` run continuing past iteration 1000.

**Compare at iteration 2000, head to head against the baseline:**

```bash
for run in pad001 pad003 all001 all003; do
  .venv/bin/python rl/learning_curve.py checkpoints/${run}_s0 --every 1000 --episodes 200 --workers 16 \
    --reference checkpoints/cluster_selfplay_s0/ppo_2000.pt --csv curve_${run}_s0.csv
done
```

In the row for iteration 2000, `vs_reference` below 4.5 means that penalty beat the baseline after the same amount of training. `padding` shows how much of the padding it dropped. Run this on a compute node (`srun` or `sbatch --wrap`), not the login node.

## 4. Evaluate a checkpoint

```powershell
python rl/evaluate.py checkpoints/selfplay1/ppo_latest.pt --episodes 100
python rl/evaluate.py checkpoints/run1/ppo_latest.pt --episodes 100 --seed 2000
```

This plays the agent against 7 random bots and always takes its top-choice move. It prints `mean_reward`, `mean_placement` and `mean_decisions`.

- An agent that plays randomly averages placement **4.5** out of 8.
- Anything below 4.5 beats random play.
- Use the same `--seed` and `--episodes` when comparing checkpoints, so every checkpoint plays the same games.

| Flag | Default | Meaning |
|---|---|---|
| `checkpoint` (positional) | `checkpoints/ppo_latest.pt` | Checkpoint to evaluate. |
| `--episodes N` | `20` | Number of games. |
| `--seed N` | `1000` | First game seed. |

## 5. Examples and tests

```powershell
python rl/minimal_example.py        # one game with an untrained actor, printing the first turns
python -m rl.critic_example         # collect 2 games and take a critic training step
python -m pytest -q                 # full test suite
python -m pytest tests/test_ppo.py -q
```

## Quick reference

| Task | Command |
|---|---|
| Play a game | `python -m hsbg.play` |
| Watch bots | `python -m hsbg.play --spectate` |
| Play vs. trained agent | `python -m hsbg.play --agent checkpoints/selfplay1/ppo_latest.pt` |
| Train (random opponents) | `python rl/train_ppo.py --iterations 100 --checkpoint-dir checkpoints/run2` |
| Train (self-play) | `python rl/train_ppo.py --self-play --pool-size 5 --iterations 200 --checkpoint-dir checkpoints/selfplay2` |
| Resume training | `python rl/train_ppo.py --resume <ckpt> --iterations 100 --checkpoint-dir <dir>` |
| Evaluate | `python rl/evaluate.py <ckpt> --episodes 100` |
| Run tests | `python -m pytest -q` |
