# Learning-to-Play-Hearthstone-Battlegrounds-Through-Self-Play-Reinforcement-Learning

Research prototype for learning recruitment decisions in a simplified eight-player
Battlegrounds game. The current profile contains 29 selected cards. The next-stage
implementation adds a player-view RL adapter, random/greedy/upgrade baselines,
from-scratch PyTorch PPO, and reproducible evaluation. Historical-policy self-play
is planned, not yet implemented. Results apply only to the simplified rules.

## Setup and first experiment

Run from the repository root with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,rl]"
python -m pytest -q
python -m hsbg.train --updates 2 --episodes-per-update 2 --output runs/smoke
python -m hsbg.evaluate --policy ppo --checkpoint runs/smoke/policy.pt --games 16 --seed 100000 --output runs/ppo_eval.json
python -m hsbg.evaluate --policy greedy --games 16 --seed 100000 --output runs/greedy_eval.json
```

On Windows activate with `.venv\Scripts\activate`. For the desktop simulator
alone, install with `python -m pip install -e .`; PyTorch is optional. A short
training run checks the pipeline and does not establish competitive performance.
See [RL_GUIDE.md](RL_GUIDE.md) for the complete experimental protocol.

## Code organization

- `hsbg/data/`: workbook import, selected cards, effects, and profile validation.
- `hsbg/game/`: shared recruit/combat engine, legality, pool, and deterministic replay.
- `hsbg/gui.py`, `hsbg/play.py`: optional desktop client.
- `hsbg/learning.py`: RL environment, observation/action encoding, and rule baselines.
- `hsbg/ppo.py`, `hsbg/train.py`: policy/value network, GAE, PPO, and training CLI.
- `hsbg/evaluate.py`: fixed-policy evaluation and per-game JSON results.
- `tests/`: engine tests plus RL interface and learning-update checks.

## Desktop client

Play as a selected seat against random bots, or use spectator mode to watch an
all-bot match:

```powershell
python -m hsbg.play --seed 7
python -m hsbg.play --spectate --seed 7
```

The client uses the shared match engine and random bots. It labels the shop,
hand, and board with minion stats, explains the available actions, and records
plain-language recruit and combat events. It supports pausing, single-action
stepping, playback speed adjustment, and replaying the most recently completed
match. RL training and evaluation use the separate headless commands above;
the GUI still uses random bots. The current game-rule values remain configurable
prototype assumptions rather than confirmed historical patch rules.

### Basic controls

1. **Buy** minions from the shop; buying spends gold and puts the minion in hand.
2. **Play** minions from hand onto your board. The leftmost board minion attacks
   first.
3. **Sell** a board minion to get the configured gold refund. You can also
   reroll the shop, freeze offers for next turn, or upgrade your Tavern.
4. Press **End Recruit** when ready. Combat starts after all players are ready.
   The event feed explains attacks, shields, deaths, summons, and damage.
