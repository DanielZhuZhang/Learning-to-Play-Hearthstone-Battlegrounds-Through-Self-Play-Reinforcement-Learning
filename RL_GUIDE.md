# First reinforcement-learning experiment

This stage connects the existing simulator to an independently written PPO
learner using PyTorch/autograd, not a packaged RL algorithm or pretrained model.
Developed with OpenAI Codex assistance: team members should review and understand
the code, disclose assistance, and record their actual individual contributions.

## Run

From the repository root, install `python -m pip install -e ".[dev,rl]"`.
Python 3.11+ is required. CPU execution is intentional for this small network.

```bash
python -m pytest -q
python -m hsbg.train --updates 20 --episodes-per-update 8 --seed 0 --output runs/ppo_seed0
python -m hsbg.evaluate --policy ppo --checkpoint runs/ppo_seed0/policy.pt --games 100 --seed 100000 --output runs/ppo_seed0_eval.json
python -m hsbg.evaluate --policy random --games 100 --seed 100000 --output runs/random_eval.json
python -m hsbg.evaluate --policy greedy --games 100 --seed 100000 --output runs/greedy_eval.json
python -m hsbg.evaluate --policy upgrade --games 100 --seed 100000 --output runs/upgrade_eval.json
```

Use `--help` for each CLI. Existing run directories/results are refused.
Checkpoints support inference; optimizer/RNG-state resume is not implemented.
`runs/` and `.pt` are ignored by Git. Logs include library versions; record
`python -m pip freeze` for a reported experiment. Smoke runs prove execution,
not improvement or superiority to baselines.

## Environment contract

`BattlegroundsEnv.reset(seed=..., seat=...)` returns `(observation, info)`;
`step(action)` returns `(observation, reward, terminated, truncated, info)`.
This is a lightweight Gym-style API, not a registered Gymnasium environment.
The shared engine and desktop client have no new numpy/PyTorch imports.

One seat learns against fixed baselines. The default `mixed` pool cycles random,
greedy, upgrade by relative opponent seat. Learner seats rotate between episodes.
All policies use the same encoding, legal mask, and action cap. The runner uses
the engine's seat order, a simplification of simultaneous recruitment.

- `features`: float32 vector with nine own/public scalars, fixed shop/hand/board
  slots, then four public scalars per opponent. Each slot contains presence,
  current attack/health, tier, three keyword indicators, and selected-card ID
  one-hot (plus a token category). Empty slots are zero. Values are scaled,
  not clipped; buffs may exceed the nominal stat scale.
- `action_mask`: boolean vector over stable action IDs. Invalid actions raise
  without changing engine state. Terminal observations have an all-false mask
  and must not be sent to the policy.

Own IDs identify card effects without text embeddings. Opponent shops, hands,
current boards, pool counts and RNG state are excluded. The observation is
partially observable; recurrent memory is not implemented.

Default profile actions:

| IDs | Action |
| --- | --- |
| 0 | End recruit |
| 1 | Reroll |
| 2 / 3 | Freeze / unfreeze |
| 4 | Upgrade tavern |
| 5-11 | Buy shop slot 0-6 |
| 12-21 | Play hand slot 0-9 |
| 22-28 | Sell board slot 0-6 |

Sizes are derived from rules. Board order is fixed in RL (play appends), for
both learner and baselines. At 64 decisions per player per recruit phase, only
End Recruit remains legal to prevent free-action loops. The remaining fraction
of that budget is observed. GUI reordering is unchanged.

Episodes end at the learner's assigned elimination placement or match completion.
The engine's max-match-turn rule ends a game by remaining health; it is a terminal
game rule, not a time-limit bootstrap. Evaluation records its incidence. No
external rollout truncations are introduced.

## Reward and ties

Intermediate rewards are zero; terminal reward is
`1 - 2 * (midrank - 1) / (player_count - 1)` (+1 first, -1 last).
Engine competition ranks are converted to average occupied ranks. Two players
tied across fourth and fifth receive rank 4.5 and 0.5 top-four credit each.
This changes metric interpretation, not engine rules. Buying, upgrading and
surviving another action have no additional reward.

## Learner and baselines

PPO uses a two-layer tanh MLP, masked categorical policy, value head,
complete-episode GAE, clipped policy objective, squared value loss, entropy
bonus, shuffled minibatches and gradient norm clipping. Old log probabilities
and masks are captured during rollout. GAE is calculated separately per episode.
Gamma defaults to 1 to avoid discounting placement by action count. Collection
is serial/on-policy; each update uses fresh episodes only.

`greedy` plays hand cards, buys stat/keyword value, replaces sufficiently weaker
full-board units, then upgrades a full board or refreshes with spare gold.
`upgrade` additionally prioritizes affordable upgrades when turn >= 2 * tier.
These are transparent heuristics, not expert policies. `random` samples uniformly
among the adapter's legal action IDs.

## Evaluation and limitations

Frozen PPO uses argmax actions; training samples actions. Compare policies on
the same opponent pool, profile, action cap, seed range and seat schedule.
Shared seeds do not guarantee identical random outcomes after different actions.
Evaluation refuses seeds overlapping a loaded checkpoint's training range.
Use separate validation seeds for model selection and reserve final test seeds.

JSON includes per-game rank, fractional top-four credit, seat, decisions, turn
and match-turn-limit incidence. Summary standard deviation describes game
variation for one policy, not variation across training runs. Use at least three
independent training seeds for reported comparisons and show each run's results.

Checkpoints bind rules/card data, engine source, encoding version/dimensions and
action cap. Changing these requires retraining. Opponent pools may change for
explicit generalization experiments and are recorded separately.

Existing engine simplifications remain: selected effects, configurable rather
than historical rules, empty ghost boards, and no full-game fidelity guarantee.
This first stage has no learned board order, recurrence, GPU batching, optimizer
resume or historical-policy self-play. Next add frozen checkpoint opponents and
compare with fixed-bot training at equal environment-decision budgets.

## References

- Schulman et al., [PPO](https://arxiv.org/abs/1707.06347).
- Schulman et al., [GAE](https://arxiv.org/abs/1506.02438).
- Huang and Ontanon, [action masking](https://arxiv.org/abs/2006.14171).
- [AutoTavern](https://github.com/MCG-NJU/AutoTavern): conceptual reference for
  player-view observations, masks and separated evaluation. No source code,
  simulator or model weights are incorporated here.
- Bansal et al., [opponent sampling](https://arxiv.org/abs/1710.03748): reference
  for planned self-play, not an implemented feature.
