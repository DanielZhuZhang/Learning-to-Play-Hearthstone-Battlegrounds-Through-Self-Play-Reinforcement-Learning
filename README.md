# Learning-to-Play-Hearthstone-Battlegrounds-Through-Self-Play-Reinforcement-Learning

## Desktop client

Play as a selected seat against random bots, or use spectator mode to watch an
all-bot match:

```powershell
python -m pip install pygame        # once; needed for the default table view
python -m hsbg.play --seed 7
python -m hsbg.play --spectate --seed 7
python -m hsbg.play --spectate --pov player-3   # follow one player only
python -m hsbg.play --ui tk                     # original text client
```

The default Pygame client draws a Battlegrounds-style table: Bob's Tavern,
your warband and hand as cards with attack/health gems, a lobby of heroes, and
an animated combat screen where attacks, damage numbers, Divine Shield pops,
deaths, and Deathrattle summons play out step by step. Click a shop card to
buy, a hand card to play, or a board minion to sell or move it; hover any card
for its text. "View as" (or clicking a hero in the lobby) shows only that
player's tavern, log, and combats. Space pauses, N steps, +/- change speed,
and Esc skips a combat.

The client uses the shared match engine and random bots. It labels the shop,
hand, and board with minion stats, explains the available actions, and records
plain-language recruit and combat events. It supports pausing, single-action
stepping, playback speed adjustment, and replaying the most recently completed
match. Week 1 intentionally does not include reinforcement learning or
performance-evaluation tools; the current game-rule values remain configurable
prototype assumptions rather than confirmed historical patch rules.

### Basic controls

1. **Buy** minions from the shop; buying spends gold and puts the minion in hand.
2. **Play** minions from hand onto your board. The leftmost board minion attacks
   first.
3. **Sell** a board minion to get the configured gold refund. You can also
   reroll the shop, freeze offers for next turn, or upgrade your Tavern.
4. Press **End Recruit** when ready. Combat starts after all players are ready.
   The event feed explains attacks, shields, deaths, summons, and damage.

## RL environment

`hsbg/env.py` wraps the engine for one learning seat (needs `numpy`:
`pip install -e .[rl]`). It has no training code; plug in any algorithm:

```python
from hsbg.env import BattlegroundsEnv

env = BattlegroundsEnv.from_profile("data/profiles/initial-15.6.json")
obs, info = env.reset(seed=0)            # obs: float32 vector, env.observation_size
mask = info["action_mask"]               # bool vector, env.action_size (112)
obs, reward, terminated, truncated, info = env.step(index)  # index where mask is True
```

- **Actions** are fixed indices (buy slot, play hand slot, optionally on a
  board target, sell, move left/right, reroll, freeze, upgrade, end turn).
  `env.describe_action(i)` names one.
- **Observation** is your own economy, shop, hand and board (per-card stats,
  tribes, keywords, card one-hot), the other seats' public health and tier,
  and your last combat result. `env.encoder.sections` gives each block's slice.
- **Step** applies your action; after "end turn" the other seats play and
  combat resolves before `step` returns. Reward is 0 until you finish, then
  +1 (1st) to -1 (8th).
- **Opponents** default to random bots. Pass `opponents={"player-2": policy}`
  with either an `ActionAgent` or an object with
  `choose_index(observation, mask, rng)` (e.g. a frozen copy of your policy
  for self-play). `env.engine` exposes the live match, including `to_replay()`.

### Actor network

`hsbg/actor.py` holds a PyTorch policy (needs `torch`, also in `.[rl]`). A
shared card MLP embeds every shop, hand and board slot; a small transformer
mixes them with a global token (economy, opponents, last combat); per-slot
heads then score buy / play / play-on-target / sell / move, and the global
token scores reroll, freeze, upgrade and end turn. Illegal actions are masked.

```python
from hsbg.actor import ActorNetwork, ActorPolicy

actor = ActorNetwork.from_env(env)                     # ActorConfig() for sizes
action, log_prob, entropy = actor.act(obs_tensor, mask_tensor)
log_prob, entropy = actor.evaluate_actions(obs, mask, actions)  # e.g. PPO update
env.opponents["player-2"] = ActorPolicy(actor)         # frozen self-play opponent
```

## Cards and profiles

All card data lives in one file, `data/cards.json`: 1,092 cards from every
patch sheet of the original `season -1 (beta).xlsx`, keyed by `sheet:row` IDs
such as `15.6:17`. Different patch versions of a card are simply different
cards. Each entry holds everything about the card:

```json
{ "id": "15.6:17", "name": "Nathrezim Overseer", "tier": 2,
  "tribe": "Demon", "type": "Demon", "attack": 2, "health": 4,
  "text": "Battlecry: Give a friendly Demon +2/+2.",
  "implemented": true,
  "effects": [{ "trigger": "battlecry", "operation": "buff",
                "target": "friendly_demon", "attack": 2, "health": 2 }] }
```

- `text` is shown to players only; behaviour comes from `effects`.
- `implemented: false` cards (no `effects` key) keep their stats and text for
  reference but cannot be put in a game. To implement one, describe it with the
  supported triggers (`passive`, `battlecry`, `deathrattle`) and operations
  (`keyword`, `summon`, `buff`, `damage_hero`), set `implemented` to `true`, and
  list its `effects` (`[]` for a vanilla minion).

A profile in `data/profiles/` chooses which cards are in the minion pool and
sets the game rules:

```json
{ "iteration_id": "initial-15.6", "cards_file": "../cards.json",
  "rules": { "player_count": 8, "...": "..." },
  "enabled_card_ids": ["15.6:1", "15.6:3", "..."] }
```

To try a different roster or ruleset, copy the profile, edit it, and pass it
with `--profile`. Every enabled card must exist in the library and be
implemented; the loader reports anything else when the game starts.