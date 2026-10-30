"""Single-seat RL adapter and observation-only baselines for the prototype engine.

The adapter deliberately keeps board order fixed. All policies see the same
information and action restrictions; the desktop client still supports reorder.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hsbg.data.models import EffectTrigger, KeywordEffect, MinionKeyword
from hsbg.data.profiles import load_profile
from hsbg.data.selection import apply_profile_overrides, select_enabled_cards
from hsbg.data.workbook import load_sheet
from hsbg.game.engine import GameEngine
from hsbg.game.legal import RecruitAction
from hsbg.game.legal import RecruitActionKind as Kind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = PROJECT_ROOT / "data/profiles/initial-15.6.json"
BASELINES = ("random", "greedy", "upgrade")
CONTRACT_VERSION = "hsbg-rl-v1"


def load_content(path=DEFAULT_PROFILE):
    profile = load_profile(Path(path))
    cards = apply_profile_overrides(
        profile,
        select_enabled_cards(
            profile, load_sheet(PROJECT_ROOT / profile.workbook, profile.sheet)
        ),
    )
    return profile, cards


class ActionCodec:
    """Stable slot indices, independent of transient offer/instance identifiers."""

    GLOBAL = (
        Kind.END_RECRUIT,
        Kind.REROLL,
        Kind.FREEZE,
        Kind.UNFREEZE,
        Kind.UPGRADE_TAVERN,
    )

    def __init__(self, rules):
        self.shop_size = max(rules.shop_slots_by_tier)
        self.hand_size = rules.hand_limit
        self.board_size = rules.board_limit
        self.buy_start = len(self.GLOBAL)
        self.play_start = self.buy_start + self.shop_size
        self.sell_start = self.play_start + self.hand_size
        self.n_actions = self.sell_start + self.board_size

    def available(self, engine, player_id):
        player = engine.player_state(player_id)
        legal = set(engine.legal_actions(player_id))
        choices = {i: RecruitAction(kind) for i, kind in enumerate(self.GLOBAL)}
        for start, kind, ids in (
            (self.buy_start, Kind.BUY, [x.offer_id for x in player.shop]),
            (self.play_start, Kind.PLAY, [x.instance_id for x in player.hand]),
            (self.sell_start, Kind.SELL, [x.instance_id for x in player.board]),
        ):
            choices.update(
                {start + i: RecruitAction(kind, target) for i, target in enumerate(ids)}
            )
        return {i: action for i, action in choices.items() if action in legal}


class ObservationEncoder:
    """Own slots and public health/tier summaries; never opponents' private zones.

    Global features: turn, health, armor, gold, tier, upgrade price, frozen,
    alive fraction, actions remaining. Each slot: present, atk, hp, tier,
    taunt, shield, poisonous, followed by card identity one-hot (last = token).
    All scales and offsets form part of CONTRACT_VERSION.
    """

    GLOBAL_SIZE = 9
    SLOT_STATS = 7

    def __init__(self, profile, cards, codec):
        self.rules = profile.rules
        self.cards = {c.card_id: c for c in cards}
        self.card_indices = {c.card_id: i for i, c in enumerate(cards)}
        self.keywords = {
            b.card_id: {
                e.operation.keyword
                for e in b.effects
                if e.trigger is EffectTrigger.PASSIVE
                and isinstance(e.operation, KeywordEffect)
            }
            for b in profile.card_effects
        }
        self.codec = codec
        self.slot_width = self.SLOT_STATS + len(cards) + 1
        self.n_slots = codec.shop_size + codec.hand_size + codec.board_size
        self.size = (
            self.GLOBAL_SIZE
            + self.n_slots * self.slot_width
            + 4 * (self.rules.player_count - 1)
        )

    def _slot(self, item):
        values = np.zeros(self.slot_width, dtype=np.float32)
        if item is None:
            return values
        token = getattr(item, "token_spec", None)
        card = self.cards.get(item.definition_id)
        if token is None and card is None:
            raise ValueError(f"Unknown card definition: {item.definition_id}")
        base = token if token is not None else card
        keywords = set(item.granted_keywords)
        keywords.update(
            token.keywords if token else self.keywords.get(item.definition_id, ())
        )
        values[: self.SLOT_STATS] = (
            1,
            (base.attack + item.attack_bonus) / 20,
            (base.health + item.health_bonus) / 20,
            (card.tier if card else 1) / self.rules.max_tavern_tier,
            MinionKeyword.TAUNT in keywords,
            MinionKeyword.DIVINE_SHIELD in keywords,
            MinionKeyword.POISONOUS in keywords,
        )
        identity = len(self.cards) if token else self.card_indices[item.definition_id]
        values[self.SLOT_STATS + identity] = 1
        return values

    def encode(self, engine, player_id, action_mask, remaining_fraction):
        player = engine.player_state(player_id)
        rules = self.rules
        upgrade = (
            rules.tavern_upgrade_costs[player.tavern_tier - 1]
            if player.tavern_tier < rules.max_tavern_tier
            else 0
        )
        values = [
            engine.state.turn / rules.max_match_turns,
            player.health / max(1, rules.starting_health),
            player.armor / max(1, rules.starting_health),
            player.gold / max(1, rules.max_gold),
            player.tavern_tier / rules.max_tavern_tier,
            upgrade / max(1, rules.max_gold),
            float(player.shop_frozen),
            sum(p.health > 0 for p in engine.state.players) / rules.player_count,
            remaining_fraction,
        ]
        for items, size in (
            (player.shop, self.codec.shop_size),
            (player.hand, self.codec.hand_size),
            (player.board, self.codec.board_size),
        ):
            for i in range(size):
                values.extend(self._slot(items[i] if i < len(items) else None))
        # Relative seat order avoids making one learner seat special.
        players = engine.state.players
        seat = next(i for i, p in enumerate(players) if p.player_id == player_id)
        for offset in range(1, len(players)):
            opponent = players[(seat + offset) % len(players)]
            values.extend(
                (
                    opponent.health / max(1, rules.starting_health),
                    opponent.armor / max(1, rules.starting_health),
                    opponent.tavern_tier / rules.max_tavern_tier,
                    float(opponent.health > 0),
                )
            )
        return {
            "features": np.asarray(values, dtype=np.float32),
            "action_mask": action_mask.copy(),
        }

    def zones(self, observation):
        slots = observation["features"][
            self.GLOBAL_SIZE : self.GLOBAL_SIZE + self.n_slots * self.slot_width
        ].reshape(self.n_slots, self.slot_width)
        shop_end = self.codec.shop_size
        hand_end = shop_end + self.codec.hand_size
        return slots[:shop_end], slots[shop_end:hand_end], slots[hand_end:]


def baseline_action(observation, encoder, rng, strategy="greedy"):
    """Hand-written policy using only the exact observation available to PPO."""
    if strategy not in BASELINES:
        raise ValueError(f"Unknown baseline: {strategy}")
    legal = np.flatnonzero(observation["action_mask"]).tolist()
    if not legal:
        raise ValueError("No legal actions in a terminal observation")
    if strategy == "random":
        return rng.choice(legal)
    codec = encoder.codec
    shop, hand, board = encoder.zones(observation)

    def score(slot):
        return float(20 * (slot[1] + slot[2]) + 3 * sum(slot[4:7]))

    plays = [a for a in legal if codec.play_start <= a < codec.sell_start]
    if plays:
        return max(plays, key=lambda a: score(hand[a - codec.play_start]))
    global_f = observation["features"]
    turn = round(float(global_f[0]) * encoder.rules.max_match_turns)
    tier = round(float(global_f[4]) * encoder.rules.max_tavern_tier)
    if strategy == "upgrade" and 4 in legal and turn >= 2 * tier:
        return 4
    buys = [a for a in legal if codec.buy_start <= a < codec.play_start]
    occupied = [slot for slot in board if slot[0]]
    hand_count = int(sum(slot[0] for slot in hand))
    if buys:
        best = max(buys, key=lambda a: score(shop[a - codec.buy_start]))
        if len(occupied) + hand_count < codec.board_size:
            return best
        sells = [a for a in legal if a >= codec.sell_start]
        if len(occupied) == codec.board_size and sells:
            weakest = min(sells, key=lambda a: score(board[a - codec.sell_start]))
            if (
                score(shop[best - codec.buy_start])
                > score(board[weakest - codec.sell_start]) + 2
            ):
                return weakest
    if 4 in legal and len(occupied) >= codec.board_size:
        return 4
    gold = round(float(global_f[3]) * encoder.rules.max_gold)
    if 1 in legal and gold >= encoder.rules.buy_cost + encoder.rules.reroll_cost:
        return 1
    return 0


def placement_metrics(placements, player_id, player_count):
    """Convert engine competition ranks to midranks and fractional top-four credit."""
    rank = placements[player_id]
    tied = Counter(placements.values())[rank]
    midrank = rank + (tied - 1) / 2
    top_four = max(0, min(tied, min(4, player_count) - rank + 1)) / tied
    reward = 1 - 2 * (midrank - 1) / (player_count - 1)
    return midrank, top_four, reward


class BattlegroundsEnv:
    """One controlled seat against fixed bots, with complete-episode rewards.

    Opponents advance in the engine's original seat order until the learner's next
    decision or elimination. A configured match-turn cap is an engine terminal
    rule (health-based placement), not a rollout truncation.
    """

    def __init__(self, profile, cards, *, opponent="mixed", max_actions=64):
        if opponent not in (*BASELINES, "mixed"):
            raise ValueError("opponent must be random, greedy, upgrade, or mixed")
        if (
            isinstance(max_actions, bool)
            or not isinstance(max_actions, int)
            or max_actions < 1
        ):
            raise ValueError("max_actions must be a positive integer")
        self.profile, self.cards = profile, tuple(cards)
        self.opponent, self.max_actions = opponent, max_actions
        self.codec = ActionCodec(profile.rules)
        self.encoder = ObservationEncoder(profile, cards, self.codec)
        self.engine = None
        self.done = True

    def reset(self, *, seed=0, seat=0):
        if (
            isinstance(seat, bool)
            or not isinstance(seat, int)
            or not 0 <= seat < self.profile.rules.player_count
        ):
            raise ValueError("seat is outside the configured lobby")
        self.engine = GameEngine(self.profile, self.cards, seed)
        self.player_id = self.engine.state.players[seat].player_id
        self.counts = {}
        self.rngs = {
            p.player_id: random.Random(seed * 1009 + i * 9176 + 42)
            for i, p in enumerate(self.engine.state.players)
        }
        self.strategies = {}
        for offset in range(1, self.profile.rules.player_count):
            player = self.engine.state.players[
                (seat + offset) % self.profile.rules.player_count
            ]
            self.strategies[player.player_id] = (
                BASELINES[(offset - 1) % len(BASELINES)]
                if self.opponent == "mixed"
                else self.opponent
            )
        self.done = False
        self._advance()
        return self._observe(self.player_id), self._info()

    def _options(self, player_id):
        options = self.codec.available(self.engine, player_id)
        if self.counts.get((self.engine.state.turn, player_id), 0) >= self.max_actions:
            return {0: options[0]}
        return options

    def _observe(self, player_id):
        mask = np.zeros(self.codec.n_actions, dtype=np.bool_)
        if not self.done:
            mask[list(self._options(player_id))] = True
        remaining = max(
            0,
            1
            - self.counts.get((self.engine.state.turn, player_id), 0)
            / self.max_actions,
        )
        return self.encoder.encode(self.engine, player_id, mask, remaining)

    def _apply(self, player_id, action):
        key = (self.engine.state.turn, player_id)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.engine.apply_action(player_id, action)

    def _advance(self):
        while True:
            if self.player_id in dict(self.engine.placements) or self.engine.finished:
                self.done = True
                return
            pending = self.engine.pending_player_ids
            if not pending:
                raise RuntimeError(
                    "Engine has no pending player before match completion"
                )
            player_id = pending[0]
            if player_id == self.player_id:
                return
            obs = self._observe(player_id)
            action = baseline_action(
                obs, self.encoder, self.rngs[player_id], self.strategies[player_id]
            )
            self._apply(player_id, self._options(player_id)[action])

    def step(self, action):
        if self.engine is None or self.done:
            raise RuntimeError("Call reset before stepping a new episode")
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError("action must be an integer index")
        options = self._options(self.player_id)
        if int(action) not in options:
            raise ValueError("Illegal or masked action; engine state is unchanged")
        self._apply(self.player_id, options[int(action)])
        self._advance()
        info = self._info()
        return (
            self._observe(self.player_id),
            info.get("reward", 0.0),
            self.done,
            False,
            info,
        )

    def _info(self):
        info = {
            "turn": self.engine.state.turn,
            "player_id": self.player_id,
            "match_turn_limit": self.engine.state.turn
            >= self.profile.rules.max_match_turns,
        }
        if self.done:
            rank, credit, reward = placement_metrics(
                dict(self.engine.placements),
                self.player_id,
                self.profile.rules.player_count,
            )
            info.update(placement=rank, top_four_credit=credit, reward=reward)
        return info

    def contract(self):
        """Reject checkpoints from changed rules, content, engine, or encodings."""
        digest = hashlib.sha256(
            json.dumps(
                {
                    "profile": asdict(self.profile),
                    "cards": [asdict(c) for c in self.cards],
                },
                sort_keys=True,
                default=str,
            ).encode()
        )
        for source in sorted((PROJECT_ROOT / "hsbg/game").glob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
        return {
            "version": CONTRACT_VERSION,
            "content_engine_sha256": digest.hexdigest(),
            "observation_size": self.encoder.size,
            "n_actions": self.codec.n_actions,
            "max_actions": self.max_actions,
        }
