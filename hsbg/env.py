"""Single-seat reinforcement-learning wrapper around ``GameEngine``.

The engine stays the source of truth; this module only translates:

* ``ObservationEncoder`` turns one player's view of the match into a flat
  ``float32`` vector (hidden information such as other players' shops, hands,
  gold and boards is left out).
* ``ActionLayout`` maps the engine's variable-length ``legal_actions`` list
  onto a fixed set of integer indices and builds a boolean action mask.
* ``BattlegroundsEnv`` gives one *learner* seat a ``reset`` / ``step``
  interface in the style of Gymnasium. Every other seat is played by an
  opponent policy (random bots by default) inside ``step``.

No training happens here. A typical setup looks like::

    env = BattlegroundsEnv.from_profile("data/profiles/initial-15.6.json")
    obs, info = env.reset(seed=0)
    # obs.shape == (env.observation_size,); info["action_mask"].shape == (env.action_size,)
    obs, reward, terminated, truncated, info = env.step(index)

Requires ``numpy`` (``pip install -e .[rl]``).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from hsbg.data.models import (
    CardRecord,
    EffectTarget,
    EffectTrigger,
    IterationProfile,
    KeywordEffect,
    MinionKeyword,
)
from hsbg.data.profiles import load_game_data
from hsbg.game.agents import ActionAgent, RandomLegalAgent
from hsbg.game.engine import GameEngine
from hsbg.game.legal import RecruitAction, RecruitActionKind
from hsbg.game.state import MinionInstance, PlayerState, ShopOffer

# Divisors that keep stat features roughly in 0-1 for early and mid game.
_STAT_SCALE = 10.0
_TURN_SCALE = 20.0
_POLICY_SEED_MASK = 0x5EED5EED  # same policy-RNG derivation as run_headless_match


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Move:
    """An engine action plus the board order a REORDER_BOARD action needs."""

    action: RecruitAction
    board_order: tuple[str, ...] | None = None


class ActionLayout:
    """Fixed integer index space for recruit actions.

    With the default rules (7 shop slots, hand limit 10, board limit 7) the
    layout has 112 indices, in this order:

    ==========================  =====================================
    buy shop slot ``s``         ``s``
    play hand slot ``h``        ``h`` (no Battlecry target)
    play ``h`` targeting ``b``  ``h * board_limit + b`` (board slot ``b``)
    sell board slot ``b``       ``b``
    move board slot left        ``b`` (swap with ``b - 1``)
    move board slot right       ``b`` (swap with ``b + 1``)
    reroll / freeze toggle / upgrade / end turn
    ==========================  =====================================

    Slots always refer to current positions in ``player.shop``,
    ``player.hand`` and ``player.board``. The engine's single REORDER_BOARD
    action is exposed as the move-left/right indices.
    """

    def __init__(self, shop_slots: int, hand_slots: int, board_slots: int) -> None:
        self.shop_slots = shop_slots
        self.hand_slots = hand_slots
        self.board_slots = board_slots
        self.buy = 0
        self.play = self.buy + shop_slots
        self.play_targeted = self.play + hand_slots
        self.sell = self.play_targeted + hand_slots * board_slots
        self.move_left = self.sell + board_slots
        self.move_right = self.move_left + board_slots
        self.reroll = self.move_right + board_slots
        self.freeze = self.reroll + 1
        self.upgrade = self.freeze + 1
        self.end_turn = self.upgrade + 1
        self.size = self.end_turn + 1

    @classmethod
    def from_rules(cls, rules) -> ActionLayout:
        return cls(max(rules.shop_slots_by_tier), rules.hand_limit, rules.board_limit)

    def legal_moves(
        self, player: PlayerState, legal_actions: Sequence[RecruitAction]
    ) -> dict[int, Move]:
        """Map every currently legal action to its index."""
        shop = {offer.offer_id: slot for slot, offer in enumerate(player.shop)}
        hand = {minion.instance_id: slot for slot, minion in enumerate(player.hand)}
        board_ids = [minion.instance_id for minion in player.board]
        board = {instance_id: slot for slot, instance_id in enumerate(board_ids)}
        moves: dict[int, Move] = {}
        for action in legal_actions:
            kind = action.kind
            if kind is RecruitActionKind.BUY:
                moves[self.buy + shop[action.target_id]] = Move(action)
            elif kind is RecruitActionKind.PLAY:
                slot = hand[action.target_id]
                if action.effect_target_id is None:
                    moves[self.play + slot] = Move(action)
                else:
                    target = board[action.effect_target_id]
                    index = self.play_targeted + slot * self.board_slots + target
                    moves[index] = Move(action)
            elif kind is RecruitActionKind.SELL:
                moves[self.sell + board[action.target_id]] = Move(action)
            elif kind is RecruitActionKind.REORDER_BOARD:
                for slot in range(len(board_ids)):
                    if slot > 0:
                        moves[self.move_left + slot] = Move(
                            action, _swapped(board_ids, slot, slot - 1)
                        )
                    if slot < len(board_ids) - 1:
                        moves[self.move_right + slot] = Move(
                            action, _swapped(board_ids, slot, slot + 1)
                        )
            elif kind is RecruitActionKind.REROLL:
                moves[self.reroll] = Move(action)
            elif kind in (RecruitActionKind.FREEZE, RecruitActionKind.UNFREEZE):
                moves[self.freeze] = Move(action)
            elif kind is RecruitActionKind.UPGRADE_TAVERN:
                moves[self.upgrade] = Move(action)
            elif kind is RecruitActionKind.END_RECRUIT:
                moves[self.end_turn] = Move(action)
            else:  # pragma: no cover - guards future action kinds
                raise ValueError(f"No index for action kind {kind.value!r}.")
        return moves

    def mask(self, moves: Mapping[int, Move]) -> np.ndarray:
        mask = np.zeros(self.size, dtype=bool)
        mask[list(moves)] = True
        return mask

    def describe(self, index: int) -> str:
        """Human-readable name of an index, for logging and debugging."""
        if not 0 <= index < self.size:
            raise ValueError(f"Action index {index} is outside 0..{self.size - 1}.")
        if index < self.play:
            return f"buy shop slot {index - self.buy}"
        if index < self.play_targeted:
            return f"play hand slot {index - self.play}"
        if index < self.sell:
            hand, board = divmod(index - self.play_targeted, self.board_slots)
            return f"play hand slot {hand} targeting board slot {board}"
        if index < self.move_left:
            return f"sell board slot {index - self.sell}"
        if index < self.move_right:
            return f"move board slot {index - self.move_left} left"
        if index < self.reroll:
            return f"move board slot {index - self.move_right} right"
        return ("reroll", "freeze/unfreeze", "upgrade tavern", "end turn")[
            index - self.reroll
        ]


def _swapped(ids: Sequence[str], first: int, second: int) -> tuple[str, ...]:
    order = list(ids)
    order[first], order[second] = order[second], order[first]
    return tuple(order)


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


class ObservationEncoder:
    """Flat feature vector of what one player can see at a recruit decision.

    ``sections`` maps each block name to its slice of the vector:

    * ``self``: health, armor, gold, tavern tier, frozen flag, next upgrade
      cost, hand and board counts, turn, fraction of players still alive.
    * ``shop`` / ``hand`` / ``board``: one fixed-size card block per slot
      (empty slots are all zeros), in the same slot order as ``ActionLayout``.
      Board slots keep attack order.
    * ``opponents``: alive, health, armor and tavern tier of every other seat,
      starting with the next seat and wrapping around.
    * ``last_combat``: fought, won, lost, tied, damage dealt, damage taken.

    Each card block holds: present, tier, attack, health, tribe multi-hot,
    keyword multi-hot (printed plus granted), has Battlecry, has Deathrattle,
    is token, and (optionally) a one-hot of the card's position in the
    profile's enabled cards. ``card_feature_names`` names those entries.
    """

    def __init__(
        self,
        profile: IterationProfile,
        cards: Sequence[CardRecord],
        *,
        include_card_identity: bool = True,
    ) -> None:
        self.rules = profile.rules
        self.cards_by_id = {card.card_id: card for card in cards}
        self.card_index = {card.card_id: index for index, card in enumerate(cards)}
        self.include_card_identity = include_card_identity
        self.tribes = tuple(sorted({card.tribe for card in cards if card.tribe}))
        self.keywords = tuple(MinionKeyword)
        self.card_feature_names = (
            ("present", "tier", "attack", "health")
            + tuple(f"tribe:{tribe}" for tribe in self.tribes)
            + tuple(f"keyword:{keyword.value}" for keyword in self.keywords)
            + ("battlecry", "deathrattle", "token")
            + (
                tuple(f"card:{card.card_id}" for card in cards)
                if include_card_identity
                else ()
            )
        )
        self.card_features = len(self.card_feature_names)

        layout = ActionLayout.from_rules(self.rules)
        widths = (
            ("self", 10),
            ("shop", layout.shop_slots * self.card_features),
            ("hand", layout.hand_slots * self.card_features),
            ("board", layout.board_slots * self.card_features),
            ("opponents", (self.rules.player_count - 1) * 4),
            ("last_combat", 6),
        )
        self.sections: dict[str, slice] = {}
        start = 0
        for name, width in widths:
            self.sections[name] = slice(start, start + width)
            start += width
        self.size = start
        self._slots = {
            "shop": layout.shop_slots,
            "hand": layout.hand_slots,
            "board": layout.board_slots,
        }

    def encode(self, engine: GameEngine, player_id: str) -> np.ndarray:
        rules = self.rules
        player = engine.player_state(player_id)
        obs = np.zeros(self.size, dtype=np.float32)

        alive = sum(1 for seat in engine.state.players if seat.health > 0)
        if player.tavern_tier < rules.max_tavern_tier:
            upgrade_cost = rules.tavern_upgrade_costs[player.tavern_tier - 1]
        else:
            upgrade_cost = 0
        obs[self.sections["self"]] = (
            player.health / rules.starting_health,
            player.armor / rules.starting_health,
            player.gold / rules.max_gold,
            player.tavern_tier / rules.max_tavern_tier,
            float(player.shop_frozen),
            upgrade_cost / max(rules.tavern_upgrade_costs, default=1),
            len(player.hand) / rules.hand_limit,
            len(player.board) / rules.board_limit,
            engine.state.turn / _TURN_SCALE,
            alive / rules.player_count,
        )

        for name, entries in (
            ("shop", player.shop),
            ("hand", player.hand),
            ("board", player.board),
        ):
            block = obs[self.sections[name]].reshape(self._slots[name], self.card_features)
            for slot, entry in enumerate(entries[: self._slots[name]]):
                block[slot] = self._card_vector(entry)

        seats = [seat.player_id for seat in engine.state.players]
        me = seats.index(player_id)
        by_id = {seat.player_id: seat for seat in engine.state.players}
        rows = obs[self.sections["opponents"]].reshape(rules.player_count - 1, 4)
        for row, offset in enumerate(range(1, len(seats))):
            seat = by_id[seats[(me + offset) % len(seats)]]
            if seat.health > 0:
                rows[row] = (
                    1.0,
                    seat.health / rules.starting_health,
                    seat.armor / rules.starting_health,
                    seat.tavern_tier / rules.max_tavern_tier,
                )

        obs[self.sections["last_combat"]] = _last_combat(engine, player_id, rules)
        return obs

    def _card_vector(self, entry: ShopOffer | MinionInstance) -> np.ndarray:
        vector = np.zeros(self.card_features, dtype=np.float32)
        token = getattr(entry, "token_spec", None)
        card = self.cards_by_id.get(entry.definition_id)
        keywords = set(entry.granted_keywords)
        tribes: set[str] = set()
        battlecry = deathrattle = False
        if token is not None:
            attack, health, tier = token.attack, token.health, 0
            keywords.update(token.keywords)
        elif card is not None:
            attack, health, tier = card.attack, card.health, card.tier
            tribes = set(self.tribes) if card.card_type == "All" else {card.tribe}
            for effect in card.effects:
                if (
                    effect.trigger is EffectTrigger.PASSIVE
                    and isinstance(effect.operation, KeywordEffect)
                    and effect.operation.target is EffectTarget.SELF
                ):
                    keywords.add(effect.operation.keyword)
                battlecry |= effect.trigger is EffectTrigger.BATTLECRY
                deathrattle |= effect.trigger is EffectTrigger.DEATHRATTLE
        else:
            raise ValueError(f"Unknown card definition {entry.definition_id!r}.")

        vector[0] = 1.0
        vector[1] = tier / self.rules.max_tavern_tier
        vector[2] = (attack + entry.attack_bonus) / _STAT_SCALE
        vector[3] = (health + entry.health_bonus) / _STAT_SCALE
        cursor = 4
        for tribe in self.tribes:
            vector[cursor] = float(tribe in tribes)
            cursor += 1
        for keyword in self.keywords:
            vector[cursor] = float(keyword in keywords)
            cursor += 1
        vector[cursor : cursor + 3] = (battlecry, deathrattle, token is not None)
        cursor += 3
        if self.include_card_identity and card is not None and token is None:
            vector[cursor + self.card_index[card.card_id]] = 1.0
        return vector


def _last_combat(engine: GameEngine, player_id: str, rules) -> tuple[float, ...]:
    for result in reversed(engine.combat_history):
        fighters = {fighter for fighter, _ in result.survivors_by_player}
        if player_id not in fighters:
            continue
        scale = rules.combat_damage_cap or 1
        won = result.winner_id == player_id
        lost = result.loser_id == player_id
        return (
            1.0,
            float(won),
            float(lost),
            float(result.winner_id is None),
            result.damage / scale if won else 0.0,
            result.damage / scale if lost else 0.0,
        )
    return (0.0,) * 6


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


@runtime_checkable
class IndexPolicy(Protocol):
    """Opponent that acts on encoded observations, e.g. a frozen policy copy."""

    def choose_index(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: random.Random
    ) -> int:
        """Return one index where ``action_mask`` is True."""


OpponentPolicy = ActionAgent | IndexPolicy
RewardFn = Callable[[int, int], float]


def placement_reward(place: int, player_count: int) -> float:
    """Linear terminal reward: +1 for 1st, -1 for last (0 when ``player_count`` is 1)."""
    middle = (player_count + 1) / 2
    half_range = (player_count - 1) / 2
    return (middle - place) / half_range if half_range else 0.0


class BattlegroundsEnv:
    """One learner seat against opponent policies, with ``reset`` and ``step``.

    The learner acts first each recruit turn. When it ends its turn (index
    ``action_layout.end_turn``), every other living seat plays its turn, the
    engine resolves combat, and ``step`` returns at the learner's next
    decision. The episode terminates when the learner gets a placement:
    eliminated, or last one standing, or the engine's turn limit is reached.

    ``step`` returns ``(observation, reward, terminated, truncated, info)``.
    Reward is 0 until termination, then ``reward_fn(place, player_count)``.
    ``truncated`` is always False because the engine itself ends long games.
    ``info`` holds ``action_mask``, ``turn`` and ``placement`` (``None`` while
    the game continues).

    ``opponents`` maps seat IDs to policies. A policy can be any
    ``ActionAgent`` (``choose_action(legal_actions, rng)``) or an
    ``IndexPolicy`` (``choose_index(observation, mask, rng)``, which receives
    that seat's own encoded view). Unlisted seats use ``RandomLegalAgent``.
    Each seat, including the learner, is limited to ``max_actions_per_turn``
    actions per turn; after that only "end turn" is allowed.
    """

    def __init__(
        self,
        profile: IterationProfile,
        cards: Sequence[CardRecord],
        *,
        learner_id: str = "player-1",
        opponents: Mapping[str, OpponentPolicy] | None = None,
        reward_fn: RewardFn = placement_reward,
        max_actions_per_turn: int = 64,
        include_card_identity: bool = True,
        record_combat_frames: bool = False,
    ) -> None:
        if max_actions_per_turn < 1:
            raise ValueError("max_actions_per_turn must be positive.")
        self.profile = profile
        self.cards = tuple(cards)
        self.learner_id = learner_id
        self.opponents = dict(opponents or {})
        if learner_id in self.opponents:
            raise ValueError("The learner seat cannot also have an opponent policy.")
        self.reward_fn = reward_fn
        self.max_actions_per_turn = max_actions_per_turn
        self.record_combat_frames = record_combat_frames
        self.encoder = ObservationEncoder(
            profile, self.cards, include_card_identity=include_card_identity
        )
        self.action_layout = ActionLayout.from_rules(profile.rules)
        self._default_opponent = RandomLegalAgent()
        self._engine: GameEngine | None = None
        self._policy_rng = random.Random(0)
        self._moves: dict[int, Move] = {}
        self._learner_actions_this_turn = 0
        self._learner_turn = 0
        self._done = True

    @classmethod
    def from_profile(cls, profile_path: str | Path, **kwargs) -> BattlegroundsEnv:
        profile, cards = load_game_data(profile_path)
        return cls(profile, cards, **kwargs)

    @property
    def observation_size(self) -> int:
        return self.encoder.size

    @property
    def action_size(self) -> int:
        return self.action_layout.size

    @property
    def engine(self) -> GameEngine:
        """The live engine, for inspection, rendering, or ``to_replay()``."""
        if self._engine is None:
            raise RuntimeError("Call reset() before using the environment.")
        return self._engine

    def reset(self, seed: int) -> tuple[np.ndarray, dict]:
        engine = GameEngine(
            self.profile,
            self.cards,
            seed,
            record_combat_frames=self.record_combat_frames,
        )
        seats = {player.player_id for player in engine.state.players}
        if self.learner_id not in seats:
            raise ValueError(f"Learner seat {self.learner_id!r} is not in this match.")
        unknown = set(self.opponents) - seats
        if unknown:
            raise ValueError(f"Opponent policies for unknown seats: {sorted(unknown)}")
        self._engine = engine
        self._policy_rng = random.Random(seed ^ _POLICY_SEED_MASK)
        self._done = False
        self._learner_turn = engine.state.turn
        self._learner_actions_this_turn = 0
        observation = self._refresh()
        return observation, self._info(None)

    def step(self, index: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("The episode is over; call reset().")
        move = self._moves.get(int(index))
        if move is None:
            raise ValueError(
                f"Action index {index} ({self.action_layout.describe(int(index))}) "
                "is not legal now; check info['action_mask']."
            )
        engine = self.engine
        engine.apply_action(self.learner_id, move.action, board_order=move.board_order)
        self._learner_actions_this_turn += 1

        while (
            not engine.finished
            and self._placement() is None
            and self.learner_id not in engine.pending_player_ids
        ):
            self._play_opponent_turns()

        if engine.state.turn != self._learner_turn:
            self._learner_turn = engine.state.turn
            self._learner_actions_this_turn = 0

        place = self._placement()
        observation = self._refresh()
        if place is None:
            return observation, 0.0, False, False, self._info(None)
        self._done = True
        reward = self.reward_fn(place, self.profile.rules.player_count)
        return observation, reward, True, False, self._info(place)

    def action_mask(self) -> np.ndarray:
        """Legal indices for the learner's current decision (all False when done)."""
        return self.action_layout.mask(self._moves)

    def describe_action(self, index: int) -> str:
        return self.action_layout.describe(index)

    def observe(self, player_id: str) -> tuple[np.ndarray, np.ndarray, dict[int, Move]]:
        """Encoded view, mask and index-to-move map for any pending seat."""
        engine = self.engine
        moves = self.action_layout.legal_moves(
            engine.player_state(player_id), engine.legal_actions(player_id)
        )
        return (
            self.encoder.encode(engine, player_id),
            self.action_layout.mask(moves),
            moves,
        )

    # -- internals ---------------------------------------------------------- #

    def _refresh(self) -> np.ndarray:
        engine = self.engine
        if (
            self._placement() is None
            and self.learner_id in engine.pending_player_ids
        ):
            moves = self.action_layout.legal_moves(
                engine.player_state(self.learner_id),
                engine.legal_actions(self.learner_id),
            )
            if self._learner_actions_this_turn >= self.max_actions_per_turn:
                end = self.action_layout.end_turn
                moves = {end: moves[end]}
            self._moves = moves
        else:
            self._moves = {}
        return self.encoder.encode(engine, self.learner_id)

    def _placement(self) -> int | None:
        return dict(self.engine.placements).get(self.learner_id)

    def _info(self, place: int | None) -> dict:
        return {
            "action_mask": self.action_mask(),
            "turn": self.engine.state.turn,
            "placement": place,
        }

    def _play_opponent_turns(self) -> None:
        engine = self.engine
        turn = engine.state.turn
        for player_id in engine.pending_player_ids:
            if player_id == self.learner_id:
                continue
            self._play_seat_turn(player_id, turn)
            if engine.finished or engine.state.turn != turn:
                return

    def _play_seat_turn(self, player_id: str, turn: int) -> None:
        engine = self.engine
        policy = self.opponents.get(player_id, self._default_opponent)
        decisions = 0
        while (
            not engine.finished
            and engine.state.turn == turn
            and player_id in engine.pending_player_ids
        ):
            if decisions >= self.max_actions_per_turn:
                move = Move(RecruitAction(RecruitActionKind.END_RECRUIT))
            else:
                move = self._opponent_move(player_id, policy)
            engine.apply_action(player_id, move.action, board_order=move.board_order)
            decisions += 1

    def _opponent_move(self, player_id: str, policy: OpponentPolicy) -> Move:
        engine = self.engine
        if isinstance(policy, IndexPolicy):
            observation, mask, moves = self.observe(player_id)
            index = int(policy.choose_index(observation, mask, self._policy_rng))
            if index not in moves:
                raise ValueError(
                    f"Opponent {player_id!r} chose illegal index {index} "
                    f"({self.action_layout.describe(index)})."
                )
            return moves[index]
        action = policy.choose_action(engine.legal_actions(player_id), self._policy_rng)
        if action.kind is RecruitActionKind.REORDER_BOARD:
            board = list(engine.player_state(player_id).board)
            order = tuple(
                minion.instance_id
                for minion in self._policy_rng.sample(board, len(board))
            )
            return Move(action, order)
        return Move(action)
