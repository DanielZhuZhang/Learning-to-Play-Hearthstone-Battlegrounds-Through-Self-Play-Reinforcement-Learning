"""Structured recruit actions available from a player snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from collections.abc import Sequence

from hsbg.data.models import CardEffectBinding, CardRecord
from hsbg.game.config import GameRules
from hsbg.game.effects import battlecry_target_options
from hsbg.game.state import PlayerState


class RecruitActionKind(StrEnum):
    """Recruit operations exposed to agents and user interfaces."""

    BUY = "buy"
    PLAY = "play"
    SELL = "sell"
    REROLL = "reroll"
    FREEZE = "freeze"
    UNFREEZE = "unfreeze"
    UPGRADE_TAVERN = "upgrade_tavern"
    REORDER_BOARD = "reorder_board"
    END_RECRUIT = "end_recruit"


@dataclass(frozen=True, slots=True)
class RecruitAction:
    """One actionable operation and its target, if it has one.

    ``effect_target_id`` is the minion chosen by a played minion's targeted
    Battlecry; only PLAY actions may carry one.
    """

    kind: RecruitActionKind
    target_id: str | None = None
    effect_target_id: str | None = None

    def __post_init__(self) -> None:
        if self.effect_target_id is not None:
            if self.kind is not RecruitActionKind.PLAY:
                raise ValueError("Only play actions may choose a Battlecry target.")
            if (
                not isinstance(self.effect_target_id, str)
                or not self.effect_target_id.strip()
            ):
                raise ValueError("Battlecry target ID must not be empty.")
        if not isinstance(self.kind, RecruitActionKind):
            raise ValueError("Recruit action kind must be a RecruitActionKind value.")
        if self.target_id is not None and (
            not isinstance(self.target_id, str) or not self.target_id.strip()
        ):
            raise ValueError("Recruit action target ID must not be empty.")
        targeted = self.kind in {
            RecruitActionKind.BUY,
            RecruitActionKind.PLAY,
            RecruitActionKind.SELL,
        }
        if targeted != (self.target_id is not None):
            raise ValueError(
                f"Recruit action {self.kind.value!r} "
                f"{'requires' if targeted else 'does not accept'} a target ID."
            )


def get_legal_recruit_actions(
    player: PlayerState,
    rules: GameRules,
    cards: Sequence[CardRecord] = (),
    effect_bindings: Sequence[CardEffectBinding] = (),
) -> tuple[RecruitAction, ...]:
    """List currently available recruit operations in stable display order.

    A minion with a targeted Battlecry yields one PLAY action per valid target
    (or a single untargeted PLAY when nothing is targetable).
    """
    actions: list[RecruitAction] = []

    if len(player.hand) < rules.hand_limit and player.gold >= rules.buy_cost:
        actions.extend(
            RecruitAction(RecruitActionKind.BUY, offer.offer_id)
            for offer in player.shop
        )

    if len(player.board) < rules.board_limit:
        for minion in player.hand:
            options = battlecry_target_options(
                player, minion.instance_id, cards, effect_bindings
            )
            if options:
                actions.extend(
                    RecruitAction(RecruitActionKind.PLAY, minion.instance_id, target)
                    for target in options
                )
            else:
                actions.append(RecruitAction(RecruitActionKind.PLAY, minion.instance_id))

    actions.extend(
        RecruitAction(RecruitActionKind.SELL, minion.instance_id)
        for minion in player.board
    )

    if player.gold >= rules.reroll_cost:
        actions.append(RecruitAction(RecruitActionKind.REROLL))

    if player.shop:
        actions.append(
            RecruitAction(
                RecruitActionKind.UNFREEZE
                if player.shop_frozen
                else RecruitActionKind.FREEZE
            )
        )

    if player.tavern_tier < rules.max_tavern_tier:
        upgrade_cost = rules.tavern_upgrade_costs[player.tavern_tier - 1]
        if player.gold >= upgrade_cost:
            actions.append(RecruitAction(RecruitActionKind.UPGRADE_TAVERN))

    if len(player.board) > 1:
        actions.append(RecruitAction(RecruitActionKind.REORDER_BOARD))

    actions.append(RecruitAction(RecruitActionKind.END_RECRUIT))
    return tuple(actions)
