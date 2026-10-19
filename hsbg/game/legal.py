"""Structured recruit actions available from a player snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from hsbg.game.config import GameRules
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
    """One actionable operation and its target, if it has one."""

    kind: RecruitActionKind
    target_id: str | None = None

    def __post_init__(self) -> None:
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
) -> tuple[RecruitAction, ...]:
    """List currently available recruit operations in stable display order."""
    actions: list[RecruitAction] = []

    if len(player.hand) < rules.hand_limit and player.gold >= rules.buy_cost:
        actions.extend(
            RecruitAction(RecruitActionKind.BUY, offer.offer_id)
            for offer in player.shop
        )

    if len(player.board) < rules.board_limit:
        actions.extend(
            RecruitAction(RecruitActionKind.PLAY, minion.instance_id)
            for minion in player.hand
        )

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
