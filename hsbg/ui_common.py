"""Display helpers shared by the Tk and Pygame clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from hsbg.data.models import CardRecord
from hsbg.game.config import GameRules
from hsbg.game.engine import ActionRecord
from hsbg.game.legal import RecruitActionKind
from hsbg.game.state import MinionInstance, PlayerState

ALL_PLAYERS = "All players"

# A log line plus the player IDs it concerns; ``None`` means visible to everyone.
EventLine = tuple[str, frozenset[str] | None]


def visible_lines(lines: Sequence[EventLine], pov: str | None) -> list[str]:
    """Return log text visible from one player's view, or everything when unset."""
    return [
        text
        for text, players in lines
        if pov is None or players is None or pov in players
    ]


def card_name(card_id: str, card_by_id: Mapping[str, CardRecord]) -> str:
    card = card_by_id.get(card_id)
    return card.name if card is not None else card_id


def minion_name(minion: MinionInstance, card_by_id: Mapping[str, CardRecord]) -> str:
    if minion.token_spec is not None:
        return minion.token_spec.name
    return card_name(minion.definition_id, card_by_id)


def describe_action(
    record: ActionRecord,
    player_before: PlayerState,
    rules: GameRules,
    card_by_id: Mapping[str, CardRecord],
    human_player_id: str | None,
) -> str:
    """Plain-language description of one recruit action, using pre-action state."""
    action = record.action
    actor = "You" if record.player_id == human_player_id else record.player_id
    if action.kind is RecruitActionKind.BUY:
        offer = next(
            offer for offer in player_before.shop if offer.offer_id == action.target_id
        )
        return (
            f"{actor} bought {card_name(offer.definition_id, card_by_id)} "
            f"for {rules.buy_cost} gold."
        )
    if action.kind is RecruitActionKind.PLAY:
        minion = next(
            minion
            for minion in player_before.hand
            if minion.instance_id == action.target_id
        )
        played = f"{actor} played {minion_name(minion, card_by_id)}"
        target = next(
            (
                entry
                for entry in player_before.board
                if entry.instance_id == action.effect_target_id
            ),
            None,
        )
        if target is not None:
            return f"{played}, targeting {minion_name(target, card_by_id)}."
        return f"{played} onto the board."
    if action.kind is RecruitActionKind.SELL:
        minion = next(
            minion
            for minion in player_before.board
            if minion.instance_id == action.target_id
        )
        return (
            f"{actor} sold {minion_name(minion, card_by_id)} for "
            f"{rules.sell_refund} gold."
        )
    if action.kind is RecruitActionKind.REROLL:
        return f"{actor} rerolled the shop for {rules.reroll_cost} gold."
    if action.kind is RecruitActionKind.FREEZE:
        return f"{actor} froze the shop; these offers stay next turn."
    if action.kind is RecruitActionKind.UNFREEZE:
        return f"{actor} unfroze the shop."
    if action.kind is RecruitActionKind.UPGRADE_TAVERN:
        return f"{actor} upgraded to Tavern Tier {player_before.tavern_tier + 1}."
    if action.kind is RecruitActionKind.REORDER_BOARD:
        return f"{actor} changed the board attack order."
    return f"{actor} ended Recruit and is ready for combat."
