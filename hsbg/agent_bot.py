"""Let a trained PPO actor control a seat in the desktop client."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from hsbg.actor import ActorNetwork
from hsbg.data.models import CardRecord, IterationProfile
from hsbg.env import BattlegroundsEnv, Move
from hsbg.game.engine import GameEngine


class TrainedBot:
    """Chooses moves for any seat of a live ``GameEngine`` from an actor checkpoint.

    Uses the same observation encoder and action layout as training, so the
    actor sees exactly what it saw in ``BattlegroundsEnv``.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        profile: IterationProfile,
        cards: Sequence[CardRecord],
        *,
        deterministic: bool = True,
    ) -> None:
        layout_env = BattlegroundsEnv(profile, cards)
        self.encoder = layout_env.encoder
        self.action_layout = layout_env.action_layout
        self.max_actions_per_turn = layout_env.max_actions_per_turn
        self.actor = ActorNetwork.from_env(layout_env)
        state = torch.load(Path(checkpoint), map_location="cpu")
        self.actor.load_state_dict(state["actor"])
        self.actor.eval()
        self.deterministic = deterministic
        self.iteration = state.get("iteration")

    @torch.no_grad()
    def choose_move(self, engine: GameEngine, player_id: str) -> Move:
        moves = self.action_layout.legal_moves(
            engine.player_state(player_id), engine.legal_actions(player_id)
        )
        obs = torch.as_tensor(self.encoder.encode(engine, player_id), dtype=torch.float32)
        mask = torch.as_tensor(self.action_layout.mask(moves), dtype=torch.bool)
        action, _, _ = self.actor.act(obs, mask, deterministic=self.deterministic)
        return moves[int(action.item())]
