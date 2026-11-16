"""Use a trained actor as an opponent seat in ``BattlegroundsEnv`` (self-play)."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

from hsbg.actor import ActorNetwork


class ActorPolicy:
    """``IndexPolicy`` wrapper around a frozen copy of an actor::

        env.opponents["player-2"] = ActorPolicy(actor)

    The copy means later training of ``actor`` does not change this opponent.
    """

    def __init__(
        self,
        actor: ActorNetwork,
        *,
        deterministic: bool = False,
        frozen_copy: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.actor = copy.deepcopy(actor) if frozen_copy else actor
        self.actor.to(device).eval()
        if frozen_copy:
            self.actor.requires_grad_(False)
        self.deterministic = deterministic
        self.device = torch.device(device)

    @torch.no_grad()
    def choose_index(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: random.Random
    ) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        probs = torch.softmax(self.actor(obs, mask), -1).cpu().numpy()
        if self.deterministic:
            return int(probs.argmax())
        # Sample with the env's RNG so seeded matches stay reproducible.
        legal = np.flatnonzero(action_mask)
        return int(rng.choices(legal.tolist(), weights=probs[legal].tolist())[0])
