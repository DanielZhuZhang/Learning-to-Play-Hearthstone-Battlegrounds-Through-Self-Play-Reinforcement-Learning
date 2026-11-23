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


class SnapshotPool:
    """The last ``max_size`` frozen copies of the learner, used as opponents.

    ``add`` after each PPO update; ``assign`` before each rollout gives every
    opponent seat a snapshot drawn uniformly from the pool. With
    ``max_size=1`` every opponent is the previous iteration's actor.
    """

    def __init__(self, max_size: int, *, device: torch.device | str = "cpu") -> None:
        if max_size < 1:
            raise ValueError("Snapshot pool size must be positive.")
        self.max_size = max_size
        self.device = torch.device(device)
        self.snapshots: list[tuple[int, ActorPolicy]] = []

    def __len__(self) -> int:
        return len(self.snapshots)

    @property
    def iterations(self) -> list[int]:
        return [iteration for iteration, _ in self.snapshots]

    def add(self, actor: ActorNetwork, iteration: int) -> None:
        self.snapshots.append((iteration, ActorPolicy(actor, device=self.device)))
        del self.snapshots[: -self.max_size]

    def assign(
        self, env, rng: random.Random, *, random_seats: int = 0
    ) -> dict[str, int | None]:
        """Set ``env.opponents``; returns seat -> snapshot iteration (None = random bot)."""
        if not self.snapshots:
            raise ValueError("Add at least one snapshot before assigning opponents.")
        seats = [
            f"player-{index + 1}"
            for index in range(env.profile.rules.player_count)
            if f"player-{index + 1}" != env.learner_id
        ]
        if not 0 <= random_seats <= len(seats):
            raise ValueError(f"random_seats must be in [0, {len(seats)}].")
        rng.shuffle(seats)
        chosen: dict[str, int | None] = {}
        for position, seat in enumerate(seats):
            if position < random_seats:
                env.opponents.pop(seat, None)
                chosen[seat] = None
            else:
                iteration, policy = rng.choice(self.snapshots)
                env.opponents[seat] = policy
                chosen[seat] = iteration
        return chosen
