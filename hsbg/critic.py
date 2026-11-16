"""State-value network for the learner's encoded Battlegrounds observation."""

import torch
from torch import nn

from hsbg.env import BattlegroundsEnv


class CriticNetwork(nn.Module):
    """Predict one expected discounted placement reward per observation."""

    def __init__(self, observation_size: int, hidden_size: int = 128):
        super().__init__()
        self.observation_size = observation_size
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    @classmethod
    def from_env(cls, env: BattlegroundsEnv) -> "CriticNetwork":
        return cls(env.observation_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim not in (1, 2) or observations.shape[-1] != self.observation_size:
            raise ValueError(f"Expected one observation or a batch of width {self.observation_size}.")
        return self.layers(observations.float()).squeeze(-1)

    def value_loss(self, observations: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = self(observations)
        if predictions.shape != targets.shape:
            raise ValueError("Expected one scalar value target per observation.")
        return (predictions - targets.detach()).square().mean()
