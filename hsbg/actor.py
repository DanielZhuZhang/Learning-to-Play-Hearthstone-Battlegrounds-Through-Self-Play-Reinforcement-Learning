"""Actor (policy) network: observation + legal-action mask -> action probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from hsbg.env import BattlegroundsEnv


@dataclass(frozen=True, slots=True)
class ActorConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)  # neurons per hidden layer
    dropout: float = 0.0  # keep 0 for on-policy PPO (rollout.py checks this)
    output_init_gain: float = 0.01  # small output layer -> near-uniform start


class ActorNetwork(nn.Module):
    """Feed-forward policy: Linear -> LayerNorm -> ReLU per hidden layer, then 112 logits."""

    def __init__(
        self, observation_size: int, action_size: int, config: ActorConfig | None = None
    ) -> None:
        super().__init__()
        self.config = config = config or ActorConfig()
        self.observation_size = observation_size
        self.action_size = action_size
        layers: list[nn.Module] = []
        width = observation_size
        for hidden in config.hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
            width = hidden
        output = nn.Linear(width, action_size)
        nn.init.orthogonal_(output.weight, config.output_init_gain)
        nn.init.zeros_(output.bias)
        self.layers = nn.Sequential(*layers, output)

    @classmethod
    def from_env(cls, env: BattlegroundsEnv, config: ActorConfig | None = None) -> ActorNetwork:
        return cls(env.observation_size, env.action_size, config)

    def forward(
        self, observation: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Logits for one observation ``(D,)`` or a batch ``(B, D)``; illegal actions -> huge negative."""
        if observation.ndim not in (1, 2) or observation.shape[-1] != self.observation_size:
            raise ValueError(f"Expected one observation or a batch of width {self.observation_size}.")
        logits = self.layers(observation.float())
        if action_mask is not None:
            if action_mask.shape != logits.shape:
                raise ValueError(f"Expected an action mask of shape {tuple(logits.shape)}.")
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def distribution(self, observation: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
        return Categorical(logits=self(observation, action_mask))

    def act(
        self, observation: torch.Tensor, action_mask: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick actions; return ``(action, log_prob, entropy)``."""
        dist = self.distribution(observation, action_mask)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self, observation: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(log_prob, entropy)`` of stored actions, for the PPO update."""
        dist = self.distribution(observation, action_mask)
        return dist.log_prob(actions), dist.entropy()
