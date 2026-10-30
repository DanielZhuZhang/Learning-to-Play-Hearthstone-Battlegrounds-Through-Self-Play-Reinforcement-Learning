"""Small, from-scratch discrete PPO learner (PyTorch, no RL framework).

References: Schulman et al., arXiv:1707.06347 (PPO), arXiv:1506.02438
(GAE); Huang and Ontanon, arXiv:2006.14171 (invalid-action masking).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPOConfig:
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 128

    def __post_init__(self):
        for name in ("hidden_size", "epochs", "minibatch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "clip_ratio", "max_grad_norm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("value_coef", "entropy_coef"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class ActorCritic(nn.Module):
    def __init__(self, observation_size, n_actions, hidden_size=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, features, masks):
        if not torch.all(masks.any(dim=-1)):
            raise ValueError(
                "Policy requires at least one legal action per observation"
            )
        hidden = self.body(features)
        logits = self.actor(hidden).masked_fill(~masks, -torch.inf)
        return Categorical(logits=logits), self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, observation, *, deterministic=False):
        features = torch.as_tensor(observation["features"], dtype=torch.float32)
        masks = torch.as_tensor(observation["action_mask"], dtype=torch.bool)
        distribution, value = self(features, masks)
        action = distribution.probs.argmax() if deterministic else distribution.sample()
        return int(action), float(distribution.log_prob(action)), float(value)


def episode_gae(rewards, values, gamma=1.0, gae_lambda=0.95):
    """GAE for ONE completed learner episode; terminal value is exactly zero.

    No trajectories are concatenated before this calculation, so elimination/reset
    boundaries cannot leak a future episode's value into the previous one.
    """
    if len(rewards) != len(values) or not rewards:
        raise ValueError("rewards and values must have equal nonzero length")
    advantages = np.zeros(len(rewards), dtype=np.float32)
    carry, next_value = 0.0, 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + gamma * next_value - values[i]
        carry = delta + gamma * gae_lambda * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def collect_episodes(env, model, count, first_seed, config, *, seat_offset=0):
    """On-policy collection; one unchanged policy for this complete batch."""
    records = {
        key: []
        for key in (
            "features",
            "masks",
            "actions",
            "log_probs",
            "advantages",
            "returns",
        )
    }
    results = []
    for episode in range(count):
        obs, _ = env.reset(
            seed=first_seed + episode,
            seat=(seat_offset + episode) % env.profile.rules.player_count,
        )
        rewards, values = [], []
        done = False
        while not done:
            action, log_prob, value = model.act(obs)
            records["features"].append(obs["features"])
            records["masks"].append(obs["action_mask"])
            records["actions"].append(action)
            records["log_probs"].append(log_prob)
            obs, reward, done, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError(
                    "Complete-episode collector cannot handle truncation"
                )
            rewards.append(reward)
            values.append(value)
        advantages, returns = episode_gae(
            rewards, values, config.gamma, config.gae_lambda
        )
        records["advantages"].extend(advantages)
        records["returns"].extend(returns)
        results.append(info)
    return {key: np.asarray(value) for key, value in records.items()}, results


def update_policy(model, optimizer, batch, config):
    tensors = {
        k: torch.as_tensor(
            v,
            dtype=(
                torch.bool
                if k == "masks"
                else torch.long
                if k == "actions"
                else torch.float32
            ),
        )
        for k, v in batch.items()
    }
    advantages = tensors["advantages"]
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-8)
    metrics = []
    for _ in range(config.epochs):
        order = torch.randperm(len(advantages))
        for ids in order.split(config.minibatch_size):
            distribution, values = model(
                tensors["features"][ids], tensors["masks"][ids]
            )
            log_probs = distribution.log_prob(tensors["actions"][ids])
            ratio = (log_probs - tensors["log_probs"][ids]).exp()
            raw = ratio * advantages[ids]
            clipped = (
                ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio)
                * advantages[ids]
            )
            policy_loss = -torch.minimum(raw, clipped).mean()
            value_loss = (values - tensors["returns"][ids]).square().mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            metrics.append(
                (
                    float(policy_loss.detach()),
                    float(value_loss.detach()),
                    float(entropy.detach()),
                )
            )
    return dict(
        zip(("policy_loss", "value_loss", "entropy"), np.mean(metrics, axis=0).tolist())
    )


def save_checkpoint(path, model, config, env, metadata):
    """Inference checkpoint; restart training explicitly rather than claiming resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "contract": env.contract(),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path, env):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["contract"] != env.contract():
        raise ValueError(
            "Checkpoint environment contract mismatch (rules/content/engine/encoding/action cap)"
        )
    config = PPOConfig(**saved["config"])
    model = ActorCritic(env.encoder.size, env.codec.n_actions, config.hidden_size)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved["metadata"]
