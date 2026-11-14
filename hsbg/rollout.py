"""Collect one on-policy batch from the current actor and critic."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

from hsbg.actor import ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.env import BattlegroundsEnv


def episode_advantages(
    rewards: list[float], values: list[float], *, bootstrap_value: float,
    gamma: float = 1.0, gae_lambda: float = 0.95,
) -> list[float]:
    """GAE in time order for one trajectory; bootstrap is zero at true termination."""
    if not rewards or len(rewards) != len(values):
        raise ValueError("Provide a reward and old value for each decision.")
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must be in [0, 1].")
    next_value = bootstrap_value
    carry = 0.0
    advantages = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        carry = delta + gamma * gae_lambda * carry
        advantages[t] = carry
        next_value = values[t]
    return advantages


@dataclass(frozen=True)
class RolloutBuffer:
    """Aligned, detached tensors for one collection. Each row is one decision."""

    observations: torch.Tensor
    next_observations: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor

    def minibatch_indices(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Shuffle fresh indices each epoch, including the last partial minibatch."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        order = torch.randperm(len(self.actions), device=self.actions.device)
        yield from order.split(batch_size)


@torch.no_grad()
def collect_rollout(
    env: BattlegroundsEnv, actor: ActorNetwork, critic: CriticNetwork,
    seeds: Iterable[int], *, gamma: float = 1.0, gae_lambda: float = 0.95,
) -> RolloutBuffer:
    """Collect consecutive episodes without model updates, one reset per seed."""
    device = next(actor.parameters()).device
    if next(critic.parameters()).device != device:
        raise ValueError("Place the actor and critic on the same device.")
    if actor.training and actor.config.dropout:
        raise ValueError("Use actor.eval() to disable dropout during on-policy collection.")
    observations, next_observations, masks, actions, log_probs = [], [], [], [], []
    rewards, values, advantages, terminations, truncations = [], [], [], [], []
    for seed in seeds:
        start = len(rewards)
        obs, info = env.reset(seed=seed)
        state = torch.tensor(obs, dtype=torch.float32, device=device)
        while True:
            mask = torch.tensor(info["action_mask"], dtype=torch.bool, device=device)
            action, log_prob, _ = actor.act(state, mask)
            value = critic(state).item()
            obs, reward, terminated, truncated, info = env.step(action.item())
            next_state = torch.tensor(obs, dtype=torch.float32, device=device)

            observations.append(state)
            next_observations.append(next_state)
            masks.append(mask)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            terminations.append(terminated)
            truncations.append(truncated)
            state = next_state
            if terminated or truncated:
                break

        bootstrap = 0.0 if terminated else critic(state).item()
        advantages.extend(episode_advantages(
            rewards[start:], values[start:], bootstrap_value=bootstrap,
            gamma=gamma, gae_lambda=gae_lambda,
        ))

    if not actions:
        raise ValueError("Provide at least one episode seed.")
    old_values = torch.tensor(values, dtype=torch.float32, device=device)
    raw_advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    return RolloutBuffer(
        observations=torch.stack(observations),
        next_observations=torch.stack(next_observations),
        action_masks=torch.stack(masks),
        actions=torch.stack(actions),
        old_log_probs=torch.stack(log_probs),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        terminated=torch.tensor(terminations, dtype=torch.bool, device=device),
        truncated=torch.tensor(truncations, dtype=torch.bool, device=device),
        old_values=old_values,
        advantages=raw_advantages,
        value_targets=old_values + raw_advantages,
    )
