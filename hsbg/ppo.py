"""PPO policy and value updates for a collected rollout."""

from dataclasses import dataclass

import torch

from hsbg.actor import ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.rollout import RolloutBuffer


@dataclass(frozen=True, slots=True)
class PPOConfig:
    epochs: int = 4
    minibatch_size: int = 64
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = 0.02


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    epochs_completed: int


def update_ppo(
    actor: ActorNetwork,
    critic: CriticNetwork,
    buffer: RolloutBuffer,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    config: PPOConfig | None = None,
) -> PPOUpdateMetrics:
    """Perform PPO epochs over one on-policy rollout."""
    config = config or PPOConfig()
    if config.epochs < 1:
        raise ValueError("PPO epochs must be positive.")
    if config.minibatch_size < 1:
        raise ValueError("PPO minibatch size must be positive.")
    if not (0 <= config.clip_epsilon < 1):
        raise ValueError("PPO clip epsilon must be in [0, 1).")
    if len(buffer.actions) < 1:
        raise ValueError("Cannot update PPO with an empty rollout.")

    device = buffer.observations.device
    actor_device = next(actor.parameters()).device
    critic_device = next(critic.parameters()).device
    if actor_device != device or critic_device != device:
        raise ValueError("Actor, critic, and rollout must use the same device.")

    advantages = buffer.advantages
    if config.normalize_advantages:
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

    policy_loss_total = 0.0
    value_loss_total = 0.0
    entropy_total = 0.0
    approximate_kl_total = 0.0
    minibatches = 0
    epochs_completed = 0
    stop_for_kl = False

    for _ in range(config.epochs):
        epochs_completed += 1
        for indices in buffer.minibatch_indices(config.minibatch_size):
            log_probs, entropy = actor.evaluate_actions(
                buffer.observations[indices],
                buffer.action_masks[indices],
                buffer.actions[indices],
            )
            old_log_probs = buffer.old_log_probs[indices]
            ratio = (log_probs - old_log_probs).exp()
            batch_advantages = advantages[indices]
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            entropy_loss = entropy.mean()
            actor_loss = policy_loss - config.entropy_coefficient * entropy_loss

            approximate_kl = (
                (old_log_probs - log_probs).abs().mean()
            )

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), config.max_grad_norm
            )
            actor_optimizer.step()

            value_loss = critic.value_loss(
                buffer.observations[indices],
                buffer.value_targets[indices],
            )
            critic_optimizer.zero_grad()
            (config.value_coefficient * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), config.max_grad_norm
            )
            critic_optimizer.step()

            policy_loss_total += policy_loss.detach().item()
            value_loss_total += value_loss.detach().item()
            entropy_total += entropy_loss.detach().item()
            approximate_kl_total += approximate_kl.detach().item()
            minibatches += 1

            if (
                config.target_kl is not None
                and approximate_kl.detach().item() > config.target_kl
            ):
                stop_for_kl = True
                break
        if stop_for_kl:
            break

    if minibatches == 0:
        raise RuntimeError("PPO update did not process any minibatches.")

    return PPOUpdateMetrics(
        policy_loss=policy_loss_total / minibatches,
        value_loss=value_loss_total / minibatches,
        entropy=entropy_total / minibatches,
        approximate_kl=approximate_kl_total / minibatches,
        epochs_completed=epochs_completed,
    )
