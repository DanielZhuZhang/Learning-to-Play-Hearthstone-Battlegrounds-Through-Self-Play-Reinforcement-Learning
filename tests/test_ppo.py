import torch

from hsbg.actor import ActorConfig, ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.ppo import PPOConfig, update_ppo
from hsbg.rollout import RolloutBuffer


def make_buffer() -> RolloutBuffer:
    size = 6
    observations = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    masks = torch.tensor([[True, True],] * size)
    actions = torch.tensor([0, 1, 0, 1, 0, 1])
    old_log_probs = torch.tensor([-0.69] * size)
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    advantages = torch.tensor([-1.0, -0.5, 0.0, 0.2, 0.7, 1.1])
    return RolloutBuffer(
        observations=observations,
        next_observations=observations,
        action_masks=masks,
        actions=actions,
        old_log_probs=old_log_probs,
        rewards=torch.zeros(size),
        terminated=torch.tensor([False, False, False, False, False, True]),
        truncated=torch.zeros(size, dtype=torch.bool),
        old_values=values,
        advantages=advantages,
        value_targets=torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )


def test_update_ppo_changes_both_networks():
    torch.manual_seed(3)
    buffer = make_buffer()
    actor = ActorNetwork(4, 2, ActorConfig(hidden_sizes=(8,), dropout=0.0))
    critic = CriticNetwork(4, hidden_size=8)
    actor_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    critic_before = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-3)

    metrics = update_ppo(
        actor,
        critic,
        buffer,
        actor_optimizer,
        critic_optimizer,
        PPOConfig(epochs=2, minibatch_size=3),
    )

    assert metrics.epochs_completed in (1, 2)
    assert metrics.approximate_kl >= 0
    assert metrics.entropy > 0
    assert any(not torch.equal(actor_before[name], value) for name, value in actor.state_dict().items())
    assert any(not torch.equal(critic_before[name], value) for name, value in critic.state_dict().items())


def test_update_ppo_rejects_empty_rollout():
    actor = ActorNetwork(4, 2)
    critic = CriticNetwork(4)
    empty = make_buffer()
    empty = RolloutBuffer(
        observations=torch.empty(0, 4),
        next_observations=torch.empty(0, 4),
        action_masks=torch.empty(0, 2, dtype=torch.bool),
        actions=torch.empty(0, dtype=torch.long),
        old_log_probs=torch.empty(0),
        rewards=torch.empty(0),
        terminated=torch.empty(0, dtype=torch.bool),
        truncated=torch.empty(0, dtype=torch.bool),
        old_values=torch.empty(0),
        advantages=torch.empty(0),
        value_targets=torch.empty(0),
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    try:
        update_ppo(
            actor,
            critic,
            empty,
            actor_optimizer,
            critic_optimizer,
        )
    except ValueError as error:
        assert "empty rollout" in str(error)
    else:
        raise AssertionError("Expected an empty-rollout ValueError")
