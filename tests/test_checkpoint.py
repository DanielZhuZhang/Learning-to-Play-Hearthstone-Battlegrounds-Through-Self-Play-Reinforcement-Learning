import torch

from hsbg.actor import ActorNetwork
from hsbg.checkpoint import load_checkpoint, save_checkpoint
from hsbg.critic import CriticNetwork
from hsbg.ppo import PPOConfig


def test_checkpoint_round_trip(tmp_path):
    actor = ActorNetwork(4, 2)
    critic = CriticNetwork(4)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    critic_optimizer = torch.optim.Adam(critic.parameters())
    path = tmp_path / "ppo.pt"

    save_checkpoint(
        path,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        iteration=12,
        seed=99,
        ppo_config=PPOConfig(epochs=3),
    )

    restored_actor = ActorNetwork(4, 2)
    restored_critic = CriticNetwork(4)
    restored_actor_optimizer = torch.optim.Adam(restored_actor.parameters())
    restored_critic_optimizer = torch.optim.Adam(restored_critic.parameters())
    state = load_checkpoint(
        path,
        restored_actor,
        restored_critic,
        actor_optimizer=restored_actor_optimizer,
        critic_optimizer=restored_critic_optimizer,
    )

    assert state["iteration"] == 12
    assert state["seed"] == 99
    assert state["ppo_config"]["epochs"] == 3
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(restored_actor.state_dict()[name], value)
