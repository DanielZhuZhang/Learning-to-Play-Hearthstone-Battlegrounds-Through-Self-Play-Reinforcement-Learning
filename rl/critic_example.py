"""Run with python -m rl.critic_example from the repository root."""

from pathlib import Path

import torch

from hsbg.actor import ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.env import BattlegroundsEnv
from hsbg.rollout import collect_rollout


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    profile = Path(__file__).resolve().parents[1] / "data/profiles/initial-15.6.json"
    env = BattlegroundsEnv.from_profile(profile)
    actor = ActorNetwork.from_env(env).eval()
    critic = CriticNetwork.from_env(env)
    buffer = collect_rollout(env, actor, critic, seeds=[10, 11])
    optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    # Confirm that the teammate's actor reconstructs the collection probabilities.
    with torch.no_grad():
        log_probs, _ = actor.evaluate_actions(buffer.observations, buffer.action_masks, buffer.actions)
        torch.testing.assert_close(log_probs, buffer.old_log_probs, atol=1e-5, rtol=1e-5)
        before = critic.value_loss(buffer.observations, buffer.value_targets).item()

    for indices in buffer.minibatch_indices(32):
        loss = critic.value_loss(buffer.observations[indices], buffer.value_targets[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        after = critic.value_loss(buffer.observations, buffer.value_targets).item()
    print(f"Episodes: {buffer.terminated.sum().item()}, decisions: {len(buffer.actions)}")
    print(f"First decision: {env.describe_action(buffer.actions[0].item())}, "
          f"old V={buffer.old_values[0]:.4f}, advantage={buffer.advantages[0]:.4f}, "
          f"value target={buffer.value_targets[0]:.4f}")
    print(f"Critic MSE on this rollout: {before:.6f} -> {after:.6f}")


if __name__ == "__main__":
    main()
