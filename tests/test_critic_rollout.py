from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hsbg.actor import ActorConfig, ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.env import BattlegroundsEnv
from hsbg.rollout import collect_rollout, episode_advantages


PROFILE = Path(__file__).resolve().parents[1] / "data/profiles/initial-15.6.json"


def test_critic_shapes_and_fixed_targets():
    torch.manual_seed(1)
    critic = CriticNetwork(4, hidden_size=8)
    obs = torch.randn(3, 4)
    targets = torch.tensor([0.5, -1.0, 0.75], requires_grad=True)
    values = critic(obs)
    assert values.shape == (3,)
    torch.testing.assert_close(critic(obs[0]), values[0])
    loss = critic.value_loss(obs, targets)
    torch.testing.assert_close(loss, (values - targets).square().mean())
    loss.backward()
    assert targets.grad is None
    assert critic.layers[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="one scalar"):
        critic.value_loss(obs, targets[:, None])


@pytest.mark.parametrize(("lam", "expected"), [
    (0.0, [0.16, 0.6]),       # one-step TD errors
    (0.5, [0.43, 0.6]),       # 0.16 + 0.9 * 0.5 * 0.6
    (1.0, [0.7, 0.6]),        # discounted episode return minus V
])
def test_hand_calculated_advantages(lam, expected):
    result = episode_advantages([0.0, 1.0], [0.2, 0.4], bootstrap_value=0,
                                gamma=0.9, gae_lambda=lam)
    assert result == pytest.approx(expected)


def test_complete_games_match_actor_and_do_not_share_returns():
    torch.manual_seed(3)
    env = BattlegroundsEnv.from_profile(PROFILE)
    actor = ActorNetwork.from_env(env, ActorConfig(d_model=16, num_layers=0)).eval()
    critic = CriticNetwork.from_env(env)
    batch = collect_rollout(env, actor, critic, [0, 1], gamma=1.0, gae_lambda=1.0)
    ends = batch.terminated.nonzero().flatten().tolist()
    assert len(ends) == 2 and ends[-1] == len(batch.actions) - 1
    assert not batch.truncated.any()
    start = 0
    for end in ends:
        rows = slice(start, end + 1)
        # With terminal-only rewards and gamma=lambda=1 every target is the final reward.
        torch.testing.assert_close(batch.value_targets[rows], batch.rewards[end].expand(end - start + 1))
        torch.testing.assert_close(batch.next_observations[start:end], batch.observations[start + 1:end + 1])
        start = end + 1
    assert batch.action_masks.gather(1, batch.actions[:, None]).all()
    with torch.no_grad():
        logs, _ = actor.evaluate_actions(batch.observations, batch.action_masks, batch.actions)
    torch.testing.assert_close(logs, batch.old_log_probs)
    torch.testing.assert_close(critic(batch.observations), batch.old_values)
    for field in (batch.old_log_probs, batch.advantages, batch.value_targets):
        assert not field.requires_grad
    indices = torch.cat(list(batch.minibatch_indices(7)))
    assert torch.equal(indices.sort().values, torch.arange(len(batch.actions)))


def test_truncation_bootstraps_final_observation_before_reset(monkeypatch):
    env = BattlegroundsEnv.from_profile(PROFILE, include_card_identity=False)
    actor = ActorNetwork.from_env(env, ActorConfig(d_model=16, num_layers=0)).eval()
    critic = CriticNetwork.from_env(env)
    real_step = env.step

    def one_step(action):
        obs, reward, terminal, _, info = real_step(action)
        return obs, reward, terminal, True, info

    monkeypatch.setattr(env, "step", one_step)
    batch = collect_rollout(env, actor, critic, [12, 13], gamma=0.8)
    assert batch.truncated.tolist() == [True, True]
    assert not batch.terminated.any()
    expected = batch.rewards + 0.8 * critic(batch.next_observations)
    torch.testing.assert_close(batch.value_targets, expected)
