from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hsbg.learning import BattlegroundsEnv, load_content  # noqa: E402
from hsbg.ppo import (  # noqa: E402
    ActorCritic,
    PPOConfig,
    collect_episodes,
    episode_gae,
    load_checkpoint,
    save_checkpoint,
    update_policy,
)


def test_gae_terminal_returns_and_single_step():
    advantages, returns = episode_gae([0, 1], [0.2, 0.3], gamma=1, gae_lambda=1)
    np.testing.assert_allclose(advantages, [0.8, 0.7])
    np.testing.assert_allclose(returns, [1, 1])
    advantages, returns = episode_gae([-1], [0.4])
    np.testing.assert_allclose(advantages, [-1.4])
    np.testing.assert_allclose(returns, [-1])


def test_masked_distribution_has_zero_invalid_probability_and_finite_gradients():
    model = ActorCritic(3, 4, 8)
    features = torch.zeros((2, 3))
    masks = torch.tensor([[True, False, False, False], [True, False, True, False]])
    distribution, values = model(features, masks)
    assert torch.all(distribution.probs[~masks] == 0)
    assert distribution.probs[0, 0] == 1
    loss = (
        -distribution.log_prob(torch.tensor([0, 2])).mean()
        - distribution.entropy().mean()
        + values.square().mean()
    )
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="legal action"):
        model(features, torch.zeros_like(masks))


def test_real_rollout_update_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(5)
    torch.set_num_threads(1)
    profile, cards = load_content()
    profile = replace(profile, rules=replace(profile.rules, max_match_turns=3))
    env = BattlegroundsEnv(profile, cards, max_actions=4)
    config = PPOConfig(hidden_size=16, epochs=2, minibatch_size=8)
    model = ActorCritic(env.encoder.size, env.codec.n_actions, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    batch, results = collect_episodes(env, model, 2, 10, config)
    assert len(results) == 2
    assert batch["masks"][np.arange(len(batch["actions"])), batch["actions"]].all()
    old = [p.detach().clone() for p in model.parameters()]
    losses = update_policy(model, optimizer, batch, config)
    assert all(np.isfinite(x) for x in losses.values())
    assert any(not torch.equal(a, b) for a, b in zip(old, model.parameters()))
    obs, _ = env.reset(seed=70)
    expected = model.act(obs, deterministic=True)
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, config, env, {"training_seed": 10, "episodes": 2})
    restored, metadata = load_checkpoint(path, env)
    assert restored.act(obs, deterministic=True) == expected
    assert metadata["episodes"] == 2
    incompatible = BattlegroundsEnv(profile, cards, max_actions=5)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_checkpoint(path, incompatible)


def test_cli_rejects_training_seeds_in_evaluation(tmp_path):
    from hsbg.evaluate import main as evaluate_main

    profile, cards = load_content()
    env = BattlegroundsEnv(profile, cards)
    config = PPOConfig(hidden_size=8)
    model = ActorCritic(env.encoder.size, env.codec.n_actions, 8)
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, config, env, {"training_seed": 10, "episodes": 20})
    with pytest.raises(SystemExit) as exc:
        evaluate_main(
            [
                "--policy",
                "ppo",
                "--checkpoint",
                str(path),
                "--seed",
                "15",
                "--games",
                "2",
                "--output",
                str(tmp_path / "eval.json"),
            ]
        )
    assert exc.value.code == 2
