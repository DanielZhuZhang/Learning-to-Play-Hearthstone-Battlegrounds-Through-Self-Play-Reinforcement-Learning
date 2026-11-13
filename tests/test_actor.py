from pathlib import Path
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hsbg.actor import ActorConfig, ActorNetwork, ActorPolicy  # noqa: E402
from hsbg.data.profiles import load_game_data  # noqa: E402
from hsbg.env import BattlegroundsEnv, IndexPolicy  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE = PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json"
SMALL = ActorConfig(d_model=32, num_heads=4, num_layers=1, feedforward_dim=64)


@pytest.fixture(scope="module")
def content():
    return load_game_data(PROFILE)


@pytest.fixture()
def env(content):
    return BattlegroundsEnv(*content)


@pytest.fixture()
def actor(env):
    torch.manual_seed(0)
    return ActorNetwork.from_env(env, SMALL).eval()


def _rollout_batch(env, steps: int, seed: int = 0):
    """Collect observations and masks by playing random legal actions."""
    rng = random.Random(seed)
    obs, info = env.reset(seed=seed)
    observations, masks = [obs], [info["action_mask"]]
    for _ in range(steps):
        index = rng.choice(np.flatnonzero(info["action_mask"]).tolist())
        obs, _, done, _, info = env.step(index)
        if done:
            obs, info = env.reset(seed=seed + 1)
        observations.append(obs)
        masks.append(info["action_mask"])
    return torch.as_tensor(np.stack(observations)), torch.as_tensor(np.stack(masks))


def test_logits_shape_and_masking(env, actor) -> None:
    obs, mask = _rollout_batch(env, 20)
    logits = actor(obs, mask)

    assert logits.shape == (obs.shape[0], env.action_size)
    assert torch.isfinite(logits[mask]).all()
    assert (logits[~mask] == torch.finfo(logits.dtype).min).all()
    probs = torch.softmax(logits, -1)
    assert torch.allclose(probs.sum(-1), torch.ones(obs.shape[0]))
    assert (probs[~mask] == 0).all()


def test_single_observation_matches_batched(env, actor) -> None:
    obs, mask = _rollout_batch(env, 10)
    batched = actor(obs, mask)
    for row in range(obs.shape[0]):
        assert torch.allclose(actor(obs[row], mask[row]), batched[row], atol=1e-5)


def test_initial_policy_is_near_uniform_over_legal_actions(env, actor) -> None:
    obs, mask = _rollout_batch(env, 10)
    _, _, entropy = actor.act(obs, mask)
    max_entropy = torch.log(mask.sum(-1).float())
    assert torch.allclose(entropy, max_entropy, atol=0.05)


def test_act_and_evaluate_actions_agree(env, actor) -> None:
    obs, mask = _rollout_batch(env, 15)
    torch.manual_seed(1)
    action, log_prob, entropy = actor.act(obs, mask)

    assert mask[torch.arange(len(action)), action].all()
    again_log_prob, again_entropy = actor.evaluate_actions(obs, mask, action)
    assert torch.allclose(log_prob, again_log_prob, atol=1e-6)
    assert torch.allclose(entropy, again_entropy, atol=1e-6)

    greedy, _, _ = actor.act(obs, mask, deterministic=True)
    assert torch.equal(greedy, actor(obs, mask).argmax(-1))


def test_shop_logits_follow_shop_order(env, actor) -> None:
    obs, info = env.reset(seed=4)
    layout, encoder = env.action_layout, env.encoder
    shop = encoder.sections["shop"]
    blocks = obs[shop].reshape(layout.shop_slots, encoder.card_features).copy()
    assert blocks[0, 0] == 1.0 and blocks[1, 0] == 1.0  # at least two shop cards

    swapped = obs.copy()
    swapped_blocks = blocks.copy()
    swapped_blocks[[0, 1]] = swapped_blocks[[1, 0]]
    swapped[shop] = swapped_blocks.reshape(-1)

    before = actor(torch.as_tensor(obs))
    after = actor(torch.as_tensor(swapped))
    assert torch.allclose(before[layout.buy], after[layout.buy + 1], atol=1e-5)
    assert torch.allclose(before[layout.buy + 1], after[layout.buy], atol=1e-5)


def test_gradients_reach_every_head(env, content) -> None:
    torch.manual_seed(0)
    actor = ActorNetwork.from_env(env, SMALL)
    obs, _ = _rollout_batch(env, 40)
    actor(obs).sum().backward()
    for name, parameter in actor.named_parameters():
        assert parameter.grad is not None, name


def test_rejects_wrong_observation_or_mask_size(actor) -> None:
    with pytest.raises(ValueError):
        actor(torch.zeros(3, actor.observation_size + 1))
    with pytest.raises(ValueError):
        actor(
            torch.zeros(3, actor.observation_size),
            torch.ones(3, actor.action_size - 1, dtype=torch.bool),
        )


def test_targeted_play_adds_target_fit_to_the_card_play_score(env, actor) -> None:
    layout = env.action_layout
    obs, _ = _rollout_batch(env, 30)
    with torch.no_grad():
        torch.nn.init.zeros_(actor.target_query.weight)
        torch.nn.init.zeros_(actor.target_query.bias)
        logits = actor(obs)
    play = logits[:, layout.play : layout.play_targeted]
    targeted = logits[:, layout.play_targeted : layout.sell].reshape(
        -1, layout.hand_slots, layout.board_slots
    )
    # With no target preference, every target of hand card h scores like playing h.
    assert torch.allclose(targeted, play.unsqueeze(-1).expand_as(targeted), atol=1e-6)


def test_train_and_eval_modes_agree(env, actor) -> None:
    obs, mask = _rollout_batch(env, 40)
    with torch.no_grad():
        eval_logits = actor.eval()(obs, mask)
        train_logits = actor.train()(obs, mask)
    assert torch.allclose(eval_logits[mask], train_logits[mask], atol=1e-5)


def test_encode_exposes_shared_trunk(env, actor) -> None:
    obs, _ = _rollout_batch(env, 5)
    g, shop, hand, board = actor.encode(obs)
    d = SMALL.d_model
    layout = env.action_layout
    assert g.shape == (obs.shape[0], d)
    assert shop.shape == (obs.shape[0], layout.shop_slots, d)
    assert hand.shape == (obs.shape[0], layout.hand_slots, d)
    assert board.shape == (obs.shape[0], layout.board_slots, d)


def test_actor_plays_full_episode_with_only_legal_actions(env, actor) -> None:
    obs, info = env.reset(seed=2)
    done, steps = False, 0
    with torch.no_grad():
        while not done:
            mask = torch.as_tensor(info["action_mask"])
            action, _, _ = actor.act(torch.as_tensor(obs), mask)
            assert info["action_mask"][int(action)]
            obs, reward, done, _, info = env.step(int(action))
            steps += 1
    assert info["placement"] is not None
    assert -1.0 <= reward <= 1.0
    assert steps > 0


def test_actor_policy_is_an_index_policy_opponent(content, actor) -> None:
    policy = ActorPolicy(actor)
    assert isinstance(policy, IndexPolicy)
    assert policy.actor is not actor
    assert not any(p.requires_grad for p in policy.actor.parameters())

    def play(seed: int) -> list[int]:
        env = BattlegroundsEnv(*content, opponents={"player-2": ActorPolicy(actor)})
        rng = random.Random(seed)
        _, info = env.reset(seed=seed)
        chosen, done = [], False
        while not done:
            index = rng.choice(np.flatnonzero(info["action_mask"]).tolist())
            chosen.append(index)
            _, _, done, _, info = env.step(index)
        return chosen + [info["placement"]]

    assert play(5) == play(5)  # seeded matches stay reproducible


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ActorConfig(d_model=30, num_heads=4)


def test_default_config_parameter_count(env) -> None:
    actor = ActorNetwork.from_env(env)
    count = sum(p.numel() for p in actor.parameters())
    assert 100_000 < count < 2_000_000
