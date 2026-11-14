# Critic NN and Sample Memory

## Critic NN

The critic predicts the learner's expected discounted placement reward from the current recruit observation. For example, when the learner can buy a minion or upgrade the tavern, the critic produces one estimate for that position. The actor assigns probabilities to the available actions.

| Stage | Specification for the default profile |
| --- | --- |
| Input | 1,076 float32 features from the environment. They describe the learner, 24 card slots, opponents' public status and the most recent combat. The input width comes from `env.observation_size`. |
| Hidden layers | Two fully connected layers of width 128, each followed by tanh. The critic has its own parameters so a value-only update leaves the team's actor fixed. |
| Output | A linear layer returns one scalar per observation. A batch of shape `[B, D]` produces `[B]`. |
| Learning target | The old estimate plus the raw generalized advantage for that decision. Targets stay fixed throughout each update. |
| Loss | Mean squared error between the current predictions and the stored targets. The example uses PyTorch autograd and Adam with a learning rate of 0.0003. |

## Sample Memory (on-policy rollout buffer)

Call `collect_rollout(env, actor, critic, seeds)` once per collection. Each seed starts a new game. The actor samples from its masked policy and the critic evaluates the same observation before the environment advances. Both models keep their parameters fixed during collection. Each row belongs to one learner decision.

| Buffer field | Shape and meaning |
| --- | --- |
| `observations`, `next_observations` | Two float32 tensors of shape `[N, D]`, recording the learner's view immediately before and after each action. |
| `action_masks`, `actions` | A boolean `[N, K]` legal-action mask at sampling time and an integer `[N]` with the selected indices. The default profile has 112 actions. |
| `old_log_probs`, `old_values` | Two float32 vectors recording the sampled action's log-probability and the critic's original prediction. |
| `rewards`, `terminated`, `truncated` | Immediate rewards and two Boolean boundary flags. The simulator gives zero intermediate reward and a terminal placement reward from −1 to +1. |
| `advantages`, `value_targets` | One raw advantage for the actor and one fixed regression target for the critic per row. |

**How the targets are computed.** Work backward through each episode with discount γ = 1 and GAE parameter λ = 0.95 by default. At each step compute the temporal-difference error and carry the next step's advantage backward.

```text
delta[t]     = reward[t] + gamma * next_value - old_value[t]
advantage[t] = delta[t] + gamma * lambda * next_advantage
target[t]    = old_value[t] + advantage[t]
value_loss   = mean((critic(observation[t]) - target[t])²)
```

At the final decision, set the next advantage to zero. Use zero for the next value at true termination. If a wrapper truncates a game, use the critic's estimate of its final observation before resetting. The built-in environment treats its own turn limit as terminal. Begin a separate backward pass for each game.

**Hand check.** Suppose the last two rewards are 0 and +1, with old values 0.2 and 0.4. For easy arithmetic use γ = 0.9 and λ = 0.5. The final advantage is 1 − 0.4 = 0.6. One step earlier the TD error is 0 + 0.9 × 0.4 − 0.2 = 0.16. The earlier advantage is 0.16 + 0.9 × 0.5 × 0.6 = 0.43, giving value targets of 0.63 and 1.0. With λ = 1 on a complete game the estimate reduces to the board's discounted return minus the old value.

**Handoff to PPO.** For each training epoch use `buffer.minibatch_indices(batch_size)` to shuffle the rows together. Recompute chosen-action log-probabilities with the actor's `evaluate_actions` and the original masks. The PPO ratio is `exp(new_log_prob - old_log_prob)`. Use the stored advantages for the policy loss and the fixed targets for the value loss. If enabling actor dropout, use evaluation mode during both sampling and probability recomputation. After the update, collect a fresh buffer with the updated policy.

## Run the integration example

From the repository root with Python 3.11 or later:

```bash
python -m pip install -e '.[rl,dev]'
python -m rl.critic_example
python -m pytest -q tests/test_critic_rollout.py
```

The example samples two seeded games with the team's actor, checks the original action probabilities, and fits the critic for one epoch. The reported MSE measures fit to that batch. Playing strength requires evaluation on separate games.
