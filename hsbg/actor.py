"""Actor (policy) network for the single-seat ``BattlegroundsEnv``.

The actor maps an encoded observation and a legal-action mask to a
categorical distribution over the ``ActionLayout`` indices.

Architecture (sizes shown for the default rules and the 15.6 profile)::

    obs (1076) ──split by ObservationEncoder.sections──┐
      self + opponents + last_combat (44) ─ MLP ─► global token   (1 × d)
      shop  (7  × 43)  ─ shared card MLP ─► + zone emb            (7 × d)
      hand  (10 × 43)  ─ shared card MLP ─► + zone emb            (10 × d)
      board (7  × 43)  ─ shared card MLP ─► + zone emb + pos emb  (7 × d)
                              │
           Transformer encoder over the 25 tokens
           (empty slots are padded out of attention)
                              │
      ┌───────────────┬───────┴────────┬──────────────────┬──────────────┐
    shop tokens    hand tokens     hand × board        board tokens   global token
      buy (7)       play (10)     play→target (70)   sell / move L/R   reroll, freeze,
                                  (play score +        (3 × 7)          upgrade, end (4)
                                   bilinear fit)
                              │
          112 logits in ActionLayout order → mask illegal → Categorical

Design notes:

* One card MLP is shared by every slot, so what the network learns about a
  card transfers between shop, hand and board. The card one-hot at the end
  of each card block passes through a linear layer, which acts as a learned
  card embedding.
* Only board tokens get a position embedding, because only board order
  matters (attack order). Shop and hand logits are therefore
  permutation-equivariant: reordering the shop reorders the buy logits.
* Action logits are scored per slot ("pointer" style) rather than by one big
  dense layer, so the same weights score buying a card in slot 0 or slot 6.
* A targeted Battlecry's logit is the card's play score plus a
  hand-card × board-card fit score, so "is this card worth playing" is
  learned once and shared by its untargeted and targeted forms.
* ``encode`` returns the shared trunk so a critic can reuse it.
* Final head layers start near zero, so the untrained policy is close to
  uniform over the legal actions.

Requires ``torch`` (``pip install -e .[rl]``).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from hsbg.env import ActionLayout, BattlegroundsEnv, ObservationEncoder

_CARD_ZONES = ("shop", "hand", "board")
_GLOBAL_SECTIONS = ("self", "opponents", "last_combat")


@dataclass(frozen=True, slots=True)
class ActorConfig:
    """Hyperparameters of ``ActorNetwork``."""

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    head_init_gain: float = 0.01

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_layers < 0:
            raise ValueError("num_layers must not be negative.")


class ActorNetwork(nn.Module):
    """Masked categorical policy over ``ActionLayout`` indices."""

    def __init__(
        self,
        encoder: ObservationEncoder,
        layout: ActionLayout,
        config: ActorConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or ActorConfig()
        self.config = config
        self.layout = layout
        self.observation_size = encoder.size
        self.action_size = layout.size
        self.card_features = encoder.card_features
        self.slots = {
            "shop": layout.shop_slots,
            "hand": layout.hand_slots,
            "board": layout.board_slots,
        }
        self.sections = {
            name: (section.start, section.stop) for name, section in encoder.sections.items()
        }
        self.present_index = encoder.card_feature_names.index("present")
        global_size = sum(
            encoder.sections[name].stop - encoder.sections[name].start
            for name in _GLOBAL_SECTIONS
        )
        d = config.d_model

        self.global_encoder = _mlp(global_size, d, d)
        self.card_encoder = _mlp(self.card_features, d, d)
        self.zone_embedding = nn.Embedding(len(_CARD_ZONES), d)
        self.board_position = nn.Embedding(layout.board_slots, d)

        if config.num_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module | None = nn.TransformerEncoder(
                layer, config.num_layers, enable_nested_tensor=False
            )
        else:
            self.transformer = None
        self.final_norm = nn.LayerNorm(d)

        # Every slot head sees its own token and the global token.
        self.buy_head = _head(2 * d, d, 1, config.head_init_gain)
        self.play_head = _head(2 * d, d, 1, config.head_init_gain)
        self.board_head = _head(2 * d, d, 3, config.head_init_gain)  # sell, left, right
        self.global_head = _head(d, d, 4, config.head_init_gain)  # reroll, freeze, upgrade, end
        self.target_query = nn.Linear(2 * d, d)
        self.target_key = nn.Linear(2 * d, d)
        nn.init.orthogonal_(self.target_query.weight, config.head_init_gain)
        nn.init.zeros_(self.target_query.bias)

    @classmethod
    def from_env(
        cls, env: BattlegroundsEnv, config: ActorConfig | None = None
    ) -> ActorNetwork:
        return cls(env.encoder, env.action_layout, config)

    # -- forward ------------------------------------------------------------ #

    def forward(
        self, observation: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``(batch, action_size)`` logits; illegal actions get a huge negative.

        ``observation`` may be ``(obs_size,)`` or ``(batch, obs_size)``; the
        output has the matching batch shape.
        """
        squeeze = observation.dim() == 1
        if squeeze:
            observation = observation.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
        if observation.shape[-1] != self.observation_size:
            raise ValueError(
                f"Expected observations of size {self.observation_size}, "
                f"got {observation.shape[-1]}."
            )
        logits = self._logits(observation.float())
        if action_mask is not None:
            if action_mask.shape != logits.shape:
                raise ValueError(
                    f"Expected an action mask of shape {tuple(logits.shape)}, "
                    f"got {tuple(action_mask.shape)}."
                )
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits.squeeze(0) if squeeze else logits

    def encode(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared trunk: ``(global, shop, hand, board)`` embeddings.

        Shapes are ``(B, d)``, ``(B, shop_slots, d)``, ``(B, hand_slots, d)``
        and ``(B, board_slots, d)``. A critic can reuse this trunk and put a
        value head on the global embedding.
        """
        batch = obs.shape[0]
        global_input = torch.cat([self._section(obs, name) for name in _GLOBAL_SECTIONS], -1)
        tokens = [self.global_encoder(global_input).unsqueeze(1)]
        padding = [torch.zeros(batch, 1, dtype=torch.bool, device=obs.device)]
        for zone_id, zone in enumerate(_CARD_ZONES):
            cards = self._section(obs, zone).reshape(batch, self.slots[zone], self.card_features)
            embedded = self.card_encoder(cards) + self.zone_embedding.weight[zone_id]
            if zone == "board":
                embedded = embedded + self.board_position.weight
            tokens.append(embedded)
            padding.append(cards[..., self.present_index] <= 0.5)
        x = torch.cat(tokens, 1)
        pad = torch.cat(padding, 1)
        if self.transformer is not None:
            x = self.transformer(x, src_key_padding_mask=pad)
        x = self.final_norm(x)
        shop, hand, board = torch.split(
            x[:, 1:], [self.slots["shop"], self.slots["hand"], self.slots["board"]], 1
        )
        return x[:, 0], shop, hand, board

    def _logits(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        layout = self.layout
        g, shop, hand, board = self.encode(obs)

        def with_global(slots: torch.Tensor) -> torch.Tensor:
            return torch.cat([slots, g.unsqueeze(1).expand_as(slots)], -1)

        shop, hand, board = with_global(shop), with_global(hand), with_global(board)
        board_logits = self.board_head(board)  # (B, board, 3)
        play_logits = self.play_head(hand).squeeze(-1)  # (B, hand)
        # Targeted play = "how good is playing this card" + "how good is this target".
        # Sharing the play score keeps targeted and untargeted plays on one scale.
        target_fit = torch.einsum(
            "bhd,btd->bht", self.target_query(hand), self.target_key(board)
        ) / math.sqrt(self.config.d_model)
        targeted = play_logits.unsqueeze(-1) + target_fit  # (B, hand, board)
        global_logits = self.global_head(g)

        logits = obs.new_zeros(batch, self.action_size)
        logits[:, layout.buy : layout.play] = self.buy_head(shop).squeeze(-1)
        logits[:, layout.play : layout.play_targeted] = play_logits
        # Row-major (hand, board) matches ActionLayout: play_targeted + h * board_slots + b.
        logits[:, layout.play_targeted : layout.sell] = targeted.reshape(batch, -1)
        logits[:, layout.sell : layout.move_left] = board_logits[..., 0]
        logits[:, layout.move_left : layout.move_right] = board_logits[..., 1]
        logits[:, layout.move_right : layout.reroll] = board_logits[..., 2]
        logits[:, layout.reroll : layout.size] = global_logits
        return logits

    def _section(self, obs: torch.Tensor, name: str) -> torch.Tensor:
        start, stop = self.sections[name]
        return obs[:, start:stop]

    # -- policy helpers ----------------------------------------------------- #

    def distribution(self, observation: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
        return Categorical(logits=self(observation, action_mask))

    def act(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick actions; return ``(action, log_prob, entropy)``."""
        dist = self.distribution(observation, action_mask)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(log_prob, entropy)`` of stored actions, e.g. for a PPO update."""
        dist = self.distribution(observation, action_mask)
        return dist.log_prob(actions), dist.entropy()


class ActorPolicy:
    """``IndexPolicy`` adapter so an actor can play seats inside ``BattlegroundsEnv``.

    By default it keeps a frozen copy of the network, so later training of the
    original does not change this opponent::

        env.opponents["player-2"] = ActorPolicy(actor)
    """

    def __init__(
        self,
        actor: ActorNetwork,
        *,
        deterministic: bool = False,
        frozen_copy: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.actor = copy.deepcopy(actor) if frozen_copy else actor
        self.actor.to(device).eval()
        if frozen_copy:
            self.actor.requires_grad_(False)
        self.deterministic = deterministic
        self.device = torch.device(device)

    @torch.no_grad()
    def choose_index(
        self, observation: np.ndarray, action_mask: np.ndarray, rng: random.Random
    ) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        probs = torch.softmax(self.actor(obs, mask), -1).cpu().numpy()
        if self.deterministic:
            return int(probs.argmax())
        # Sample with the env's RNG so seeded matches stay reproducible.
        legal = np.flatnonzero(action_mask)
        weights = probs[legal]
        return int(rng.choices(legal.tolist(), weights=weights.tolist())[0])


def _mlp(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


def _head(in_features: int, hidden: int, out_features: int, gain: float) -> nn.Sequential:
    head = _mlp(in_features, hidden, out_features)
    nn.init.orthogonal_(head[-1].weight, gain)
    nn.init.zeros_(head[-1].bias)
    return head
