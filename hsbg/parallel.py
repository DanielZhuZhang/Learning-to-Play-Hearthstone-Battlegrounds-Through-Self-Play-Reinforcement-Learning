"""Collect one PPO rollout with several worker processes.

The serial ``collect_rollout`` plays its episodes one after another on one CPU
core, and nearly all of that time is game simulation, not the networks. This
module splits the episode seeds across worker processes. Each worker:

1. builds its own ``BattlegroundsEnv`` once, when the pool starts;
2. each iteration, receives the current actor and critic weights, the frozen
   self-play snapshots, the seat -> snapshot assignment and its share of seeds;
3. plays those episodes with ``collect_rollout`` (same code as serial, so GAE
   is still computed per episode);
4. sends the finished decisions back as NumPy arrays.

The main process concatenates the pieces in seed order and runs the PPO
update as usual. Workers never train; they only play games.

Results are reproducible for a fixed worker count. Different worker counts
give different (equally valid) action samples, because each chunk of seeds
seeds its own torch RNG.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hsbg.actor import ActorConfig, ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.env import BattlegroundsEnv
from hsbg.rollout import RolloutBuffer, collect_rollout
from hsbg.selfplay import ActorPolicy, SnapshotPool

# ---------------------------------------------------------------- helpers -- #

StateArrays = dict[str, np.ndarray]


def _to_arrays(module: torch.nn.Module) -> StateArrays:
    """State dict as plain NumPy arrays (pickles cheaply, no shared-memory handles)."""
    return {key: value.detach().cpu().numpy() for key, value in module.state_dict().items()}


def _load_arrays(module: torch.nn.Module, arrays: StateArrays) -> None:
    module.load_state_dict({key: torch.from_numpy(value) for key, value in arrays.items()})


def split_seeds(seeds: Sequence[int], parts: int) -> list[list[int]]:
    """Contiguous, near-equal chunks in order; never returns an empty chunk."""
    seeds = list(seeds)
    parts = max(1, min(parts, len(seeds)))
    size, extra = divmod(len(seeds), parts)
    chunks, start = [], 0
    for index in range(parts):
        stop = start + size + (1 if index < extra else 0)
        chunks.append(seeds[start:stop])
        start = stop
    return chunks


def concat_buffers(buffers: Sequence[RolloutBuffer]) -> RolloutBuffer:
    """Stack rollout pieces row-wise, keeping their order."""
    if not buffers:
        raise ValueError("No rollout pieces to concatenate.")
    return RolloutBuffer(**{
        field.name: torch.cat([getattr(buffer, field.name) for buffer in buffers])
        for field in fields(RolloutBuffer)
    })


# --------------------------------------------------------- worker process -- #

_WORKER_ENV: BattlegroundsEnv | None = None


def _init_worker(profile_path: str, env_kwargs: dict[str, Any]) -> None:
    global _WORKER_ENV
    torch.set_num_threads(1)
    _WORKER_ENV = BattlegroundsEnv.from_profile(profile_path, **env_kwargs)


def _collect_chunk(job: dict[str, Any]) -> dict[str, np.ndarray]:
    env = _WORKER_ENV
    if env is None:
        raise RuntimeError("Worker environment was not initialised.")

    actor = ActorNetwork.from_env(env, job["actor_config"])
    _load_arrays(actor, job["actor"])
    actor.eval()
    critic = CriticNetwork(env.observation_size, job["critic_hidden"])
    _load_arrays(critic, job["critic"])
    critic.eval()

    snapshots = []
    for arrays in job["snapshots"]:
        snapshot = ActorNetwork.from_env(env, job["actor_config"])
        _load_arrays(snapshot, arrays)
        snapshots.append(ActorPolicy(snapshot, frozen_copy=False))
    env.opponents.clear()
    for seat, index in job["assignment"].items():
        if index is not None:
            env.opponents[seat] = snapshots[index]

    torch.manual_seed(job["torch_seed"])
    buffer = collect_rollout(
        env, actor, critic, job["seeds"],
        gamma=job["gamma"], gae_lambda=job["gae_lambda"],
    )
    return {field.name: getattr(buffer, field.name).numpy() for field in fields(RolloutBuffer)}


# ------------------------------------------------------------ main process -- #


class ParallelRolloutCollector:
    """A pool of rollout workers. Use as a context manager, or call ``close()``.

    ``collect`` is a drop-in replacement for ``collect_rollout`` that also
    takes the self-play pool and assignment from ``SnapshotPool.draw``.
    """

    def __init__(
        self,
        profile_path: str | Path,
        workers: int,
        env_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """``env_kwargs`` go to each worker's ``BattlegroundsEnv`` (e.g. ``action_penalty``)."""
        if workers < 1:
            raise ValueError("workers must be positive.")
        self.workers = workers
        # "spawn" works the same on Windows, macOS and Linux and avoids
        # forking a process that has already started torch threads.
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(profile_path), dict(env_kwargs or {})),
        )

    def __enter__(self) -> ParallelRolloutCollector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def collect(
        self,
        actor: ActorNetwork,
        critic: CriticNetwork,
        seeds: Sequence[int],
        *,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        pool: SnapshotPool | None = None,
        assignment: Mapping[str, int | None] | None = None,
    ) -> RolloutBuffer:
        if not isinstance(actor.config, ActorConfig) or actor.config.dropout:
            raise ValueError("Parallel collection needs an actor without dropout.")
        if (pool is None) != (assignment is None):
            raise ValueError("Pass both pool and assignment for self-play, or neither.")
        device = next(actor.parameters()).device

        shared = {
            "actor": _to_arrays(actor),
            "actor_config": actor.config,
            "critic": _to_arrays(critic),
            "critic_hidden": critic.layers[0].out_features,
            "snapshots": [_to_arrays(policy.actor) for _, policy in pool.snapshots] if pool else [],
            "assignment": dict(assignment or {}),
            "gamma": gamma,
            "gae_lambda": gae_lambda,
        }
        jobs = [
            {**shared, "seeds": chunk, "torch_seed": chunk[0]}
            for chunk in split_seeds(seeds, self.workers)
        ]
        pieces = list(self._executor.map(_collect_chunk, jobs))
        return concat_buffers([
            RolloutBuffer(**{name: torch.from_numpy(array).to(device) for name, array in piece.items()})
            for piece in pieces
        ])
