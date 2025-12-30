"""Minimal multiprocessing training loop using REINFORCE."""
from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass, field
from typing import Dict, List

from .env import BalanceEnv, EnvConfig
from .policy import PolicyConfig, PolicyNetwork, mean, std


@dataclass
class TrainingConfig:
    iterations: int = 10
    steps_per_worker: int = 256
    num_workers: int = 2
    envs_per_worker: int = 4
    gamma: float = 0.99
    learning_rate: float = 0.05
    seed: int = 0
    env_config: EnvConfig = field(default_factory=EnvConfig)
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass
class Batch:
    obs: List[List[float]]
    actions: List[int]
    returns: List[float]


def rollout_worker(conn, env_config: EnvConfig, policy_config: PolicyConfig, envs_per_worker: int, base_seed: int):
    rng = random.Random(base_seed)
    policy = PolicyNetwork(policy_config, rng)
    envs = [BalanceEnv(env_config, seed=base_seed + i * 31) for i in range(envs_per_worker)]
    observations = [env.reset() for env in envs]

    while True:
        msg = conn.recv()
        if msg["cmd"] == "close":
            break
        if msg["cmd"] != "run":
            continue

        steps = msg["steps"]
        gamma = msg["gamma"]
        policy.set_weights(msg["weights"])

        records: List[tuple] = []
        for _ in range(steps):
            for env_id, env in enumerate(envs):
                obs = observations[env_id]
                action, _ = policy.sample_action(obs)
                next_obs, reward, done, _ = env.step(action)
                records.append((obs, action, reward, done, env_id))
                observations[env_id] = next_obs if not done else env.reset()

        returns = [0.0 for _ in records]
        tail_returns = {i: 0.0 for i in range(envs_per_worker)}
        for idx in reversed(range(len(records))):
            _, _, reward, done, env_id = records[idx]
            carry = 0.0 if done else tail_returns[env_id]
            ret = reward + gamma * carry
            returns[idx] = ret
            tail_returns[env_id] = ret

        batch_obs = [r[0] for r in records]
        batch_actions = [r[1] for r in records]
        conn.send({"obs": batch_obs, "actions": batch_actions, "returns": returns})

    conn.close()


def aggregate_batches(batches: List[Batch]) -> Batch:
    obs: List[List[float]] = []
    actions: List[int] = []
    returns: List[float] = []
    for b in batches:
        obs.extend(b.obs)
        actions.extend(b.actions)
        returns.extend(b.returns)
    return Batch(obs=obs, actions=actions, returns=returns)


def train(config: TrainingConfig) -> PolicyNetwork:
    # Choose the best available multiprocessing start method across platforms.
    # Windows lacks "fork", so fall back to a supported option instead of raising.
    available_methods = mp.get_all_start_methods()
    if "fork" in available_methods:
        mp.set_start_method("fork", force=True)
    elif "forkserver" in available_methods:
        mp.set_start_method("forkserver", force=True)
    else:
        mp.set_start_method("spawn", force=True)
    rng = random.Random(config.seed)
    policy = PolicyNetwork(config.policy_config, rng)

    parent_conns = []
    procs = []
    for worker_idx in range(config.num_workers):
        parent_conn, child_conn = mp.Pipe()
        seed = config.seed + worker_idx * 100
        proc = mp.Process(
            target=rollout_worker,
            args=(child_conn, config.env_config, config.policy_config, config.envs_per_worker, seed),
            daemon=True,
        )
        proc.start()
        parent_conns.append(parent_conn)
        procs.append(proc)

    try:
        for iteration in range(config.iterations):
            weights = policy.get_weights()
            for conn in parent_conns:
                conn.send({
                    "cmd": "run",
                    "weights": weights,
                    "steps": config.steps_per_worker,
                    "gamma": config.gamma,
                })

            batches: List[Batch] = []
            for conn in parent_conns:
                payload = conn.recv()
                batches.append(Batch(obs=payload["obs"], actions=payload["actions"], returns=payload["returns"]))

            batch = aggregate_batches(batches)
            advantages = batch.returns
            adv_mean = mean(advantages)
            adv_std = std(advantages) + 1e-6
            normalized_adv = [(a - adv_mean) / adv_std for a in advantages]
            grads = policy.gradients(batch.obs, batch.actions, normalized_adv)
            policy.apply_gradients(grads, lr=config.learning_rate)

            avg_return = mean(batch.returns)
            total_steps = len(batch.obs)
            print(f"Iteration {iteration + 1}/{config.iterations} - steps: {total_steps} avg_return: {avg_return:.3f}")
    finally:
        for conn in parent_conns:
            conn.send({"cmd": "close"})
        for proc in procs:
            proc.join(timeout=1)

    return policy


if __name__ == "__main__":
    train(TrainingConfig(iterations=3, steps_per_worker=128, num_workers=2, envs_per_worker=4))
