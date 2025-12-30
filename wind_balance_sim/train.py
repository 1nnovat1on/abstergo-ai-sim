"""Minimal multiprocessing training loop using REINFORCE."""
from __future__ import annotations

import os
import multiprocessing as mp
import random
import time
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
    seed: int | None = None
    save_path: str | None = "checkpoints/latest_policy.json"
    load_path: str | None = None
    render_demo: bool = False
    demo_steps: int = 240
    demo_sleep: float = 0.0
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


def _choose_seed(seed: int | None) -> int:
    return seed if seed is not None else random.randrange(0, 2**31)


def render_ascii(env: BalanceEnv, width: int = 64) -> str:
    c = env.config
    def project(value: float, domain: float) -> int:
        return max(0, min(width - 1, int((value / domain) * (width - 1))))

    plat_x = project(env.platform.x, c.width)
    stick_x = project(env.stick.x, c.width)
    ball_x = project(env.ball.x, c.width)

    base = [" " for _ in range(width)]
    base[plat_x] = "="
    base[stick_x] = "|"
    base[ball_x] = "o"
    return "".join(base)


def render_episode(policy: PolicyNetwork, env_config: EnvConfig, steps: int, sleep: float = 0.0) -> None:
    env = BalanceEnv(env_config)
    obs = env.reset()
    for step in range(steps):
        action, _ = policy.sample_action(obs)
        obs, reward, done, info = env.step(action)
        print(f"demo step {step:04d} action={action:+d} reward={reward:+.1f} supported={info['ball_supported']}\n{render_ascii(env)}")
        if sleep:
            time.sleep(sleep)
        if done:
            print("episode ended early")
            break


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
    seed = _choose_seed(config.seed)
    rng = random.Random(seed)
    policy: PolicyNetwork
    load_path = config.load_path or config.save_path
    if load_path and os.path.exists(load_path):
        policy = PolicyNetwork.load(load_path, rng)
        print(f"Loaded existing policy from {load_path}")
    else:
        policy = PolicyNetwork(config.policy_config, rng)
        if load_path:
            print(f"No existing policy at {load_path}, starting fresh")

    parent_conns = []
    procs = []
    for worker_idx in range(config.num_workers):
        parent_conn, child_conn = mp.Pipe()
        worker_seed = seed + worker_idx * 100
        proc = mp.Process(
            target=rollout_worker,
            args=(child_conn, config.env_config, config.policy_config, config.envs_per_worker, worker_seed),
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

            if config.save_path:
                policy.save(config.save_path)
                print(f"Saved policy to {config.save_path}")
    finally:
        for conn in parent_conns:
            conn.send({"cmd": "close"})
        for proc in procs:
            proc.join(timeout=1)

    if config.render_demo:
        print("\nRendering a demo episode with the latest policy:\n")
        render_episode(policy, config.env_config, config.demo_steps, sleep=config.demo_sleep)

    return policy


if __name__ == "__main__":
    train(TrainingConfig(iterations=3, steps_per_worker=128, num_workers=2, envs_per_worker=4))
