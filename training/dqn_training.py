"""
DQN training script with hyperparameter sweep for the Cyber Threat Hunt environment.
Runs 10+ experiments with different hyperparameter combinations.
"""

import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import CyberThreatHuntEnv
from training.callbacks import TrainingMetricsCallback


def make_env(seed=None):
    def _init():
        env = CyberThreatHuntEnv(num_nodes=14, num_threats=4, max_steps=100)
        env.reset(seed=seed)
        return env
    return _init


# 12 hyperparameter combinations
EXPERIMENTS = [
    {"name": "dqn_baseline", "lr": 1e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_high_lr", "lr": 5e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_low_lr", "lr": 1e-5, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_low_gamma", "lr": 1e-4, "gamma": 0.9, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_high_gamma", "lr": 1e-4, "gamma": 0.999, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_small_batch", "lr": 1e-4, "gamma": 0.99, "batch_size": 32,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_large_batch", "lr": 1e-4, "gamma": 0.99, "batch_size": 128,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_small_buffer", "lr": 1e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 10000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [128, 128]},

    {"name": "dqn_slow_explore", "lr": 1e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.01, "eps_fraction": 0.5,
     "net_arch": [128, 128]},

    {"name": "dqn_fast_explore", "lr": 1e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.1, "eps_fraction": 0.1,
     "net_arch": [128, 128]},

    {"name": "dqn_deep_net", "lr": 1e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [256, 256, 128]},

    {"name": "dqn_wide_net", "lr": 1e-4, "gamma": 0.99, "batch_size": 64,
     "buffer_size": 50000, "eps_start": 1.0, "eps_end": 0.05, "eps_fraction": 0.3,
     "net_arch": [512, 256]},
]


def train_single(exp, total_timesteps=50000, seed=42):
    """Train a single DQN experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp['name']}")
    print(f"lr={exp['lr']}, gamma={exp['gamma']}, batch={exp['batch_size']}, "
          f"buffer={exp['buffer_size']}, eps={exp['eps_start']}->{exp['eps_end']}, "
          f"arch={exp['net_arch']}")
    print(f"{'='*60}")

    env = DummyVecEnv([make_env(seed=seed)])
    eval_env = DummyVecEnv([make_env(seed=seed + 100)])

    model_dir = os.path.join("models", "dqn", exp["name"])
    log_dir = os.path.join("results", "dqn", exp["name"])
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    metrics_callback = TrainingMetricsCallback(log_path=log_dir)
    callback = CallbackList([eval_callback, metrics_callback])

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=exp["lr"],
        gamma=exp["gamma"],
        batch_size=exp["batch_size"],
        buffer_size=exp["buffer_size"],
        exploration_initial_eps=exp["eps_start"],
        exploration_final_eps=exp["eps_end"],
        exploration_fraction=exp["eps_fraction"],
        policy_kwargs={"net_arch": exp["net_arch"]},
        tensorboard_log=os.path.join("results", "tensorboard"),
        verbose=1,
        seed=seed,
    )

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=f"dqn_{exp['name']}",
    )
    train_time = time.time() - start

    model.save(os.path.join(model_dir, "final_model"))

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"\nResult: {mean_reward:.2f} +/- {std_reward:.2f} ({train_time:.1f}s)")

    env.close()
    eval_env.close()

    return {
        "experiment": exp["name"],
        "algorithm": "DQN",
        "lr": exp["lr"],
        "gamma": exp["gamma"],
        "batch_size": exp["batch_size"],
        "buffer_size": exp["buffer_size"],
        "eps_start": exp["eps_start"],
        "eps_end": exp["eps_end"],
        "eps_fraction": exp["eps_fraction"],
        "net_arch": str(exp["net_arch"]),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "training_time_s": train_time,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DQN Hyperparameter Sweep")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run a specific experiment by name")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.experiment]
        if not experiments:
            print(f"Experiment '{args.experiment}' not found.")
            return

    results = []
    for exp in experiments:
        result = train_single(exp, args.timesteps, args.seed)
        results.append(result)

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "dqn_results.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAll DQN results saved to {output_path}")


if __name__ == "__main__":
    main()
