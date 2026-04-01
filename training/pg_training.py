"""
Policy Gradient training script — PPO and A2C with hyperparameter sweeps.
Runs 10+ experiments per algorithm.
"""

import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, A2C
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


PPO_EXPERIMENTS = [
    {"name": "ppo_baseline", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_high_lr", "lr": 1e-3, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_low_lr", "lr": 1e-5, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_low_gamma", "lr": 3e-4, "gamma": 0.9, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_wide_clip", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.3, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_narrow_clip", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.1, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_high_entropy", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.05,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_no_entropy", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.0,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_long_rollout", "lr": 3e-4, "gamma": 0.99, "n_steps": 512,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_deep_net", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [256, 256, 128]},

    {"name": "ppo_few_epochs", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 64, "n_epochs": 3, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "ppo_large_batch", "lr": 3e-4, "gamma": 0.99, "n_steps": 256,
     "batch_size": 128, "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01,
     "vf_coef": 0.5, "net_arch": [128, 128]},
]

A2C_EXPERIMENTS = [
    {"name": "a2c_baseline", "lr": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_high_lr", "lr": 1e-3, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_low_lr", "lr": 1e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_low_gamma", "lr": 7e-4, "gamma": 0.9, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_high_gamma", "lr": 7e-4, "gamma": 0.999, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_long_rollout", "lr": 7e-4, "gamma": 0.99, "n_steps": 20,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_high_entropy", "lr": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.05, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_no_entropy", "lr": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.0, "vf_coef": 0.5, "net_arch": [128, 128]},

    {"name": "a2c_high_vf", "lr": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 1.0, "net_arch": [128, 128]},

    {"name": "a2c_deep_net", "lr": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [256, 256, 128]},

    {"name": "a2c_wide_net", "lr": 7e-4, "gamma": 0.99, "n_steps": 5,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [512, 256]},

    {"name": "a2c_medium_rollout", "lr": 7e-4, "gamma": 0.99, "n_steps": 10,
     "ent_coef": 0.01, "vf_coef": 0.5, "net_arch": [128, 128]},
]


def train_ppo(exp, total_timesteps=50000, seed=42):
    """Train a single PPO experiment."""
    print(f"\n{'='*60}")
    print(f"PPO Experiment: {exp['name']}")
    print(f"{'='*60}")

    env = DummyVecEnv([make_env(seed=seed)])
    eval_env = DummyVecEnv([make_env(seed=seed + 100)])

    model_dir = os.path.join("models", "pg", "ppo", exp["name"])
    log_dir = os.path.join("results", "ppo", exp["name"])
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

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=exp["lr"],
        gamma=exp["gamma"],
        n_steps=exp["n_steps"],
        batch_size=exp["batch_size"],
        n_epochs=exp["n_epochs"],
        clip_range=exp["clip_range"],
        ent_coef=exp["ent_coef"],
        vf_coef=exp["vf_coef"],
        policy_kwargs={"net_arch": exp["net_arch"]},
        tensorboard_log=os.path.join("results", "tensorboard"),
        verbose=1,
        seed=seed,
    )

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=f"ppo_{exp['name']}",
    )
    train_time = time.time() - start

    model.save(os.path.join(model_dir, "final_model"))

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Result: {mean_reward:.2f} +/- {std_reward:.2f} ({train_time:.1f}s)")

    env.close()
    eval_env.close()

    return {
        "experiment": exp["name"],
        "algorithm": "PPO",
        "lr": exp["lr"],
        "gamma": exp["gamma"],
        "n_steps": exp["n_steps"],
        "batch_size": exp["batch_size"],
        "n_epochs": exp["n_epochs"],
        "clip_range": exp["clip_range"],
        "ent_coef": exp["ent_coef"],
        "vf_coef": exp["vf_coef"],
        "net_arch": str(exp["net_arch"]),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "training_time_s": train_time,
    }


def train_a2c(exp, total_timesteps=50000, seed=42):
    """Train a single A2C experiment."""
    print(f"\n{'='*60}")
    print(f"A2C Experiment: {exp['name']}")
    print(f"{'='*60}")

    env = DummyVecEnv([make_env(seed=seed)])
    eval_env = DummyVecEnv([make_env(seed=seed + 100)])

    model_dir = os.path.join("models", "pg", "a2c", exp["name"])
    log_dir = os.path.join("results", "a2c", exp["name"])
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

    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=exp["lr"],
        gamma=exp["gamma"],
        n_steps=exp["n_steps"],
        ent_coef=exp["ent_coef"],
        vf_coef=exp["vf_coef"],
        policy_kwargs={"net_arch": exp["net_arch"]},
        tensorboard_log=os.path.join("results", "tensorboard"),
        verbose=1,
        seed=seed,
    )

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=f"a2c_{exp['name']}",
    )
    train_time = time.time() - start

    model.save(os.path.join(model_dir, "final_model"))

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Result: {mean_reward:.2f} +/- {std_reward:.2f} ({train_time:.1f}s)")

    env.close()
    eval_env.close()

    return {
        "experiment": exp["name"],
        "algorithm": "A2C",
        "lr": exp["lr"],
        "gamma": exp["gamma"],
        "n_steps": exp["n_steps"],
        "ent_coef": exp["ent_coef"],
        "vf_coef": exp["vf_coef"],
        "net_arch": str(exp["net_arch"]),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "training_time_s": train_time,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO & A2C Hyperparameter Sweep")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="both",
                        choices=["ppo", "a2c", "both"])
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    all_results = []

    if args.algorithm in ("ppo", "both"):
        experiments = PPO_EXPERIMENTS
        if args.experiment:
            experiments = [e for e in experiments if e["name"] == args.experiment]
        for exp in experiments:
            result = train_ppo(exp, args.timesteps, args.seed)
            all_results.append(result)

        os.makedirs("results", exist_ok=True)
        ppo_results = [r for r in all_results if r["algorithm"] == "PPO"]
        if ppo_results:
            with open(os.path.join("results", "ppo_results.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ppo_results[0].keys())
                writer.writeheader()
                writer.writerows(ppo_results)
            print(f"\nPPO results saved to results/ppo_results.csv")

    if args.algorithm in ("a2c", "both"):
        experiments = A2C_EXPERIMENTS
        if args.experiment:
            experiments = [e for e in experiments if e["name"] == args.experiment]
        for exp in experiments:
            result = train_a2c(exp, args.timesteps, args.seed)
            all_results.append(result)

        os.makedirs("results", exist_ok=True)
        a2c_results = [r for r in all_results if r["algorithm"] == "A2C"]
        if a2c_results:
            with open(os.path.join("results", "a2c_results.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=a2c_results[0].keys())
                writer.writeheader()
                writer.writerows(a2c_results)
            print(f"\nA2C results saved to results/a2c_results.csv")

    print(f"\nTotal experiments completed: {len(all_results)}")


if __name__ == "__main__":
    main()
