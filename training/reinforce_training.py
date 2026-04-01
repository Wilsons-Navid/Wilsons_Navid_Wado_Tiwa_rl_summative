"""
REINFORCE (vanilla policy gradient) implementation from scratch using PyTorch.
SB3 does not include vanilla REINFORCE, so this is a custom implementation
that works with the same Gymnasium environment interface.
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import CyberThreatHuntEnv


class PolicyNetwork(nn.Module):
    """Simple MLP policy network for REINFORCE."""

    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x)
        return Categorical(logits=logits)

    def get_action(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class REINFORCEAgent:
    """Vanilla REINFORCE with optional baseline (mean return)."""

    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99,
                 hidden_sizes=(128, 128), use_baseline=True):
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode storage
        self.log_probs = []
        self.rewards = []

    def select_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob = self.policy.get_action(obs_tensor)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        """Compute discounted returns and update policy."""
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Baseline subtraction (mean return)
        if self.use_baseline and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.log_probs = []
        self.rewards = []

        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def evaluate_agent(agent, env, n_episodes=20):
    """Evaluate REINFORCE agent deterministically."""
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            dist = agent.policy(obs_tensor)
            action = dist.probs.argmax().item()  # deterministic
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def train_reinforce(config, total_episodes=500, eval_interval=50, seed=42):
    """Train REINFORCE with given config."""
    print(f"\n{'='*60}")
    print(f"REINFORCE Experiment: {config['name']}")
    print(f"lr={config['lr']}, gamma={config['gamma']}, "
          f"arch={config['hidden_sizes']}, baseline={config['use_baseline']}")
    print(f"{'='*60}")

    env = CyberThreatHuntEnv(num_nodes=14, num_threats=4, max_steps=100)
    eval_env = CyberThreatHuntEnv(num_nodes=14, num_threats=4, max_steps=100)

    obs, _ = env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=config["lr"],
        gamma=config["gamma"],
        hidden_sizes=config["hidden_sizes"],
        use_baseline=config["use_baseline"],
    )

    model_dir = os.path.join("models", "pg", "reinforce", config["name"])
    os.makedirs(model_dir, exist_ok=True)

    best_mean_reward = -float("inf")
    episode_rewards = []
    start = time.time()

    for episode in range(total_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            episode_reward += reward
            done = terminated or truncated

        loss = agent.update()
        episode_rewards.append(episode_reward)

        if (episode + 1) % eval_interval == 0:
            mean_r, std_r = evaluate_agent(agent, eval_env, n_episodes=10)
            avg_train = np.mean(episode_rewards[-eval_interval:])
            print(f"Episode {episode+1}/{total_episodes} | "
                  f"Train avg: {avg_train:.1f} | Eval: {mean_r:.1f} +/- {std_r:.1f}")

            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                agent.save(os.path.join(model_dir, "best_model.pt"))

    train_time = time.time() - start

    # Final save
    agent.save(os.path.join(model_dir, "final_model.pt"))

    # Final evaluation
    mean_reward, std_reward = evaluate_agent(agent, eval_env, n_episodes=20)
    print(f"\nFinal: {mean_reward:.2f} +/- {std_reward:.2f} ({train_time:.1f}s)")
    print(f"Best eval reward: {best_mean_reward:.2f}")

    env.close()
    eval_env.close()

    return {
        "experiment": config["name"],
        "algorithm": "REINFORCE",
        "lr": config["lr"],
        "gamma": config["gamma"],
        "hidden_sizes": str(config["hidden_sizes"]),
        "use_baseline": config["use_baseline"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "best_eval_reward": best_mean_reward,
        "training_time_s": train_time,
    }


# 12 hyperparameter combinations
EXPERIMENTS = [
    {"name": "reinforce_baseline", "lr": 1e-3, "gamma": 0.99,
     "hidden_sizes": (128, 128), "use_baseline": True},

    {"name": "reinforce_no_baseline", "lr": 1e-3, "gamma": 0.99,
     "hidden_sizes": (128, 128), "use_baseline": False},

    {"name": "reinforce_high_lr", "lr": 5e-3, "gamma": 0.99,
     "hidden_sizes": (128, 128), "use_baseline": True},

    {"name": "reinforce_low_lr", "lr": 1e-4, "gamma": 0.99,
     "hidden_sizes": (128, 128), "use_baseline": True},

    {"name": "reinforce_low_gamma", "lr": 1e-3, "gamma": 0.9,
     "hidden_sizes": (128, 128), "use_baseline": True},

    {"name": "reinforce_high_gamma", "lr": 1e-3, "gamma": 0.999,
     "hidden_sizes": (128, 128), "use_baseline": True},

    {"name": "reinforce_small_net", "lr": 1e-3, "gamma": 0.99,
     "hidden_sizes": (64, 64), "use_baseline": True},

    {"name": "reinforce_deep_net", "lr": 1e-3, "gamma": 0.99,
     "hidden_sizes": (256, 256, 128), "use_baseline": True},

    {"name": "reinforce_wide_net", "lr": 1e-3, "gamma": 0.99,
     "hidden_sizes": (512, 256), "use_baseline": True},

    {"name": "reinforce_tiny_net", "lr": 1e-3, "gamma": 0.99,
     "hidden_sizes": (32, 32), "use_baseline": True},

    {"name": "reinforce_med_lr_no_bl", "lr": 5e-4, "gamma": 0.99,
     "hidden_sizes": (128, 128), "use_baseline": False},

    {"name": "reinforce_tuned", "lr": 3e-4, "gamma": 0.995,
     "hidden_sizes": (256, 128), "use_baseline": True},
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="REINFORCE Hyperparameter Sweep")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.experiment:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.experiment]
        if not experiments:
            print(f"Experiment '{args.experiment}' not found.")
            return

    results = []
    for config in experiments:
        result = train_reinforce(config, args.episodes, seed=args.seed)
        results.append(result)

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "reinforce_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAll REINFORCE results saved to results/reinforce_results.csv")


if __name__ == "__main__":
    main()
