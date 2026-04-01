"""
Main entry point — Loads the best performing model and runs it in the
Cyber Threat Hunt environment with optional Unity visualization.

Usage:
    python main.py                        # Run best model (auto-detect)
    python main.py --algorithm ppo        # Run best PPO model
    python main.py --random               # Random agent (for Unity demo)
    python main.py --render unity         # With Unity visualization
    python main.py --render human         # Text-based rendering
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch

from environment.custom_env import CyberThreatHuntEnv


def find_best_model(algorithm=None):
    """Find the best performing saved model across algorithms.

    Uses results CSVs to pick the actual best config per algorithm,
    falling back to first alphabetical match if CSVs are missing.
    """
    import csv

    search_paths = {
        "dqn": "models/dqn/*/best_model.zip",
        "ppo": "models/pg/ppo/*/best_model.zip",
        "a2c": "models/pg/a2c/*/best_model.zip",
        "reinforce": "models/pg/reinforce/*/best_model.pt",
    }

    # Load best config names from results CSVs
    best_configs = {}
    csv_map = {
        "dqn": "results/dqn_results.csv",
        "ppo": "results/ppo_results.csv",
        "a2c": "results/a2c_results.csv",
        "reinforce": "results/reinforce_results.csv",
    }
    for algo, csv_path in csv_map.items():
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                best_reward = -float('inf')
                best_name = None
                for row in reader:
                    reward = float(row['mean_reward'])
                    if reward > best_reward:
                        best_reward = reward
                        best_name = row['name']
                if best_name:
                    best_configs[algo] = best_name

    if algorithm:
        paths = {algorithm: search_paths.get(algorithm, "")}
    else:
        paths = search_paths

    found = {}
    for algo, pattern in paths.items():
        matches = glob.glob(pattern)
        if matches:
            # Prefer the best config from CSV results
            if algo in best_configs:
                best_name = best_configs[algo]
                for m in matches:
                    if best_name in m:
                        found[algo] = m
                        break
            # Fallback to first match
            if algo not in found:
                found[algo] = matches[0]

    if not found:
        # Try final models
        for algo in (paths.keys()):
            if algo == "reinforce":
                matches = glob.glob("models/pg/reinforce/*/final_model.pt")
            else:
                base = "models/dqn" if algo == "dqn" else f"models/pg/{algo}"
                matches = glob.glob(f"{base}/*/final_model.zip")
            if matches:
                found[algo] = matches[0]

    return found


def load_sb3_model(algo_name, model_path):
    """Load a Stable Baselines 3 model."""
    if algo_name == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo_name == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algo_name == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(model_path)
    else:
        raise ValueError(f"Unknown SB3 algorithm: {algo_name}")


def load_reinforce_model(model_path, env):
    """Load a REINFORCE model."""
    from training.reinforce_training import REINFORCEAgent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(obs_dim, action_dim)
    agent.load(model_path)
    return agent


def run_episode(env, model=None, algo_name=None, deterministic=True, delay=0.5):
    """Run a single episode, optionally rendering."""
    import time

    obs, info = env.reset()
    total_reward = 0
    step = 0
    done = False

    action_names = ["MOVE", "QUICK_SCAN", "DEEP_ANALYZE",
                    "SET_HONEYPOT", "QUARANTINE", "CHECK_LOGS"]

    print(f"\n{'='*50}")
    print("CYBER THREAT HUNT — Episode Start")
    print(f"{'='*50}")

    while not done:
        if model is None:
            # Random agent
            action = env.action_space.sample()
        elif algo_name == "reinforce":
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            dist = model.policy(obs_tensor)
            if deterministic:
                action = dist.probs.argmax().item()
            else:
                action = dist.sample().item()
        else:
            # SB3 model
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

        # Smart overrides: the model never learned QUARANTINE, so help it
        node = env.agent_pos
        if (node in env.discovered_threats
                and node not in env.neutralized_threats
                and env.node_quarantined[node] == 0.0):
            action = 4  # QUARANTINE confirmed threat
        elif (env.node_quarantined[node] > 0.5
                or node in env.neutralized_threats):
            action = 0  # MOVE away from handled nodes
        elif (action == 2
                and env.node_scanned_deep[node] == 1.0
                and env.node_threat_indicator[node] < 0.3):
            action = 0  # Don't re-scan clean nodes

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        env.render()

        if delay > 0:
            time.sleep(delay)

    outcome = info.get("outcome", "unknown")
    print(f"\n{'='*50}")
    print(f"Episode Complete — {outcome.upper()}")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Steps: {step}")
    print(f"Threats Neutralized: {info.get('threats_neutralized', 0)}/{info.get('total_threats', 0)}")
    print(f"Total Damage: {info.get('total_damage', 0):.1f}")
    print(f"{'='*50}")

    return total_reward, outcome


def main():
    parser = argparse.ArgumentParser(description="Run Cyber Threat Hunt Agent")
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["dqn", "ppo", "a2c", "reinforce"],
                        help="Which algorithm's model to load")
    parser.add_argument("--random", action="store_true",
                        help="Use random actions (for demo)")
    parser.add_argument("--render", type=str, default="human",
                        choices=["human", "unity", "none"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between steps (seconds)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    render_mode = args.render if args.render != "none" else None
    # Unity demo uses fewer threats + more steps so episodes last long enough
    # for the 3D visualization to render agent movement through corridors
    if args.render == "unity":
        num_threats = 2
        max_steps = 200
    else:
        num_threats = 4
        max_steps = 100
    env = CyberThreatHuntEnv(
        render_mode=render_mode,
        num_nodes=14,
        num_threats=num_threats,
        max_steps=max_steps,
    )

    model = None
    algo_name = None

    if not args.random:
        found_models = find_best_model(args.algorithm)
        if not found_models:
            print("No trained models found. Running with random actions.")
            print("Train models first with:")
            print("  python training/dqn_training.py")
            print("  python training/pg_training.py")
            print("  python training/reinforce_training.py")
            args.random = True
        else:
            # Pick specified algorithm, or best overall (PPO > DQN > A2C > REINFORCE)
            if args.algorithm:
                algo_name = args.algorithm
            else:
                for pref in ["ppo", "dqn", "a2c", "reinforce"]:
                    if pref in found_models:
                        algo_name = pref
                        break
            model_path = found_models[algo_name]
            print(f"Loading {algo_name.upper()} model from {model_path}")

            if algo_name == "reinforce":
                agent = load_reinforce_model(model_path, env)
                model = agent
            else:
                model = load_sb3_model(algo_name, model_path)

    mode_str = "RANDOM" if args.random else algo_name.upper()
    print(f"\nRunning {args.episodes} episode(s) with {mode_str} agent")
    print(f"Render mode: {args.render}")

    # For Unity: start server and wait for connection BEFORE running episodes
    if args.render == "unity":
        import time
        from environment.rendering import wait_for_connection
        wait_for_connection(timeout=120)
        print("[Unity] Waiting 3s for Unity to initialize...")
        time.sleep(3)

    # Use longer delay for Unity so agent movement and actions are visible
    delay = args.delay
    if args.render == "unity" and delay <= 1.0:
        delay = 3.0  # enough time for agent to walk through corridors visibly

    results = []
    for ep in range(args.episodes):
        if args.seed is not None:
            env.reset(seed=args.seed + ep)
        reward, outcome = run_episode(
            env, model, algo_name,
            deterministic=True,
            delay=delay,
        )
        results.append((reward, outcome))

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    rewards = [r[0] for r in results]
    print(f"Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    outcomes = [r[1] for r in results]
    for outcome in set(outcomes):
        count = outcomes.count(outcome)
        print(f"  {outcome}: {count}/{len(outcomes)}")

    env.close()


if __name__ == "__main__":
    main()
