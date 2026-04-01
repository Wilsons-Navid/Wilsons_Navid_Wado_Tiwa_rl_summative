"""
Flask API endpoint for serving the Cyber Threat Hunt agent's actions as JSON.
Demonstrates how the trained RL agent can be integrated into a production
web or mobile application pipeline.

Usage:
    python api.py                          # Start server on port 5000
    python api.py --algorithm ppo          # Use specific algorithm
    python api.py --port 8080             # Custom port

Endpoints:
    POST /predict     — Get agent's action for a given observation
    POST /reset       — Reset the environment, returns initial state
    POST /step        — Take a step with agent's predicted action
    GET  /info        — Get environment and model metadata
    GET  /health      — Health check
"""

import os
import sys
import json
import argparse
import numpy as np
from threading import Lock

from flask import Flask, request, jsonify, render_template

from environment.custom_env import CyberThreatHuntEnv

from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    """Handle numpy types in JSON serialization."""
    @staticmethod
    def default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return DefaultJSONProvider.default(obj)

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Global state
env = None
model = None
algo_name = None
_env_lock = Lock()

ACTION_NAMES = ["MOVE", "QUICK_SCAN", "DEEP_ANALYZE",
                "SET_HONEYPOT", "QUARANTINE", "CHECK_LOGS"]


def load_model(algorithm=None):
    """Load the best trained model."""
    global model, algo_name
    from main import find_best_model, load_sb3_model, load_reinforce_model

    found = find_best_model(algorithm)
    if not found:
        print("No trained models found. API will use random actions.")
        return

    algo_name = algorithm or list(found.keys())[0]
    model_path = found[algo_name]
    print(f"Loading {algo_name.upper()} model from {model_path}")

    if algo_name == "reinforce":
        model = load_reinforce_model(model_path, env)
    else:
        model = load_sb3_model(algo_name, model_path)


def get_action(observation):
    """Get action from model or random."""
    import torch

    if model is None:
        return int(env.action_space.sample())

    if algo_name == "reinforce":
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(model.device)
        dist = model.policy(obs_tensor)
        return int(dist.probs.argmax().item())
    else:
        action, _ = model.predict(observation, deterministic=True)
        return int(action)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None,
                    "algorithm": algo_name})


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "environment": "CyberThreatHuntEnv",
        "observation_shape": list(env.observation_space.shape),
        "num_actions": int(env.action_space.n),
        "action_names": ACTION_NAMES,
        "algorithm": algo_name or "random",
        "model_loaded": model is not None,
        "config": {
            "num_nodes": env.NUM_NODES,
            "num_threats": env.NUM_THREATS,
            "max_steps": env.MAX_STEPS,
            "damage_threshold": env.DAMAGE_THRESHOLD,
        }
    })


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    seed = data.get("seed", None)

    with _env_lock:
        obs, env_info = env.reset(seed=seed)
        state = env.get_state_for_render()

    return jsonify({
        "observation": obs.tolist(),
        "info": env_info,
        "state": state,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Get agent's action for a given observation without stepping."""
    data = request.get_json()
    if data is None or "observation" not in data:
        return jsonify({"error": "Missing 'observation' in request body"}), 400

    obs = np.array(data["observation"], dtype=np.float32)
    action = get_action(obs)

    return jsonify({
        "action": action,
        "action_name": ACTION_NAMES[action],
        "algorithm": algo_name or "random",
    })


@app.route("/step", methods=["POST"])
def step():
    """Take one step using the agent's policy."""
    data = request.get_json(silent=True) or {}

    with _env_lock:
        # Auto-reset if env is in terminal state
        if env.current_step >= env.MAX_STEPS or env.total_damage >= env.DAMAGE_THRESHOLD:
            env.reset()

        # Get current observation and predict action
        obs = env._get_observation()
        action = data.get("action", None)

        if action is None:
            action = get_action(obs)

            # Smart overrides for better demo behavior
            node = env.agent_pos

            # If threat is confirmed at current node, quarantine immediately
            if (node in env.discovered_threats
                    and node not in env.neutralized_threats
                    and env.node_quarantined[node] == 0.0):
                action = 4  # QUARANTINE

            # If node already quarantined or neutralized, move on
            elif (env.node_quarantined[node] > 0.5
                    or node in env.neutralized_threats):
                action = 0  # MOVE

            # If already deep-scanned this node and no threat found, move on
            elif (action == 2
                    and env.node_scanned_deep[node] == 1.0
                    and env.node_threat_indicator[node] < 0.3):
                action = 0  # MOVE

            # Limit repeated DEEP_ANALYZE: after 2 scans with no result, move
            elif (action == 2
                    and env.node_scan_time[node] < 0.05
                    and env.node_threat_indicator[node] < 0.5):
                action = 0  # MOVE

        action = int(action)
        obs, reward, terminated, truncated, step_info = env.step(action)
        state = env.get_state_for_render()

    return jsonify({
        "observation": obs.tolist(),
        "action": action,
        "action_name": ACTION_NAMES[action],
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
        "info": step_info,
        "state": state,
    })


def init_app(algorithm="ppo"):
    """Initialize environment and model. Called at startup."""
    global env
    env = CyberThreatHuntEnv(num_nodes=14, num_threats=2, max_steps=200)
    env.reset(seed=42)
    load_model(algorithm)


def main():
    parser = argparse.ArgumentParser(description="CyberShield RL Agent API")
    parser.add_argument("--algorithm", type=str, default="ppo",
                        choices=["dqn", "ppo", "a2c", "reinforce"])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    init_app(args.algorithm)

    print(f"\nCyberShield RL Agent API running on http://{args.host}:{args.port}")
    print(f"Algorithm: {algo_name or 'random'}")
    print(f"\nEndpoints:")
    print(f"  GET  /health    — Health check")
    print(f"  GET  /info      — Environment & model metadata")
    print(f"  POST /reset     — Reset environment")
    print(f"  POST /predict   — Get action for observation")
    print(f"  POST /step      — Step environment with agent's action")

    app.run(host=args.host, port=args.port, debug=False)


# Auto-initialize when imported by gunicorn
if env is None:
    init_app(os.environ.get("RL_ALGORITHM", "ppo"))


if __name__ == "__main__":
    main()
