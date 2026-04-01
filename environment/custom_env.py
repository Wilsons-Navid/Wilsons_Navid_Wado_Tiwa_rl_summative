"""
CyberThreatHuntEnv — A custom Gymnasium environment for cyber threat hunting.

An RL agent navigates an enterprise network, scanning nodes and neutralizing
hidden cyber threats (malware, backdoors, cryptominers, data exfiltration)
before they cause critical damage to the organization.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.network_graph import (
    NodeType,
    ThreatType,
    THREAT_PROPERTIES,
    generate_network,
    get_neighbors,
)


class CyberThreatHuntEnv(gym.Env):
    """Cyber Threat Hunting Environment.

    The agent navigates a network graph of enterprise nodes, using various
    cybersecurity tools to detect and neutralize hidden threats.

    Actions:
        0: Move to next adjacent node
        1: Quick Scan — fast, shallow threat detection
        2: Deep Analyze — slow, thorough threat detection
        3: Set Honeypot — lures threats to reveal over time
        4: Quarantine Node — isolates node from network
        5: Check Logs — reveals historical activity patterns

    Observation (per node + global):
        Per node (6 features each):
            - cpu_usage (0-1)
            - memory_usage (0-1)
            - network_anomaly_score (0-1)
            - time_since_last_scan (0-1, normalized)
            - threat_indicator_level (0-1)
            - quarantine_status (0 or 1)
        Global features (5):
            - agent_position (normalized 0-1)
            - time_remaining (0-1)
            - threats_neutralized_ratio (0-1)
            - damage_level (0-1)
            - honeypot_active_count (0-1)
    """

    metadata = {"render_modes": ["human", "unity"], "render_fps": 4}

    # Actions
    ACTION_MOVE = 0
    ACTION_QUICK_SCAN = 1
    ACTION_DEEP_ANALYZE = 2
    ACTION_SET_HONEYPOT = 3
    ACTION_QUARANTINE = 4
    ACTION_CHECK_LOGS = 5
    NUM_ACTIONS = 6

    # Environment parameters
    NUM_NODES = 14
    NUM_THREATS = 4
    MAX_STEPS = 100
    DAMAGE_THRESHOLD = 100.0
    FEATURES_PER_NODE = 6
    GLOBAL_FEATURES = 5

    def __init__(self, render_mode=None, num_nodes=14, num_threats=4, max_steps=100):
        super().__init__()

        self.render_mode = render_mode
        self.NUM_NODES = num_nodes
        self.NUM_THREATS = num_threats
        self.MAX_STEPS = max_steps

        obs_size = self.NUM_NODES * self.FEATURES_PER_NODE + self.GLOBAL_FEATURES
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

        # State variables initialized in reset()
        self.adjacency = None
        self.node_types = None
        self.threats = None
        self.agent_pos = 0
        self.current_step = 0
        self.total_damage = 0.0
        self.threats_neutralized = 0

        # Per-node state
        self.node_cpu = None
        self.node_memory = None
        self.node_anomaly = None
        self.node_scan_time = None
        self.node_threat_indicator = None
        self.node_quarantined = None
        self.node_honeypot = None
        self.node_scanned_deep = None
        self.node_log_checked = None

        # Threat tracking
        self.active_threats = None  # dict: node -> ThreatType
        self.discovered_threats = None  # set of node indices where threat is known
        self.neutralized_threats = None  # set of node indices

        # For rendering
        self._socket = None
        self._last_action = None
        self._last_reward = 0.0
        self._last_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate network
        self.adjacency, self.node_types, self.threats = generate_network(
            num_nodes=self.NUM_NODES,
            num_threats=self.NUM_THREATS,
            seed=seed,
        )

        # Agent starts at firewall (node 0)
        self.agent_pos = 0
        self.current_step = 0
        self.total_damage = 0.0
        self.threats_neutralized = 0
        self._move_index = {}  # tracks which neighbor to move to next per node

        # Initialize node states with realistic baseline values
        rng = self.np_random
        self.node_cpu = rng.uniform(0.1, 0.5, size=self.NUM_NODES).astype(np.float32)
        self.node_memory = rng.uniform(0.2, 0.6, size=self.NUM_NODES).astype(np.float32)
        self.node_anomaly = np.zeros(self.NUM_NODES, dtype=np.float32)
        self.node_scan_time = np.ones(self.NUM_NODES, dtype=np.float32)  # all "long ago"
        self.node_threat_indicator = np.zeros(self.NUM_NODES, dtype=np.float32)
        self.node_quarantined = np.zeros(self.NUM_NODES, dtype=np.float32)
        self.node_honeypot = np.zeros(self.NUM_NODES, dtype=np.float32)
        self.node_scanned_deep = np.zeros(self.NUM_NODES, dtype=np.float32)
        self.node_log_checked = np.zeros(self.NUM_NODES, dtype=np.float32)

        # Infected nodes have elevated metrics
        self.active_threats = dict(self.threats)
        self.discovered_threats = set()
        self.neutralized_threats = set()

        for node, threat in self.active_threats.items():
            props = THREAT_PROPERTIES[threat]
            # Stealthy threats show less anomaly
            self.node_anomaly[node] = rng.uniform(0.1, 0.4) * (1 - props["stealth"])
            # Resource usage increases
            self.node_cpu[node] += rng.uniform(0.1, 0.3)
            self.node_memory[node] += rng.uniform(0.05, 0.2)

        self.node_cpu = np.clip(self.node_cpu, 0, 1)
        self.node_memory = np.clip(self.node_memory, 0, 1)

        self._last_action = None
        self._last_reward = 0.0
        self._last_info = {}

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.current_step += 1
        reward = -1.0  # time pressure penalty
        info = {}

        node = self.agent_pos

        if action == self.ACTION_MOVE:
            neighbors = get_neighbors(self.adjacency, node)
            if neighbors:
                # Cycle through neighbors deterministically
                idx = self._move_index.get(node, 0)
                self.agent_pos = neighbors[idx % len(neighbors)]
                self._move_index[node] = idx + 1

        elif action == self.ACTION_QUICK_SCAN:
            self.node_scan_time[node] = 0.0
            if node in self.active_threats and node not in self.neutralized_threats:
                threat = self.active_threats[node]
                stealth = THREAT_PROPERTIES[threat]["stealth"]
                # Detection probability inversely proportional to stealth
                if self.np_random.random() > stealth:
                    self.discovered_threats.add(node)
                    self.node_threat_indicator[node] = min(
                        self.node_threat_indicator[node] + 0.5, 1.0
                    )
                    reward += 2.0
                    info["scan_result"] = "threat_detected"
                else:
                    # Partial info even on miss
                    self.node_threat_indicator[node] = min(
                        self.node_threat_indicator[node] + 0.1, 1.0
                    )
                    info["scan_result"] = "clean"
            else:
                info["scan_result"] = "clean"

        elif action == self.ACTION_DEEP_ANALYZE:
            self.node_scan_time[node] = 0.0
            self.node_scanned_deep[node] = 1.0
            if node in self.active_threats and node not in self.neutralized_threats:
                threat = self.active_threats[node]
                stealth = THREAT_PROPERTIES[threat]["stealth"]
                # Deep analysis has much higher detection rate
                if self.np_random.random() > stealth * 0.3:
                    self.discovered_threats.add(node)
                    self.node_threat_indicator[node] = 1.0
                    reward += 5.0
                    info["analysis_result"] = "threat_confirmed"
                else:
                    self.node_threat_indicator[node] = min(
                        self.node_threat_indicator[node] + 0.3, 1.0
                    )
                    info["analysis_result"] = "suspicious"
            else:
                # Reduces indicator for clean nodes (confidence)
                self.node_threat_indicator[node] = max(
                    self.node_threat_indicator[node] - 0.3, 0.0
                )
                info["analysis_result"] = "verified_clean"

        elif action == self.ACTION_SET_HONEYPOT:
            if self.node_honeypot[node] == 0.0:
                self.node_honeypot[node] = 1.0
                # Honeypots on infected nodes eventually reveal threats
                if node in self.active_threats and node not in self.neutralized_threats:
                    reward += 1.0
                info["honeypot"] = "deployed"
            else:
                info["honeypot"] = "already_active"

        elif action == self.ACTION_QUARANTINE:
            if self.node_quarantined[node] == 0.0:
                self.node_quarantined[node] = 1.0
                if node in self.active_threats and node not in self.neutralized_threats:
                    # Correct quarantine — threat neutralized
                    self.neutralized_threats.add(node)
                    self.threats_neutralized += 1
                    reward += 10.0
                    info["quarantine"] = "threat_neutralized"
                else:
                    # False positive — unnecessary downtime
                    reward -= 5.0
                    info["quarantine"] = "false_positive"
            else:
                info["quarantine"] = "already_quarantined"

        elif action == self.ACTION_CHECK_LOGS:
            self.node_log_checked[node] = 1.0
            if node in self.active_threats and node not in self.neutralized_threats:
                threat = self.active_threats[node]
                # Logs reveal patterns — moderate detection for non-stealthy threats
                if self.np_random.random() > THREAT_PROPERTIES[threat]["stealth"] * 0.5:
                    self.node_threat_indicator[node] = min(
                        self.node_threat_indicator[node] + 0.4, 1.0
                    )
                    reward += 1.5
                    info["logs"] = "suspicious_activity"
                else:
                    info["logs"] = "normal"
            else:
                info["logs"] = "normal"

        # --- Threat dynamics (happens every step) ---
        self._evolve_threats()

        # Update scan time decay
        self.node_scan_time = np.clip(self.node_scan_time + 0.02, 0, 1)

        # Check honeypot reveals
        for n in range(self.NUM_NODES):
            if (self.node_honeypot[n] == 1.0
                    and n in self.active_threats
                    and n not in self.neutralized_threats):
                # Honeypots have a chance to reveal threats each step
                if self.np_random.random() < 0.2:
                    self.discovered_threats.add(n)
                    self.node_threat_indicator[n] = min(
                        self.node_threat_indicator[n] + 0.3, 1.0
                    )

        # Terminal conditions
        terminated = False
        truncated = False

        if self.threats_neutralized >= len(self.active_threats):
            reward += 25.0  # all threats cleared bonus
            terminated = True
            info["outcome"] = "all_threats_cleared"

        if self.total_damage >= self.DAMAGE_THRESHOLD:
            terminated = True
            info["outcome"] = "critical_breach"

        if self.current_step >= self.MAX_STEPS:
            truncated = True
            info["outcome"] = "timeout"

        self._last_action = action
        self._last_reward = reward
        self._last_info = info

        obs = self._get_observation()
        info.update(self._get_info())

        return obs, reward, terminated, truncated, info

    def _evolve_threats(self):
        """Simulate threat behavior each timestep."""
        new_infections = {}

        for node, threat in list(self.active_threats.items()):
            if node in self.neutralized_threats:
                continue
            if self.node_quarantined[node] == 1.0:
                continue

            props = THREAT_PROPERTIES[threat]

            # Accumulate damage
            self.total_damage += props["damage_rate"]

            # Threats may increase anomaly signals over time
            self.node_anomaly[node] = np.clip(
                self.node_anomaly[node] + self.np_random.uniform(0, 0.05), 0, 1
            )
            self.node_cpu[node] = np.clip(
                self.node_cpu[node] + self.np_random.uniform(0, 0.02), 0, 1
            )

            # Threat spreading
            if props["spread_rate"] > 0:
                neighbors = get_neighbors(self.adjacency, node)
                for neighbor in neighbors:
                    if (neighbor not in self.active_threats
                            and neighbor not in new_infections
                            and self.node_quarantined[neighbor] == 0.0
                            and self.np_random.random() < props["spread_rate"] * 0.1):
                        new_infections[neighbor] = threat
                        self.node_anomaly[neighbor] += 0.1

        # Apply new infections after iteration
        self.active_threats.update(new_infections)

    def _get_observation(self):
        """Build flattened observation vector."""
        node_features = np.column_stack([
            self.node_cpu,
            self.node_memory,
            self.node_anomaly,
            self.node_scan_time,
            self.node_threat_indicator,
            self.node_quarantined,
        ]).flatten()

        # Global features
        total_threats = max(len(self.active_threats), 1)
        global_features = np.array([
            self.agent_pos / max(self.NUM_NODES - 1, 1),
            1.0 - (self.current_step / self.MAX_STEPS),
            self.threats_neutralized / total_threats,
            min(self.total_damage / self.DAMAGE_THRESHOLD, 1.0),
            np.sum(self.node_honeypot) / self.NUM_NODES,
        ], dtype=np.float32)

        obs = np.concatenate([node_features, global_features]).astype(np.float32)
        return obs

    def _get_info(self):
        """Return auxiliary info dict."""
        return {
            "agent_position": int(self.agent_pos),
            "step": self.current_step,
            "total_damage": float(self.total_damage),
            "threats_neutralized": self.threats_neutralized,
            "total_threats": len(self.active_threats),
            "discovered_threats": len(self.discovered_threats),
        }

    def get_state_for_render(self):
        """Return full state dict for Unity visualization."""
        return {
            "adjacency_flat": self.adjacency.flatten().tolist(),
            "num_nodes": self.NUM_NODES,
            "node_types": self.node_types.tolist(),
            "agent_pos": int(self.agent_pos),
            "node_cpu": self.node_cpu.tolist(),
            "node_memory": self.node_memory.tolist(),
            "node_anomaly": self.node_anomaly.tolist(),
            "node_threat_indicator": self.node_threat_indicator.tolist(),
            "node_quarantined": self.node_quarantined.tolist(),
            "node_honeypot": self.node_honeypot.tolist(),
            "discovered_threats": list(self.discovered_threats),
            "neutralized_threats": list(self.neutralized_threats),
            "total_damage": float(self.total_damage),
            "damage_threshold": float(self.DAMAGE_THRESHOLD),
            "threats_neutralized": self.threats_neutralized,
            "total_threats": len(self.active_threats),
            "current_step": self.current_step,
            "max_steps": self.MAX_STEPS,
            "last_action": self._last_action,
            "last_reward": float(self._last_reward),
        }

    def render(self):
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "unity":
            self._render_unity()

    def _render_text(self):
        """Simple text rendering for debugging."""
        info = self._get_info()
        print(f"\n--- Step {info['step']}/{self.MAX_STEPS} ---")
        print(f"Agent at node {info['agent_position']} "
              f"(type: {NodeType(self.node_types[self.agent_pos]).name})")
        print(f"Damage: {info['total_damage']:.1f}/{self.DAMAGE_THRESHOLD}")
        print(f"Threats: {info['threats_neutralized']}/{info['total_threats']} neutralized, "
              f"{info['discovered_threats']} discovered")
        if self._last_action is not None:
            action_names = ["MOVE", "QUICK_SCAN", "DEEP_ANALYZE",
                            "SET_HONEYPOT", "QUARANTINE", "CHECK_LOGS"]
            print(f"Last action: {action_names[self._last_action]} "
                  f"(reward: {self._last_reward:+.1f})")

    def _render_unity(self):
        """Send state to Unity via TCP socket."""
        from environment.rendering import send_state_to_unity
        state = self.get_state_for_render()
        send_state_to_unity(state)

    def close(self):
        if self._socket:
            self._socket.close()
            self._socket = None
