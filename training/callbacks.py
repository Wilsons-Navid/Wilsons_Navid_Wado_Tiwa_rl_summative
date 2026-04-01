"""
Custom training callbacks for logging episode rewards and metrics.
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """Logs episode rewards and losses during training for plotting."""

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_log = []
        self.losses = []
        self.loss_timesteps = []
        self.entropies = []
        self.entropy_timesteps = []

        self._current_episode_reward = 0.0
        self._current_episode_length = 0

    def _on_step(self) -> bool:
        # Track episode reward
        self._current_episode_reward += self.locals.get("rewards", [0])[0]
        self._current_episode_length += 1

        # Check if episode ended
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self._current_episode_length)
            self.timesteps_log.append(self.num_timesteps)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

        # Extract loss from logger (SB3 logs these internally)
        if hasattr(self.model, "logger") and self.model.logger is not None:
            name_to_value = getattr(self.model.logger, "name_to_value", {})

            # DQN loss
            if "train/loss" in name_to_value:
                self.losses.append(name_to_value["train/loss"])
                self.loss_timesteps.append(self.num_timesteps)

            # Policy gradient entropy
            if "train/entropy_loss" in name_to_value:
                self.entropies.append(name_to_value["train/entropy_loss"])
                self.entropy_timesteps.append(self.num_timesteps)

        return True

    def _on_training_end(self):
        """Save all logged metrics."""
        np.savez(
            os.path.join(self.log_path, "training_metrics.npz"),
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            timesteps=np.array(self.timesteps_log),
            losses=np.array(self.losses),
            loss_timesteps=np.array(self.loss_timesteps),
            entropies=np.array(self.entropies),
            entropy_timesteps=np.array(self.entropy_timesteps),
        )
