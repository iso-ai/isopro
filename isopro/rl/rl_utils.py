"""
Reinforcement Learning Utilities

This module provides utility functions for Reinforcement Learning tasks,
including reward calculation, Q-table updates, action selection, and more.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def update_q_table(q_table: np.ndarray, state: int, action: int, reward: float, next_state: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Update a Q-table using the Q-learning algorithm.

    Args:
        q_table (np.ndarray): The current Q-table.
        state (int): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (int): The next state.
        alpha (float): The learning rate.
        gamma (float): The discount factor.

    Returns:
        np.ndarray: The updated Q-table.
    """
    current_q = q_table[state, action]
    max_future_q = np.max(q_table[next_state, :])
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
    q_table[state, action] = new_q
    return q_table

def epsilon_greedy_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    """
    Select an action using an epsilon-greedy policy.

    Args:
        q_table (np.ndarray): The current Q-table.
        state (int): The current state.
        epsilon (float): The exploration rate.

    Returns:
        int: The selected action.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    else:
        return np.argmax(q_table[state, :])

def calculate_discounted_rewards(rewards: List[float], gamma: float) -> np.ndarray:
    """
    Calculate discounted rewards for a list of rewards.

    Args:
        rewards (List[float]): The list of rewards.
        gamma (float): The discount factor.

    Returns:
        np.ndarray: The array of discounted rewards.
    """
    discounted_rewards = np.zeros_like(rewards, dtype=float)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[t]
        discounted_rewards[t] = running_sum
    return discounted_rewards

def normalize_rewards(rewards: np.ndarray) -> np.ndarray:
    """
    Normalize rewards to have zero mean and unit variance.

    Args:
        rewards (np.ndarray): The array of rewards.

    Returns:
        np.ndarray: The normalized rewards.
    """
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

def create_epsilon_decay_schedule(start_epsilon: float, end_epsilon: float, decay_steps: int) -> callable:
    """
    Create an epsilon decay schedule function.

    Args:
        start_epsilon (float): The starting epsilon value.
        end_epsilon (float): The final epsilon value.
        decay_steps (int): The number of steps over which to decay epsilon.

    Returns:
        callable: A function that takes the current step and returns the current epsilon value.
    """
    def epsilon_decay(step: int) -> float:
        return max(end_epsilon, start_epsilon - (start_epsilon - end_epsilon) * (step / decay_steps))
    return epsilon_decay

class ExperienceReplayBuffer:
    """A simple experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int):
        """
        Initialize the experience replay buffer.

        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer: List[Tuple[int, int, float, int, bool]] = []
        self.position = 0

    def push(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Add a transition to the buffer.

        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The next state.
            done (bool): Whether the episode is done.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[int, int, float, int, bool]]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            List[Tuple[int, int, float, int, bool]]: A list of sampled transitions.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

def soft_update(target_network: Any, source_network: Any, tau: float):
    """
    Perform a soft update of the target network parameters.

    Args:
        target_network (Any): The target network to be updated.
        source_network (Any): The source network to update from.
        tau (float): The soft update coefficient (0 < tau < 1).
    """
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def huber_loss(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Compute the Huber loss.

    Args:
        x (np.ndarray): The input array.
        delta (float): The Huber loss parameter.

    Returns:
        np.ndarray: The Huber loss values.
    """
    return np.where(np.abs(x) < delta, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))

logger.info("RL utilities module loaded")