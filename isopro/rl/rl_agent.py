"""
This module contains the RLAgent class, which implements a flexible
Reinforcement Learning agent capable of using different algorithms.
"""

from stable_baselines3 import PPO, DQN, A2C
from ..agents.ai_agent import AI_Agent
import logging

# Set up logging
logger = logging.getLogger(__name__)

class RLAgent(AI_Agent):
    """
    A flexible Reinforcement Learning agent that can use different algorithms.
    """

    def __init__(self, name, env, algorithm='PPO', policy='MlpPolicy', **kwargs):
        """
        Initialize the RL agent.

        Args:
            name (str): The name of the agent.
            env: The environment the agent will interact with.
            algorithm (str): The RL algorithm to use ('PPO', 'DQN', or 'A2C').
            policy (str): The policy network architecture to use.
            **kwargs: Additional arguments to pass to the RL algorithm.
        """
        super().__init__(name)
        self.env = env
        self.algorithm = algorithm
        self.policy = policy
        self.model = self._create_model(**kwargs)
        logger.info(f"Initialized RLAgent '{name}' with {algorithm} algorithm")

    def _create_model(self, **kwargs):
        """
        Create the RL model based on the specified algorithm.

        Args:
            **kwargs: Additional arguments to pass to the RL algorithm.

        Returns:
            The created RL model.

        Raises:
            ValueError: If an unsupported algorithm is specified.
        """
        if self.algorithm == 'PPO':
            return PPO(self.policy, self.env, verbose=1, **kwargs)
        elif self.algorithm == 'DQN':
            return DQN(self.policy, self.env, verbose=1, **kwargs)
        elif self.algorithm == 'A2C':
            return A2C(self.policy, self.env, verbose=1, **kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(self, total_timesteps=10000):
        """
        Train the RL agent.

        Args:
            total_timesteps (int): The total number of timesteps to train for.
        """
        logger.info(f"Starting training of RLAgent '{self.name}' for {total_timesteps} timesteps")
        self.model.learn(total_timesteps=total_timesteps)
        logger.info(f"Completed training of RLAgent '{self.name}'")

    def run(self, episodes=1):
        """
        Run the trained RL agent for a specified number of episodes.

        Args:
            episodes (int): The number of episodes to run.

        Returns:
            float: The average reward per episode.
        """
        total_reward = 0
        logger.info(f"Running RLAgent '{self.name}' for {episodes} episodes")
        for episode in range(episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            total_reward += episode_reward
            logger.debug(f"Episode {episode + 1} reward: {episode_reward}")
        average_reward = total_reward / episodes
        logger.info(f"RLAgent '{self.name}' average reward over {episodes} episodes: {average_reward}")
        return average_reward