"""
This module contains the base and specific implementations of RL environments.
It includes a base class for RL environments and two subclasses:
one for LLM-based environments and another for traditional gym environments.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import anthropic
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class BaseRLEnvironment(gym.Env):
    """
    Base class for RL environments in the isopro package.
    """

    def __init__(self):
        """Initialize the base RL environment."""
        super().__init__()
        self.action_space = None
        self.observation_space = None
        logger.info("Initialized BaseRLEnvironment")

    def reset(self):
        """Reset the environment to its initial state."""
        raise NotImplementedError("Subclasses must implement reset method")

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the next observation, reward, done flag, and info dictionary.
        """
        raise NotImplementedError("Subclasses must implement step method")

class LLMRLEnvironment(BaseRLEnvironment):
    """
    RL environment that uses a Language Model for interactions.
    """

    def __init__(self, agent_prompt, ai_agent):
        """
        Initialize the LLM-based RL environment.

        Args:
            agent_prompt (str): The prompt to guide the AI agent's behavior.
            ai_agent: The AI agent to interact with.
        """
        super().__init__()
        self.agent_prompt = agent_prompt
        self.ai_agent = ai_agent
        self.action_space = spaces.Discrete(5)  # Define the action space
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.current_step = 0
        self.conversation_history = []
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info("Initialized LLMRLEnvironment")

    def reset(self):
        """
        Reset the LLM environment to its initial state.

        Returns:
            numpy.ndarray: The initial observation.
        """
        self.current_step = 0
        self.conversation_history = []
        logger.info("Reset LLMRLEnvironment")
        return np.random.random(10)  # Initial observation

    def step(self, action):
        """
        Take a step in the LLM environment.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.current_step += 1
        done = self.current_step >= 10

        # Prepare the message for the AI model
        messages = [
            {"role": "system", "content": self.agent_prompt},
            {"role": "user", "content": f"Action: {action}"},
        ] + self.conversation_history

        # Get response from the AI model
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=messages
        )

        ai_response = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        reward = self.calculate_reward(ai_response)
        observation = self.update_observation(ai_response)

        logger.info(f"Step {self.current_step}: Action={action}, Reward={reward}, Done={done}")
        return observation, reward, done, {}

    def calculate_reward(self, response):
        """
        Calculate the reward based on the AI's response.

        Args:
            response (str): The AI's response.

        Returns:
            float: The calculated reward.
        """
        adherence_score = self.evaluate_persona_adherence(response)
        human_feedback = self.get_human_feedback()
        total_reward = adherence_score + human_feedback
        logger.debug(f"Calculated reward: {total_reward}")
        return total_reward

    def evaluate_persona_adherence(self, response):
        """
        Evaluate how well the AI's response adheres to the given persona.

        Args:
            response (str): The AI's response.

        Returns:
            float: The adherence score.
        """
        evaluation = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[
                {"role": "system", "content": "Evaluate how well the following response adheres to the given persona. Return a score between 0 and 1."},
                {"role": "user", "content": f"Persona: {self.agent_prompt}\nResponse: {response}"}
            ]
        )
        adherence_score = float(evaluation.content[0].text)
        logger.debug(f"Persona adherence score: {adherence_score}")
        return adherence_score

    def get_human_feedback(self):
        """
        Simulate human feedback.

        Returns:
            float: A random value between -0.5 and 0.5 to simulate human feedback.
        """
        feedback = np.random.uniform(-0.5, 0.5)
        logger.debug(f"Simulated human feedback: {feedback}")
        return feedback

    def update_observation(self, response):
        """
        Update the observation based on the AI's response.

        Args:
            response (str): The AI's response.

        Returns:
            numpy.ndarray: The updated observation.
        """
        # TODO: Implement actual feature extraction from the response
        observation = np.random.random(10)
        logger.debug(f"Updated observation: {observation}")
        return observation

class GymRLEnvironment(BaseRLEnvironment):
    """
    Wrapper for standard gym environments to be used in the isopro framework.
    """

    def __init__(self, env_name):
        """
        Initialize the gym environment wrapper.

        Args:
            env_name (str): The name of the gym environment to create.
        """
        super().__init__()
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        logger.info(f"Initialized GymRLEnvironment with {env_name}")

    def reset(self):
        """
        Reset the gym environment.

        Returns:
            The initial observation from the gym environment.
        """
        logger.info("Reset GymRLEnvironment")
        return self.env.reset()

    def step(self, action):
        """
        Take a step in the gym environment.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: (observation, reward, done, info) as returned by the gym environment.
        """
        result = self.env.step(action)
        logger.debug(f"GymRLEnvironment step: action={action}, result={result}")
        return result

    def render(self, mode='human'):
        """
        Render the gym environment.

        Args:
            mode (str): The mode to render the environment in.

        Returns:
            The rendering of the environment.
        """
        return self.env.render(mode)

    def close(self):
        """Close the gym environment."""
        logger.info("Closing GymRLEnvironment")
        return self.env.close()