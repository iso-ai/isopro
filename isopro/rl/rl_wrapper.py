"""
This module contains the RLEnvironmentWrapper class, which provides a wrapper
for creating and managing RL environments and agents within the isopro framework.
"""

from ..environments.simulation_environment import SimulationEnvironment
from .rl_environment import LLMRLEnvironment, GymRLEnvironment
from .rl_agent import RLAgent
from ..agents.ai_agent import AI_Agent
import logging

# Set up logging
logger = logging.getLogger(__name__)

class RLEnvironmentWrapper(SimulationEnvironment):
    """
    A wrapper class for creating and managing RL environments and agents.
    """

    def __init__(self, env_type='llm', num_agents=1, agent_prompt=None, gym_env_name=None):
        """
        Initialize the RLEnvironmentWrapper.

        Args:
            env_type (str): The type of environment to create ('llm' or 'gym').
            num_agents (int): The number of agents to create.
            agent_prompt (str, optional): The prompt to use for LLM-based environments.
            gym_env_name (str, optional): The name of the gym environment to create.
        """
        super().__init__()
        self.num_agents = num_agents
        self.agent_prompt = agent_prompt
        self.gym_env_name = gym_env_name
        self.env_type = env_type
        self._create_rl_agents()
        logger.info(f"Initialized RLEnvironmentWrapper with {env_type} environment and {num_agents} agents")

    def _create_rl_agents(self):
        """
        Create RL agents based on the specified environment type and number of agents.
        """
        for i in range(self.num_agents):
            if self.