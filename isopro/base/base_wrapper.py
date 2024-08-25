"""Base Wrapper for Simulation Environment."""
from abc import ABC, abstractmethod
import logging
from ..utils.logging_utils import setup_logger

class BaseWrapper(ABC):
    """Base Wrapper for Simulation Environment."""

    def __init__(self, agent):
        """
        Initialize the BaseWrapper.

        Args:
            agent: The agent to be wrapped.
        """
        self.agent = agent
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def step(self):
        """Execute one time step within the environment."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the state of the environment to an initial state."""
        pass

    @abstractmethod
    def render(self):
        """Render the environment."""
        pass

    @abstractmethod
    def close(self):
        """Close the environment, clean up any resources."""
        pass

    @abstractmethod
    def convert_to_agent_input(self, sim_state):
        """
        Convert simulation state to agent input format.

        Args:
            sim_state (dict): The current state of the simulation.

        Returns:
            dict: The converted input for the agent.
        """
        pass

    @abstractmethod
    def convert_from_agent_output(self, agent_output):
        """
        Convert agent output to simulation input format.

        Args:
            agent_output (dict): The output from the agent.

        Returns:
            dict: The converted input for the simulation.
        """
        pass

    def __getattr__(self, name):
        """
        Attempt to get an attribute from the agent if it's not found in the wrapper.

        Args:
            name (str): The name of the attribute.

        Returns:
            The requested attribute.

        Raises:
            AttributeError: If the attribute is not found in the agent or wrapper.
        """
        try:
            return getattr(self.agent, name)
        except AttributeError:
            self.logger.warning(f"Attribute '{name}' not found in agent or wrapper")
            raise