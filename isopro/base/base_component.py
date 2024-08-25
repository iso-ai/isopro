"""Base Component for Simulation Environment."""
from abc import ABC, abstractmethod
from ..utils.logging_utils import setup_logger

class BaseComponent(ABC):
    """Base Component for Simulation Environment."""

    def __init__(self, name):
        """
        Initialize the BaseComponent.

        Args:
            name (str): The name of the component.
        """
        self.name = name
        self.logger = setup_logger(f"{self.__class__.__name__}_{self.name}")

    @abstractmethod
    def run(self):
        """Execute the component's main functionality."""
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

def agent_component(cls):
    """
    Decorator to mark a class as an agent component.

    This decorator can be used to add metadata or perform
    additional setup for agent components.
    """
    cls._is_agent_component = True
    return cls