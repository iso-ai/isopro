"""AI Agent for Simulation Environment."""
from ..base.base_component import BaseComponent, agent_component

@agent_component
class AI_Agent(BaseComponent):
    """AI Agent for Simulation Environment."""

    def __init__(self, name):
        """
        Initialize the AI_Agent.

        Args:
            name (str): The name of the agent.
        """
        super().__init__(name)
        self.components = []

    def add_component(self, component):
        """
        Add a component to the agent.

        Args:
            component (BaseComponent): The component to add.
        """
        if getattr(component, '_is_agent_component', False):
            self.components.append(component)
        else:
            raise ValueError(f"Component {component} is not decorated with @agent_component")

    def run(self, input_data):
        """
        Run the agent's components and process input data.

        Args:
            input_data (dict): The input data for the agent.

        Returns:
            dict: The processed output data.
        """
        self.logger.info(f"Running agent: {self.name}")
        output = input_data
        for component in self.components:
            output = component.run(output)
        return output