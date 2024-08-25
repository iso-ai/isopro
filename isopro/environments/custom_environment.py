"""Custom Environment for creating user-defined simulation environments."""
from ..environments.simulation_environment import SimulationEnvironment
from ..agents.ai_agent import AI_Agent
from ..base.base_component import BaseComponent, agent_component

class CustomAgent(AI_Agent):
    """
    CustomAgent

    This class defines a custom agent. Users can extend this class to implement their own agents.
    """
    def __init__(self, name, custom_param):
        """
        Initialize the CustomAgent.

        Args:
            name (str): The name of the agent.
            custom_param: A custom parameter for the agent.
        """
        super().__init__(name)
        self.custom_param = custom_param

    def run(self, input_data):
        """
        Run the custom agent.

        Args:
            input_data (dict): The input data for the agent.

        Returns:
            dict: The processed output data.
        """
        self.logger.info(f"Running custom agent: {self.name} with parameter: {self.custom_param}")
        # Implement custom behavior here
        return super().run(input_data)

@agent_component
class CustomComponent(BaseComponent):
    """
    CustomComponent

    This class defines a custom component. Users can extend this class to implement their own components.
    """
    def __init__(self, name, custom_param):
        """
        Initialize the CustomComponent.

        Args:
            name (str): The name of the component.
            custom_param: A custom parameter for the component.
        """
        super().__init__(name)
        self.custom_param = custom_param

    def run(self, input_data):
        """
        Run the custom component.

        Args:
            input_data (dict): The input data for the component.

        Returns:
            dict: The processed output data.
        """
        self.logger.info(f"Running custom component: {self.name} with parameter: {self.custom_param}")
        # Implement custom behavior here
        return input_data

class CustomEnvironment(SimulationEnvironment):
    """
    CustomEnvironment

    This class provides a template for creating a custom training environment.
    Users can define their own agents and components, and integrate them into the simulation environment.
    """
    def __init__(self, num_agents=1, custom_param=None):
        """
        Initialize the CustomEnvironment.

        Args:
            num_agents (int): The number of agents to create.
            custom_param: A custom parameter for the environment.
        """
        super().__init__()
        self.num_agents = num_agents
        self.custom_param = custom_param
        self._create_custom_agents()

    def _create_custom_agents(self):
        """Create custom agents and add them to the environment."""
        for i in range(self.num_agents):
            agent = CustomAgent(name=f"Custom Agent {i+1}", custom_param=self.custom_param)
            component = CustomComponent(name=f"Custom Component {i+1}", custom_param=self.custom_param)
            agent.add_component(component)
            self.add_agent(agent)

    def add_custom_agent(self, agent_name, custom_param):
        """
        Add a custom agent to the environment.

        Args:
            agent_name (str): The name of the agent.
            custom_param: A custom parameter for the agent.
        """
        agent = CustomAgent(name=agent_name, custom_param=custom_param)
        component = CustomComponent(name=f"Component for {agent_name}", custom_param=custom_param)
        agent.add_component(component)
        self.add_agent(agent)