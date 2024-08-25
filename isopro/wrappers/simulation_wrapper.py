"""Simulation Wrapper for integrating agents with the simulation environment."""
from ..base.base_wrapper import BaseWrapper

class SimulationWrapper(BaseWrapper):
    """Simulation Wrapper for integrating agents with the simulation environment."""

    def __init__(self, agent, simulation):
        """
        Initialize the SimulationWrapper.

        Args:
            agent (AI_Agent): The agent to be wrapped.
            simulation (SimulationEnvironment): The simulation environment.
        """
        super().__init__(agent)
        self.simulation = simulation

    def step(self):
        """
        Execute one time step within the environment.

        Returns:
            The result of the simulation step.
        """
        sim_state = self.simulation.get_state()
        agent_input = self.convert_to_agent_input(sim_state)
        agent_output = self.agent.run(agent_input)
        sim_input = self.convert_from_agent_output(agent_output)
        return self.simulation.step(sim_input)

    def convert_to_agent_input(self, sim_state):
        """
        Convert the simulation state to a format the agent can understand.

        Args:
            sim_state (dict): The current state of the simulation.

        Returns:
            dict: A dictionary containing the formatted input for the agent.
        """
        text_data = sim_state.get('text_data', {})
        return {
            'text': {
                'task': text_data.get('reason', ''),
                'step': text_data.get('step', ''),
                'reasoning': text_data.get('reasoning', ''),
                'max_steps': getattr(self.simulation, 'max_steps', None)
            }
        }

    def convert_from_agent_output(self, agent_output):
        """
        Convert agent output to simulation input format.

        Args:
            agent_output (dict): The output from the agent.

        Returns:
            dict: The converted input for the simulation.
        """
        return agent_output

    def reset(self):
        """Reset the simulation environment."""
        self.simulation.reset()

    def render(self):
        """Render the simulation environment."""
        return self.simulation.render()

    def close(self):
        """Close the simulation environment."""
        self.simulation.close()