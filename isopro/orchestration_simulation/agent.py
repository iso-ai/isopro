from .exceptions import AI_AgentException, ComponentException

class AI_Agent:
    def __init__(self, name):
        self.name = name
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def run(self, agent_input):
        print(f"Running agent: {self.name}")
        agent_output = {}
        for component in self.components:
            try:
                component_output = component.run(agent_input)
                agent_output.update(component_output)
            except ComponentException as e:
                raise AI_AgentException(f"Exception in agent {self.name}: {e}")
        return agent_output

    def reset(self):
        print(f"Resetting agent: {self.name}")
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()

    def close(self):
        print(f"Closing agent: {self.name}")
        for component in self.components:
            if hasattr(component, 'close'):
                component.close()