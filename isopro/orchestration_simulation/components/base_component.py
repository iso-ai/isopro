class BaseComponent:
    def __init__(self, name, priority=0):
        self.name = name
        self.priority = priority

    def run(self, input_data):
        raise NotImplementedError("Subclasses must implement the 'run' method")
