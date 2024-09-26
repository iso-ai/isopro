# components/subagent.py

from .base_component import BaseComponent
from ..exceptions import ComponentException
import logging

logger = logging.getLogger(__name__)

class SubAgent(BaseComponent):
    def __init__(self, name, behavior, priority=0):
        super().__init__(name, priority)
        self.behavior = behavior

    def run(self, input_data=None):
        try:
            logger.info(f"Running subagent: {self.name}")
            result = self.behavior(input_data)
            if not result:
                raise ValueError("Empty result from subagent")
            return result
        except Exception as e:
            logger.error(f"Exception in subagent {self.name}: {e}")
            raise ComponentException(f"Exception in subagent {self.name}: {e}")