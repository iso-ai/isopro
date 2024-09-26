"""
Adversarial Agent

This module defines the AdversarialAgent class, which can apply various attacks to input or output text.
"""

from typing import Dict, Any
from isopro.agents.ai_agent import AI_Agent
import logging

logger = logging.getLogger(__name__)

class AdversarialAgent(AI_Agent):
    def __init__(self, name: str, attack, target: str = "input"):
        """
        Initialize the AdversarialAgent.

        Args:
            name (str): The name of the agent.
            attack (callable): The attack function to apply.
            target (str): The target of the attack, either "input" or "output".
        """
        super().__init__(name)
        self.attack = attack
        self.target = target
        logger.info(f"Initialized AdversarialAgent '{name}' targeting {target}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the adversarial attack to the input or output data.

        Args:
            input_data (Dict[str, Any]): The input data containing 'text' and 'output' keys.

        Returns:
            Dict[str, Any]: The perturbed data.
        """
        logger.info(f"Running adversarial agent: {self.name}")
        if self.target == "input":
            if input_data.get('text'):
                input_data['text'] = self.attack(input_data['text'])
            else:
                logger.warning("Input text is empty or missing. Skipping attack.")
        elif self.target == "output":
            if input_data.get('output'):
                input_data['output'] = self.attack(input_data['output'])
            else:
                logger.warning("Output text is empty or missing. Skipping attack.")
        else:
            raise ValueError(f"Invalid target: {self.target}")
        return input_data