"""
Adversarial Simulator

This module provides a high-level interface for running adversarial simulations.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AdversarialSimulator:
    def __init__(self, environment):
        """
        Initialize the AdversarialSimulator.

        Args:
            environment: The AdversarialEnvironment to use in the simulation.
        """
        self.environment = environment
        logger.info("Initialized AdversarialSimulator")

    def run_simulation(self, input_data: List[str], num_steps: int = 1) -> List[Dict[str, Any]]:
        """
        Run an adversarial simulation.

        Args:
            input_data (List[str]): The list of input texts to use in the simulation.
            num_steps (int): The number of steps to run the simulation for each input.

        Returns:
            List[Dict[str, Any]]: A list of simulation results, including original and perturbed inputs and outputs.
        """
        results = []
        for text in input_data:
            sim_state = {"text": text, "output": ""}
            original_output = self.environment.agent_wrapper.run({"text": text})
            for _ in range(num_steps):
                sim_state = self.environment.step(sim_state)
            results.append({
                "original_input": text,
                "perturbed_input": sim_state["text"],
                "original_output": original_output,
                "perturbed_output": sim_state["output"]
            })
        logger.info(f"Completed simulation with {len(input_data)} inputs and {num_steps} steps each")
        return results