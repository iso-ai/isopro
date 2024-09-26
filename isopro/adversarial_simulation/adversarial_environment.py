"""
Adversarial Environment

This module defines the AdversarialEnvironment class, which manages adversarial agents and applies attacks to the simulation state.
"""

import random
from typing import List, Dict, Any
from isopro.environments.simulation_environment import SimulationEnvironment
from .adversarial_agent import AdversarialAgent
from .attack_utils import get_model_and_tokenizer, create_attack, get_available_attacks
import logging

logger = logging.getLogger(__name__)

class AdversarialEnvironment(SimulationEnvironment):
    def __init__(self, agent_wrapper, num_adversarial_agents: int = 1, attack_types: List[str] = None, attack_targets: List[str] = None):
        """
        Initialize the AdversarialEnvironment.

        Args:
            agent_wrapper: The wrapped agent to pass the adversarially modified state to.
            num_adversarial_agents (int): The number of adversarial agents to create.
            attack_types (List[str], optional): The types of attacks to use. If None, all available attacks will be used.
            attack_targets (List[str], optional): The targets for the attacks ("input", "output", or both). If None, both will be used.
        """
        super().__init__()
        self.agent_wrapper = agent_wrapper
        self.num_adversarial_agents = num_adversarial_agents
        self.attack_types = attack_types or get_available_attacks()
        self.attack_targets = attack_targets or ["input", "output"]
        self.model, self.tokenizer = get_model_and_tokenizer()
        self._create_adversarial_agents()
        logger.info(f"Initialized AdversarialEnvironment with {num_adversarial_agents} agents")

    def _create_adversarial_agents(self):
        """Create adversarial agents with random attack types and targets."""
        for i in range(self.num_adversarial_agents):
            attack_type = random.choice(self.attack_types)
            attack_target = random.choice(self.attack_targets)
            attack = create_attack(attack_type, self.model, self.tokenizer)
            agent = AdversarialAgent(name=f"Adversarial Agent {i+1} ({attack_type}, {attack_target})", attack=attack, target=attack_target)
            self.add_agent(agent)
        logger.info(f"Created {self.num_adversarial_agents} adversarial agents")

    def step(self, sim_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply adversarial attacks and step the environment.

        Args:
            sim_state (Dict[str, Any]): The current simulation state.

        Returns:
            Dict[str, Any]: The updated simulation state after applying attacks and stepping the wrapped agent.
        """
        # Apply adversarial attacks
        for agent in self.agents:
            sim_state = agent.run(sim_state)

        # Pass the adversarially modified state to the wrapped agent
        return self.agent_wrapper.step(sim_state)

    def reset(self):
        """Reset the environment and recreate adversarial agents."""
        super().reset()
        self._create_adversarial_agents()
        logger.info("Reset AdversarialEnvironment and recreated agents")

    def get_attack_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of attack types and targets among the adversarial agents.

        Returns:
            Dict[str, int]: A dictionary containing the count of each attack type and target.
        """
        attack_counts = {f"{attack_type}_{target}": 0 for attack_type in self.attack_types for target in self.attack_targets}
        for agent in self.agents:
            attack_type, target = agent.name.split('(')[-1].split(')')[0].split(', ')
            attack_counts[f"{attack_type}_{target}"] += 1
        logger.info(f"Current attack distribution: {attack_counts}")
        return attack_counts