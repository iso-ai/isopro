"""
Conversation Simulator

This module provides a high-level interface for running conversation simulations
with different personas and analyzing the results using Anthropic's Claude API.
"""

import logging
from .conversation_environment import ConversationEnvironment
from .custom_persona import create_custom_persona

logger = logging.getLogger(__name__)

class ConversationSimulator:
    """
    ConversationSimulator orchestrates conversation simulations with various personas using Claude.
    """

    def __init__(self, ai_prompt="You are a helpful customer service agent. Respond politely and professionally."):
        """
        Initialize the ConversationSimulator.

        Args:
            ai_prompt (str): The prompt to guide the Claude-based AI agent's behavior.
        """
        self.environment = ConversationEnvironment(ai_prompt)
        logger.info("Initialized ConversationSimulator with Claude")

    def run_simulation(self, persona_type, num_turns=5, claude_model="claude-3-opus-20240229", **persona_kwargs):
        """
        Run a conversation simulation with a specified persona using Claude.

        Args:
            persona_type (str): The type of persona to use in the simulation.
            num_turns (int): The number of conversation turns to simulate.
            claude_model (str): The specific Claude model to use for the simulation.
            **persona_kwargs: Additional arguments for creating the persona.

        Returns:
            list: A list of dictionaries containing the conversation history.
        """
        self.environment.set_ai_agent(model=claude_model)
        self.environment.set_user_persona(persona_type, **persona_kwargs)
        conversation_history = self.environment.run_conversation(num_turns)
        logger.info(f"Completed simulation with {persona_type} persona using Claude model {claude_model}")
        return conversation_history

    def run_custom_simulation(self, name, characteristics, message_templates, num_turns=5, claude_model="claude-3-opus-20240229"):
        """
        Run a conversation simulation with a custom persona using Claude.

        Args:
            name (str): The name of the custom persona.
            characteristics (list): A list of characteristics that define the persona.
            message_templates (list): A list of message templates the persona can use.
            num_turns (int): The number of conversation turns to simulate.
            claude_model (str): The specific Claude model to use for the simulation.

        Returns:
            list: A list of dictionaries containing the conversation history.
        """
        custom_persona = create_custom_persona(name, characteristics, message_templates)
        self.environment.set_ai_agent(model=claude_model)
        self.environment.user_persona = custom_persona
        conversation_history = self.environment.run_conversation(num_turns)
        logger.info(f"Completed simulation with custom persona: {name} using Claude model {claude_model}")
        return conversation_history