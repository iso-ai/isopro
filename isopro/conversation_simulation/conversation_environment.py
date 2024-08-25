"""
Conversation Environment

This module defines the environment for simulating conversations between a Claude-based AI agent and users with various personas.
"""

import logging
from ..environments.simulation_environment import SimulationEnvironment
from .conversation_agent import ConversationAgent
from .user_personas import UserPersona

logger = logging.getLogger(__name__)

class ConversationEnvironment(SimulationEnvironment):
    """
    ConversationEnvironment

    This class provides an environment for simulating conversations between Claude-based AI agents and users with various personas.
    """

    def __init__(self, ai_prompt="You are a helpful customer service agent. Respond politely and professionally."):
        """
        Initialize the ConversationEnvironment.

        Args:
            ai_prompt (str): The prompt to guide the AI agent's behavior.
        """
        super().__init__()
        self.ai_prompt = ai_prompt
        self.ai_agent = None
        self.user_persona = None
        logger.info("Initialized ConversationEnvironment")

    def set_ai_agent(self, model="claude-3-opus-20240229"):
        """
        Set up the Claude-based AI agent for the conversation.

        Args:
            model (str): The name of the Claude model to use.
        """
        self.ai_agent = ConversationAgent("Customer Service AI", self.ai_prompt, model)
        logger.info(f"Set AI agent with Claude model: {model}")
    def set_user_persona(self, persona_type, **kwargs):
        """
        Set the user persona for the conversation.

        Args:
            persona_type (str): The type of user persona to use.
            **kwargs: Additional arguments for the user persona.
        """
        self.user_persona = UserPersona.create(persona_type, **kwargs)
        logger.info(f"Set user persona: {persona_type}")

    def run_conversation(self, num_turns=5):
        """
        Run a conversation between the AI agent and the user persona.

        Args:
            num_turns (int): The number of conversation turns to simulate.

        Returns:
            list: A list of dictionaries containing the conversation history.
        """
        if not self.ai_agent or not self.user_persona:
            raise ValueError("Both AI agent and user persona must be set before running a conversation.")

        conversation_history = []
        for _ in range(num_turns):
            user_message = self.user_persona.generate_message(conversation_history)
            conversation_history.append({"role": "user", "content": user_message})
            logger.debug(f"User: {user_message}")

            ai_response = self.ai_agent.generate_response(conversation_history)
            conversation_history.append({"role": "assistant", "content": ai_response})
            logger.debug(f"AI: {ai_response}")

        logger.info("Completed conversation simulation")
        return conversation_history