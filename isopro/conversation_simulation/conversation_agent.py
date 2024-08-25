"""
Conversation Agent

This module defines the AI agent used in the conversation simulation, using Anthropic's Claude API.
"""

import anthropic
import os
import logging
from ..agents.ai_agent import AI_Agent

logger = logging.getLogger(__name__)

class ConversationAgent(AI_Agent):
    """
    ConversationAgent

    This class defines an AI agent that engages in conversations using Anthropic's Claude API.
    """

    def __init__(self, name, prompt, model="claude-3-opus-20240229"):
        """
        Initialize the ConversationAgent.

        Args:
            name (str): The name of the agent.
            prompt (str): The prompt to guide the agent's behavior.
            model (str): The name of the Claude model to use.
        """
        super().__init__(name)
        self.prompt = prompt
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info(f"Initialized ConversationAgent '{name}' with Claude model {model}")

    def generate_response(self, conversation_history):
        """
        Generate a response based on the conversation history using Claude.

        Args:
            conversation_history (list): A list of dictionaries containing the conversation history.

        Returns:
            str: The generated response.
        """
        messages = [{"role": "system", "content": self.prompt}] + conversation_history

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=messages
            )
            ai_message = response.content[0].text.strip()
            logger.debug(f"Generated response: {ai_message}")
            return ai_message
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble responding at the moment."