"""
Conversation Agent

This module defines the AI agent used in the conversation simulation, using Anthropic's Claude API.
"""

import anthropic
import os
import logging
from ..agents.ai_agent import AI_Agent
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

class ConversationAgent(AI_Agent):
    def __init__(self, name, prompt, model="claude-3-opus-20240229"):
        super().__init__(name)
        self.prompt = prompt
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info(f"Initialized ConversationAgent '{name}' with Claude model {model}")

    def generate_response(self, conversation_history):
        try:
            messages = [{"role": "user" if msg["role"] != "assistant" else "assistant", "content": msg["content"]} 
                        for msg in conversation_history]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=self.prompt,
                messages=messages
            )
            ai_message = response.content[0].text.strip()
            logger.debug(f"Generated response: {ai_message}")
            return ai_message
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble responding at the moment."