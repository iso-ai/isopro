"""
Custom Persona

This module allows users to create custom personas for the conversation simulation.
"""

import logging
from .user_personas import UserPersona

logger = logging.getLogger(__name__)

class CustomPersona(UserPersona):
    """
    CustomPersona allows users to create their own persona with specific characteristics.
    """

    def __init__(self, name, characteristics, message_templates):
        """
        Initialize the CustomPersona.

        Args:
            name (str): The name of the custom persona.
            characteristics (list): A list of characteristics that define the persona.
            message_templates (list): A list of message templates the persona can use.
        """
        super().__init__(name)
        self.characteristics = characteristics
        self.message_templates = message_templates
        logger.info(f"Created CustomPersona: {name}")

    def generate_message(self, conversation_history):
        """
        Generate a message based on the custom persona's characteristics and templates.

        Args:
            conversation_history (list): A list of dictionaries containing the conversation history.

        Returns:
            str: The generated message.
        """
        import random
        message = random.choice(self.message_templates)
        logger.debug(f"CustomPersona '{self.name}' generated message: {message}")
        return message

def create_custom_persona(name, characteristics, message_templates):
    """
    Create a custom persona with the given characteristics and message templates.

    Args:
        name (str): The name of the custom persona.
        characteristics (list): A list of characteristics that define the persona.
        message_templates (list): A list of message templates the persona can use.

    Returns:
        CustomPersona: An instance of the custom persona.
    """
    return CustomPersona(name, characteristics, message_templates)