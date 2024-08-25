"""
User Personas

This module defines various user personas for the conversation simulation.
"""

import random
import logging

logger = logging.getLogger(__name__)

class UserPersona:
    """
    Base class for user personas in the conversation simulation.
    """

    def __init__(self, name):
        self.name = name

    def generate_message(self, conversation_history):
        """
        Generate a message based on the persona and conversation history.

        Args:
            conversation_history (list): A list of dictionaries containing the conversation history.

        Returns:
            str: The generated message.
        """
        raise NotImplementedError("Subclasses must implement generate_message method")

    @staticmethod
    def create(persona_type, **kwargs):
        """
        Factory method to create user personas.

        Args:
            persona_type (str): The type of user persona to create.
            **kwargs: Additional arguments for the user persona.

        Returns:
            UserPersona: An instance of the specified user persona.
        """
        persona_classes = {
            "upset": UpsetCustomer,
            "human_request": HumanRequestCustomer,
            "inappropriate": InappropriateCustomer,
            "incomplete_info": IncompleteInfoCustomer,
        }

        if persona_type not in persona_classes:
            raise ValueError(f"Unknown persona type: {persona_type}")

        return persona_classes[persona_type](**kwargs)

class UpsetCustomer(UserPersona):
    def __init__(self):
        super().__init__("Upset Customer")
        self.complaints = [
            "This is unacceptable!",
            "I've been waiting for hours!",
            "I want to speak to your manager!",
            "This is the worst service I've ever experienced!",
            "I'm extremely disappointed with your company!",
        ]

    def generate_message(self, conversation_history):
        message = random.choice(self.complaints)
        logger.debug(f"UpsetCustomer generated message: {message}")
        return message

class HumanRequestCustomer(UserPersona):
    def __init__(self):
        super().__init__("Human Request Customer")
        self.requests = [
            "Can I speak to a human representative?",
            "I don't want to talk to a bot. Get me a real person.",
            "Is there a way to talk to an actual employee?",
            "I need to speak with a human agent, not an AI.",
            "Please transfer me to a live representative.",
        ]

    def generate_message(self, conversation_history):
        message = random.choice(self.requests)
        logger.debug(f"HumanRequestCustomer generated message: {message}")
        return message

class InappropriateCustomer(UserPersona):
    def __init__(self):
        super().__init__("Inappropriate Customer")
        self.inappropriate_words = ["[INAPPROPRIATE1]", "[INAPPROPRIATE2]", "[INAPPROPRIATE3]"]

    def generate_message(self, conversation_history):
        message = f"You're a {random.choice(self.inappropriate_words)} and this service is {random.choice(self.inappropriate_words)}!"
        logger.debug(f"InappropriateCustomer generated message: {message}")
        return message

class IncompleteInfoCustomer(UserPersona):
    def __init__(self):
        super().__init__("Incomplete Info Customer")
        self.vague_requests = [
            "I need help with my account.",
            "There's a problem with my order.",
            "Something's not working right.",
            "I have a question about your service.",
            "Can you check on the status of my thing?",
        ]

    def generate_message(self, conversation_history):
        message = random.choice(self.vague_requests)
        logger.debug(f"IncompleteInfoCustomer generated message: {message}")
        return message