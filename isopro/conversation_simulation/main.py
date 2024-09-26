import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from dotenv import load_dotenv
from .conversation_simulator import ConversationSimulator
from .custom_persona import create_custom_persona

# Load environment variables
load_dotenv()

# Set up logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, "conversation_simulator.log")

# Create a rotating file handler
file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def save_output(content, filename):
    """Save the output content to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def get_user_choice():
    """Get user's choice of AI model."""
    while True:
        choice = input("Choose AI model (claude/openai): ").lower()
        if choice in ['claude', 'openai']:
            return choice
        print("Invalid choice. Please enter 'claude' or 'openai'.")

def main():
    # Get user's choice of AI model
    ai_choice = get_user_choice()

    # Set up the appropriate model and API key
    if ai_choice == 'claude':
        model = "claude-3-opus-20240229"
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        ai_name = "Claude"
    else:  # openai
        model = "gpt-4-1106-preview"
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        ai_name = "GPT-4 Turbo"

    # Initialize the ConversationSimulator
    simulator = ConversationSimulator(
        ai_prompt=f"You are {ai_name}, an AI assistant created to be helpful, harmless, and honest. You are a customer service agent for a tech company. Respond politely and professionally."
    )

    output_content = f"Conversation Simulator using {ai_name} model: {model}\n\n"

    # Run simulations with different personas
    personas = ["upset", "human_request", "inappropriate", "incomplete_info"]
    
    for persona in personas:
        logger.info(f"Running simulation with {persona} persona using {ai_name}")
        conversation_history = simulator.run_simulation(persona, num_turns=3)
        
        output_content += f"\nConversation with {persona} persona:\n"
        for message in conversation_history:
            output_line = f"{message['role'].capitalize()}: {message['content']}\n"
            output_content += output_line
            logger.debug(output_line.strip())
        output_content += "\n" + "-"*50 + "\n"

    # Create and run a simulation with a custom persona
    custom_persona_name = "Techie Customer"
    custom_characteristics = ["tech-savvy", "impatient", "detail-oriented"]
    custom_message_templates = [
        "I've tried rebooting my device, but the error persists. Can you help?",
        "What's the latest update on the cloud service outage?",
        "I need specifics on the API rate limits for the enterprise plan.",
        "The latency on your servers is unacceptable. What's being done about it?",
        "Can you explain the technical details of your encryption method?"
    ]

    logger.info(f"Running simulation with custom persona: {custom_persona_name} using {ai_name}")
    custom_conversation = simulator.run_custom_simulation(
        custom_persona_name,
        custom_characteristics,
        custom_message_templates,
        num_turns=3
    )

    output_content += f"\nConversation with {custom_persona_name}:\n"
    for message in custom_conversation:
        output_line = f"{message['role'].capitalize()}: {message['content']}\n"
        output_content += output_line
        logger.debug(output_line.strip())

    # Save the output to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f"{ai_name.lower()}_conversation_output_{timestamp}.txt")
    save_output(output_content, output_file)
    logger.info(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()