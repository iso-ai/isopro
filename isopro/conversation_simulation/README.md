# Conversation Simulator

This module is part of the `isopro` package and simulates conversations between an AI assistant (either Claude or GPT-4) and various user personas. It's designed to test and demonstrate how the AI handles different types of customer service scenarios.

## Project Structure

The Conversation Simulator is located in the `conversation_simulator` folder within the `isopro` package:

```
isopro/
└── conversation_simulator/
    ├── main.py
    ├── conversation_simulator.ipynb
    ├── conversation_agent.py
    ├── conversation_environment.py
    ├── custom_persona.py
    └── user_personas.py
```

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.7 or later.
* You have an Anthropic API key (for Claude) and/or an OpenAI API key (for GPT-4).
* You have installed the `isopro` package.
* For the Jupyter notebook, you have Jupyter Notebook or JupyterLab installed.

## Setting up the Conversation Simulator

1. If you haven't already, install the `isopro` package:
   ```
   pip install isopro
   ```

2. Create a `.env` file in your project root and add your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Conversation Simulator

You can run the Conversation Simulator either as a Python script or interactively using a Jupyter notebook.

### Using the Python Script

1. Basic usage:
   ```python
   from isopro.conversation_simulator.main import main

   if __name__ == "__main__":
       main()
   ```

2. Running from the command line:
   ```
   python -m isopro.conversation_simulator.main
   ```

### Using the Jupyter Notebook

Navigate to the `isopro/conversation_simulator/` directory and open the `conversation_simulator.ipynb` file using Jupyter Notebook or JupyterLab. Here's what you'll find in the notebook:

```python
# Conversation Simulator Jupyter Notebook

## Setup

import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from dotenv import load_dotenv
from isopro.conversation_simulation.conversation_simulator import ConversationSimulator
from isopro.conversation_simulation.custom_persona import create_custom_persona

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

print("Setup complete.")

## Helper Functions

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

print("Helper functions defined.")

## Main Simulation Function

def run_simulation():
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

    return output_content

print("Main simulation function defined.")

## Run the Simulation

simulation_output = run_simulation()
print(simulation_output)

## Analyze the Results

# Example analysis: Count the number of apologies
apology_count = simulation_output.lower().count("sorry") + simulation_output.lower().count("apologi")
print(f"Number of apologies: {apology_count}")

# Example analysis: Average length of AI responses
ai_responses = [line.split(": ", 1)[1] for line in simulation_output.split("\n") if line.startswith("Assistant: ")]
avg_response_length = sum(len(response.split()) for response in ai_responses) / len(ai_responses)
print(f"Average length of AI responses: {avg_response_length:.2f} words")

## Conclusion

# This notebook demonstrates how to use the Conversation Simulator from the isopro package. 
# You can modify the personas, adjust the number of turns, or add your own analysis to 
# further explore the capabilities of the AI models in customer service scenarios.
```

## Output and Logs

- Simulation outputs are saved in the `output` directory within your current working directory.
- Logs are saved in the `logs` directory within your current working directory.

## Customizing the Simulation

You can customize the simulation by modifying the `main.py` file or the Jupyter notebook:

- To change the predefined personas, modify the `personas` list.
- To adjust the custom persona, modify the `custom_persona_name`, `custom_characteristics`, and `custom_message_templates` variables.
- To change the number of turns in each conversation, modify the `num_turns` parameter in the `run_simulation` and `run_custom_simulation` method calls.

In the Jupyter notebook, you can also add new cells for additional analysis or visualization of the results.

## Troubleshooting

If you encounter any issues:

1. Make sure your API keys are correctly set in the `.env` file or environment variables.
2. Check the logs in the `logs` directory for detailed error messages.
3. Ensure you have the latest version of the `isopro` package installed.
4. For Jupyter notebook issues, make sure you have Jupyter installed and are running the notebook from the correct directory.

If problems persist, please open an issue in the project repository.

## Contributing

Contributions to the Conversation Simulator are welcome. Please feel free to submit a Pull Request to the `isopro` repository.

## License

This project is licensed under the MIT License - see the LICENSE file in the `isopro` package for details.