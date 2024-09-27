# ISOPro: Pro Tools for Intelligent Simulation Orchestration for Large Language Models

ISOPRO is a powerful and flexible Python package designed for creating, managing, and analyzing simulations involving Large Language Models (LLMs). It provides a comprehensive suite of tools for reinforcement learning, conversation simulations, adversarial testing, custom environment creation, and advanced orchestration of multi-agent systems.

## Features

- **Custom Environment Creation**: Easily create and manage custom simulation environments for LLMs. 
- **Conversation Simulation**: Simulate and analyze conversations with AI agents using various user personas. 
- **Adversarial Testing**: Conduct adversarial simulations to test the robustness of LLM-based systems. 
- **Reinforcement Learning**: Implement and experiment with RL algorithms in LLM contexts. 
- **Utility Functions**: Analyze simulation results, calculate LLM metrics, and more.
- **Flexible Integration**: Works with popular LLM platforms like OpenAI's GPT models, Claude (Anthropic), and Hugging Face models.
- **Orchestration Simulation**: Manage and execute complex multi-agent simulations with different execution modes. 

# isopro

isopro is a Python package for running various types of AI simulations and experiments. It provides tools for adversarial testing, conversation simulation, reinforcement learning, and AI orchestration.

## Installation

You can install isopro using pip:

```bash
pip install isopro
```

## Usage

### Adversarial Simulation

Test the robustness of AI models against adversarial attacks.

```python
from isopro.adversarial_simulation import AdversarialSimulator, AdversarialEnvironment
from isopro.agents.ai_agent import AI_Agent
import anthropic

class ClaudeAgent(AI_Agent):
    def __init__(self, name):
        super().__init__(name)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, input_data):
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": input_data['text']}]
        )
        return response.content[0].text

# Create the AdversarialEnvironment
adv_env = AdversarialEnvironment(
    agent_wrapper=ClaudeAgent("Claude Agent"),
    num_adversarial_agents=2,
    attack_types=["textbugger", "deepwordbug"],
    attack_targets=["input", "output"]
)

# Set up the adversarial simulator
simulator = AdversarialSimulator(adv_env)

# Run the simulation
input_data = ["What is the capital of France?", "How does photosynthesis work?"]
simulation_results = simulator.run_simulation(input_data, num_steps=1)
```

### Conversation Simulation

Simulate conversations between an AI assistant and various user personas.

```python
from isopro.conversation_simulation.conversation_simulator import ConversationSimulator

# Initialize the ConversationSimulator
simulator = ConversationSimulator(
    ai_prompt="You are an AI assistant created to be helpful, harmless, and honest. You are a customer service agent for a tech company. Respond politely and professionally."
)

# Run a simulation with a predefined persona
conversation_history = simulator.run_simulation("upset", num_turns=3)

# Run a simulation with a custom persona
custom_persona_name = "Techie Customer"
custom_characteristics = ["tech-savvy", "impatient", "detail-oriented"]
custom_message_templates = [
    "I've tried rebooting my device, but the error persists. Can you help?",
    "What's the latest update on the cloud service outage?",
    "I need specifics on the API rate limits for the enterprise plan."
]

custom_conversation = simulator.run_custom_simulation(
    custom_persona_name,
    custom_characteristics,
    custom_message_templates,
    num_turns=3
)
```

### Reinforcement Learning with LLM

Integrate Large Language Models with reinforcement learning environments.

```python
import gymnasium as gym
from isopro.rl.rl_agent import RLAgent
from isopro.rl.rl_environment import LLMRLEnvironment
from stable_baselines3 import PPO
from isopro.rl.llm_cartpole_wrapper import LLMCartPoleWrapper



agent_prompt = """You are an AI trained to play the CartPole game. 
Your goal is to balance a pole on a moving cart for as long as possible. 
You will receive observations about the cart's position, velocity, pole angle, and angular velocity. 
Based on these, you should decide whether to move the cart left or right."""

env = LLMCartPoleWrapper(agent_prompt, llm_call_limit=100, api_key=os.getenv("ANTHROPIC_API_KEY"))
rl_agent = RLAgent("LLM_CartPole_Agent", env, algorithm='PPO')

# Train the model
model.learn(total_timesteps=2)

# Test the model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
```

### AI Orchestration

Orchestrate multiple AI agents to work together on complex tasks.

```python
from isopro.orchestration_simulation import OrchestrationEnv
from isopro.orchestration_simulation.components import LLaMAAgent, AnalysisAgent, WritingAgent
from isopro.orchestration_simulation.evaluator import Evaluator

# Create the orchestration environment
env = OrchestrationEnv()

# Add agents to the environment
env.add_component(LLaMAAgent("Research", "conduct thorough research on the impact of artificial intelligence on job markets"))
env.add_component(AnalysisAgent("Analysis"))
env.add_component(WritingAgent("Writing"))

# Define the task
task = "Prepare a comprehensive report on the impact of artificial intelligence on job markets in the next decade."

# Run simulations in different modes
modes = ['parallel', 'sequence', 'node']
results = {}

for mode in modes:
    result = env.run_simulation(mode=mode, input_data={'task': task, 'run_order': 'first'})
    results[mode] = result

# Evaluate the results
evaluator = Evaluator()
best_mode = evaluator.evaluate(results)
print(f"The best execution mode for this task was: {best_mode}")
```

## Documentation

For more detailed information on each module and its usage, please refer to the [full documentation](https://isopro.readthedocs.io).

## Examples

The [isopro examples](https://github.com/iso-ai/isopro_examples) repository in the repository contains Jupyter notebooks with more detailed examples:

- `adversarial_example.ipynb`: Demonstrates adversarial testing of language models.
- `conversation_simulation_example.ipynb`: Shows how to simulate conversations with various user personas.
- `run_cartpole_example.ipynb`: Illustrates the integration of LLMs with reinforcement learning.
- `orchestrator_example.ipynb`: Provides a tutorial on using the AI orchestration capabilities.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any problems or have any questions, please [open an issue](https://github.com/yourusername/isopro/issues) on our GitHub repository.

## Citation

If you use ISOPRO in your research, please cite it as follows:

```
@software{isopro2024,
  author = {Jazmia Henry},
  title = {ISOPRO: Intelligent Simulation Orchestration for Large Language Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/iso-ai/isopro}}
}
```

## Contact

For questions or support, please open an issue on our [GitHub issue tracker](https://github.com/iso-ai/isopro/issues).
