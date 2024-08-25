# ISOPro: Pro Tools for Intelligent Simulation Orchestration for Large Language Models

ISOPRO is a powerful and flexible Python package designed for creating, managing, and analyzing simulations involving Large Language Models (LLMs). It provides a comprehensive suite of tools for reinforcement learning, conversation simulations, adversarial testing, and custom environment creation.

## Features

- **Custom Environment Creation**: Easily create and manage custom simulation environments for LLMs.
- **Conversation Simulation**: Simulate and analyze conversations with AI agents using various user personas.
- **Adversarial Testing**: Conduct adversarial simulations to test the robustness of LLM-based systems.
- **Reinforcement Learning**: Implement and experiment with RL algorithms in LLM contexts.
- **Utility Functions**: Analyze simulation results, calculate LLM metrics, and more.
- **Flexible Integration**: Works with popular LLM platforms like Claude (Anthropic) and Hugging Face models.

## Installation

You can install ISOPRO directly from GitHub using pip:

```bash
pip install git+https://github.com/iso-ai/isopro.git
```

## Quick Start

Here's a simple example of how to use ISOPRO for a conversation simulation:

```python
from isopro.conversation_simulation import ConversationSimulator
from isopro.utils.conversation_analysis import analyze_conversation

# Create a conversation simulator
simulator = ConversationSimulator(ai_prompt="You are a helpful assistant.")

# Run a simulation
conversation_history = simulator.run_simulation(persona_type="curious", num_turns=5)

# Analyze the conversation
analysis_results = analyze_conversation(conversation_history)
print(f"Overall sentiment: {analysis_results['overall_sentiment']:.2f}")
```

For more detailed examples, check out the Jupyter notebooks in the `examples/` directory.

## Documentation

For full documentation, including API references and advanced usage examples, please visit our [GitHub Wiki](https://github.com/yourusername/isopro/wiki).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

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

For questions or support, please open an issue on our [GitHub issue tracker](https://https://github.com/iso-ai/isopro//tree/main/.github/ISSUE_TEMPLATE.md).
