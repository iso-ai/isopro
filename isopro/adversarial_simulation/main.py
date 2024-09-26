import logging
from typing import List
from .adversarial_simulator import AdversarialSimulator
from .adversarial_environment import AdversarialEnvironment
from isopro.utils.analyze_adversarial_sim import analyze_adversarial_results, summarize_adversarial_impact
from isopro.agents.ai_agent import AI_Agent
import anthropic
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import numpy as np
import torch

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def step(self, sim_state):
        sim_state['output'] = self.run(sim_state)
        return sim_state

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def setup_logging(log_dir: str, run_id: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"adv-{run_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_scores(output_dir: str, run_id: str, analysis_results: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"adv-{run_id}.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, cls=NumpyEncoder)
        
        logging.info(f"Saved analysis results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving analysis results: {str(e)}")

def get_sample_inputs() -> List[str]:
    return [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Explain the theory of relativity."
    ]

def main():
    try:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        log_dir = "logs"
        setup_logging(log_dir, run_id)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting adversarial simulation run {run_id}")

        claude_agent = ClaudeAgent("Claude Agent")

        # Create the AdversarialEnvironment
        adv_env = AdversarialEnvironment(
            agent_wrapper=claude_agent,
            num_adversarial_agents=2,
            attack_types=["textbugger", "deepwordbug"],
            attack_targets=["input", "output"]
        )

        # Set up the adversarial simulator with the environment
        simulator = AdversarialSimulator(adv_env)

        input_data = get_sample_inputs()

        logger.info("Starting adversarial simulation...")
        simulation_results = simulator.run_simulation(input_data, num_steps=1)

        logger.info("Analyzing simulation results...")
        analysis_results = analyze_adversarial_results(simulation_results)

        summary = summarize_adversarial_impact(analysis_results)

        print("\nAdversarial Simulation Summary:")
        print(summary)

        output_dir = "output"
        save_scores(output_dir, run_id, analysis_results)

        logger.info("Simulation complete.")
    
    except Exception as e:
        logger.error(f"An error occurred during the simulation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
