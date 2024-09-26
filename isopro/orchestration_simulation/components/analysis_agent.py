# components/analysis_agent.py

from .subagent import SubAgent
import logging
import os
from dotenv import load_dotenv
from isozero.reason_sim import OpenAIAgent, ClaudeAgent, ReasonSimulation, ReasonSimulationWrapper
from typing import Dict, Any

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AnalysisAgent(SubAgent):
    def __init__(self, name, priority=0):
        super().__init__(name, self.analyze, priority)
        self.model_type = self.determine_model_type()
        self.agent = self.initialize_agent()

    def determine_model_type(self):
        task = "Determine the best AI model for text analysis considering model capability, ease of use, and computational requirements. Options are OpenAI and Claude. Respond with only the name of the chosen model."
        simulation = ReasonSimulation(task, max_steps=5)
        wrapper = ReasonSimulationWrapper(OpenAIAgent(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4"), simulation)

        logger.info("Starting model determination simulation")

        for step in range(5):
            state = wrapper.step()
            logger.info(f"Model determination - Step {step + 1} completed")

        final_state = wrapper.render()
        logger.info("Model determination simulation completed")

        wrapper.close()

        # Parse the string output to determine the chosen model
        chosen_model = final_state.strip().lower()
        if 'openai' in chosen_model:
            return "OpenAI"
        elif 'claude' in chosen_model:
            return "Claude"
        else:
            logger.warning(f"Unclear model choice: {chosen_model}. Defaulting to OpenAI.")
            return "OpenAI"

    def initialize_agent(self):
        if self.model_type == "OpenAI":
            return OpenAIAgent(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
        elif self.model_type == "Claude":
            return ClaudeAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def run_simulation(self, task: str, max_steps: int = 5) -> Dict[str, Any]:
        simulation = ReasonSimulation(task, max_steps=max_steps)
        wrapper = ReasonSimulationWrapper(self.agent, simulation)

        logger.info(f"Starting analysis simulation with {self.model_type}")

        for step in range(max_steps):
            state = wrapper.step()
            logger.info(f"{self.model_type} Analysis - Step {step + 1} completed")

        final_state = wrapper.render()
        logger.info(f"Analysis simulation with {self.model_type} completed")

        wrapper.close()
        return {'output': final_state}

    def analyze(self, input_data):
        try:
            task = f"Analyze the following text:\n\n{input_data}\n\nProvide a detailed analysis."
            result = self.run_simulation(task)
            
            # Extract the final analysis from the simulation result
            analysis = result.get('output', 'No analysis produced')
            
            logger.info(f"Analysis Agent output:\n{analysis}")
            return {"result": analysis}
        except Exception as e:
            logger.error(f"Error in Analysis Agent: {e}")
            return {"result": f"Error in Analysis Agent: {str(e)}"}