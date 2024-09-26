# components/llama_agent.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .subagent import SubAgent
import logging

logger = logging.getLogger(__name__)

class LLaMAAgent(SubAgent):
    def __init__(self, name, task, model_name="facebook/opt-350m", priority=0):
        super().__init__(name, self.llama_behavior, priority)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.task = task
        self.initialize_model()
        
    def initialize_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Error initializing LLaMA model: {e}")
            raise
        
    def llama_behavior(self, input_data):
        if not self.tokenizer or not self.model:
            logger.error("LLaMA model not properly initialized")
            return {"result": "Error: Model not initialized"}

        run_order = input_data.get('run_order', 'unknown')
        previous_output = input_data.get('previous_output', '')

        prompt = f"""
        Task: {self.task}
        
        Your current position in the run order: {run_order}
        
        Previous output (if any):
        {previous_output}

        Based on your position in the run order, provide a concise and informative response.
        Response:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=300,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("Response:")[-1].strip()

            if not answer:
                logger.warning(f"LLaMA Agent {self.name} generated an empty response")
                return {"result": "Error: Empty response generated"}

            logger.info(f"LLaMA Agent {self.name} response:\n{answer}")
            return {"result": answer}
        except Exception as e:
            logger.error(f"Error in LLaMA Agent {self.name}: {e}")
            return {"result": f"Error in LLaMA Agent: {str(e)}"}