"""
Core Simulation API for ISOPro

This module provides REST API endpoints for various simulation types within the ISOPro framework:
- reason_sim: Basic reasoning over prompt + context
- qa_sim: Question + persona-based simulations
- adversarial_sim: Prompt + attack type simulations
- orchestration_sim: Multi-agent interaction simulations

The API provides a unified interface for accessing all simulation types.
"""

import os
import uuid
import datetime
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import anthropic
from typing import Dict, Any, List, Optional
import traceback

# Import simulation modules
from isopro.adversarial_simulation.adversarial_simulator import AdversarialSimulator
from isopro.adversarial_simulation.adversarial_environment import AdversarialEnvironment
from isopro.adversarial_simulation.adversarial_agent import AdversarialAgent

from isopro.conversation_simulation.conversation_simulator import ConversationSimulator
from isopro.conversation_simulation.custom_persona import create_custom_persona

from isopro.orchestration_simulation.orchestration_env import OrchestrationEnv
from isopro.orchestration_simulation.agent import AgentComponent
from isopro.orchestration_simulation.utils import setup_logging

# For reasoning simulation, we use isozero imports if available or implement a basic version
try:
    from isozero.reason_sim import ClaudeAgent, ReasonSimulation
except ImportError:
    # Fallback implementation if isozero is not available
    class ClaudeAgent:
        def __init__(self, api_key, model="claude-3-sonnet-20240229"):
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            
        def generate(self, prompt, context=None):
            system_prompt = "You are a helpful, accurate assistant that provides clear and concise answers."
            if context:
                system_prompt += f"\n\nHere is important context: {context}"
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
    
    class ReasonSimulation:
        def __init__(self, agent):
            self.agent = agent
            
        def run(self, prompt, context=None):
            return {
                "output": self.agent.generate(prompt, context),
                "metadata": {
                    "model": self.agent.model,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            }

# Set up environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create and configure the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

def generate_run_id() -> str:
    """Generate a unique run ID for the simulation."""
    return str(uuid.uuid4())

def format_response(run_id: str, output: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Format the simulation response in a standardized way."""
    return {
        "run_id": run_id,
        "output": output,
        "metadata": {
            **metadata,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    }

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().isoformat()})

@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Main simulation endpoint that routes to specific simulation types.
    
    Requires a 'type' parameter to determine which simulation to run.
    Valid types: 'reason', 'qa', 'adversarial', 'orchestration'
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        sim_type = data.get('type')
        if not sim_type:
            return jsonify({"error": "Simulation type not specified"}), 400
            
        # Route to the appropriate simulation
        if sim_type == 'reason':
            return reason_sim()
        elif sim_type == 'qa':
            return qa_sim()
        elif sim_type == 'adversarial':
            return adversarial_sim()
        elif sim_type == 'orchestration':
            return orchestration_sim()
        else:
            return jsonify({"error": f"Unknown simulation type: {sim_type}"}), 400
            
    except Exception as e:
        logger.error(f"Error processing simulation request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/simulate/reason', methods=['POST'])
def reason_sim():
    """
    Reasoning simulation endpoint.
    
    Requires:
    - prompt: The reasoning prompt
    - context (optional): Additional context for reasoning
    - model (optional): The model to use (defaults to claude-3-sonnet)
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
            
        context = data.get('context', '')
        model = data.get('model', 'claude-3-sonnet-20240229')
        
        # Generate run ID
        run_id = generate_run_id()
        
        # Create and run reason simulation
        if not ANTHROPIC_API_KEY:
            return jsonify({"error": "ANTHROPIC_API_KEY not set in environment"}), 500
            
        agent = ClaudeAgent(api_key=ANTHROPIC_API_KEY, model=model)
        simulator = ReasonSimulation(agent)
        
        result = simulator.run(prompt, context)
        
        # Format and return the response
        return jsonify(format_response(
            run_id=run_id,
            output=result.get("output", ""),
            metadata={
                "simulation_type": "reason",
                "model": model,
                "prompt_length": len(prompt),
                "context_length": len(context) if context else 0,
            }
        ))
        
    except Exception as e:
        logger.error(f"Error in reason_sim: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/simulate/qa', methods=['POST'])
def qa_sim():
    """
    Question-answering simulation with personas.
    
    Requires:
    - question: The question to ask
    - persona_type (optional): Type of persona for the conversation 
      (or provide custom persona parameters)
    - custom_persona (optional): Settings for a custom persona
      - name: Persona name
      - characteristics: List of characteristics
      - message_templates: List of message templates
    - num_turns (optional): Number of conversation turns (default: 1)
    - model (optional): Claude model to use
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        question = data.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400
            
        # Get persona settings
        persona_type = data.get('persona_type')
        custom_persona = data.get('custom_persona')
        num_turns = data.get('num_turns', 1)
        model = data.get('model', 'claude-3-sonnet-20240229')
        
        # Generate run ID
        run_id = generate_run_id()
        
        # Create conversation simulator
        ai_prompt = data.get('ai_prompt', "You are a helpful assistant. Answer questions accurately and concisely.")
        simulator = ConversationSimulator(ai_prompt=ai_prompt)
        
        # Run conversation simulation
        if custom_persona:
            # Use custom persona
            name = custom_persona.get('name', 'Custom User')
            characteristics = custom_persona.get('characteristics', ['polite', 'curious'])
            message_templates = custom_persona.get('message_templates', [question])
            
            conversation_history = simulator.run_custom_simulation(
                name=name,
                characteristics=characteristics,
                message_templates=message_templates,
                num_turns=num_turns,
                claude_model=model
            )
        else:
            # Use predefined persona type
            conversation_history = simulator.run_simulation(
                persona_type=persona_type or 'neutral',
                num_turns=num_turns,
                claude_model=model
            )
        
        # Format and return the response
        return jsonify(format_response(
            run_id=run_id,
            output=conversation_history,
            metadata={
                "simulation_type": "qa",
                "model": model,
                "persona_type": persona_type if not custom_persona else "custom",
                "num_turns": num_turns,
                "question_length": len(question)
            }
        ))
        
    except Exception as e:
        logger.error(f"Error in qa_sim: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/simulate/adversarial', methods=['POST'])
def adversarial_sim():
    """
    Adversarial simulation endpoint.
    
    Requires:
    - prompt: The prompt to test
    - attack_type: Type of adversarial attack 
      (e.g., 'jailbreak', 'prompt_injection', 'data_extraction')
    - num_steps (optional): Number of simulation steps (default: 1)
    - model (optional): The model to test against (default: claude-3-sonnet)
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
            
        attack_type = data.get('attack_type')
        if not attack_type:
            return jsonify({"error": "No attack_type provided"}), 400
            
        num_steps = data.get('num_steps', 1)
        model = data.get('model', 'claude-3-sonnet-20240229')
        
        # Generate run ID
        run_id = generate_run_id()
        
        # Create the agent and environment
        adversarial_agent = AdversarialAgent(
            anthropic_api_key=ANTHROPIC_API_KEY,
            model=model
        )
        
        environment = AdversarialEnvironment(
            agent_wrapper=adversarial_agent,
            attack_type=attack_type
        )
        
        # Create and run simulator
        simulator = AdversarialSimulator(environment)
        results = simulator.run_simulation(
            input_data=[prompt],
            num_steps=num_steps
        )
        
        # Format and return the response
        return jsonify(format_response(
            run_id=run_id,
            output=results,
            metadata={
                "simulation_type": "adversarial",
                "model": model,
                "attack_type": attack_type,
                "num_steps": num_steps,
                "prompt_length": len(prompt)
            }
        ))
        
    except Exception as e:
        logger.error(f"Error in adversarial_sim: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/simulate/orchestration', methods=['POST'])
def orchestration_sim():
    """
    Multi-agent orchestration simulation endpoint.
    
    Requires:
    - input_data: The question or task for the orchestration
    - tools (optional): List of tools to use in the orchestration
    - mode (optional): Simulation mode (default: 'agent')
    - model (optional): The model to use (default: depends on available API keys)
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        input_data = data.get('input_data')
        if not input_data:
            return jsonify({"error": "No input_data provided"}), 400
            
        mode = data.get('mode', 'agent')
        
        # Set up logging for orchestration
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, 
            f"orchestration_sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        sim_logger = setup_logging(log_file=log_file)
        
        # Generate run ID
        run_id = generate_run_id()
        
        # Create simulation environment
        sim_env = OrchestrationEnv()
        
        # Create tools based on request or use defaults
        tools_config = data.get('tools', [])
        if not tools_config:
            # Use default tools
            from langchain.agents import Tool
            from langchain_openai import OpenAI
            from langchain.prompts import PromptTemplate
            
            # Create default tool components
            tools = []
            
            # Add research tool
            tools.append(Tool(
                name="Research",
                func=lambda x: f"Research results for: {x}",
                description="Research information on a topic"
            ))
            
            # Add analysis tool
            tools.append(Tool(
                name="Analysis",
                func=lambda x: f"Analysis of: {x}",
                description="Analyze data or information"
            ))
            
            # Create agent with tools
            if OPENAI_API_KEY:
                llm = OpenAI(temperature=0.7)
                from langchain.agents import create_react_agent
                
                # Create a prompt template
                prompt = PromptTemplate(
                    input_variables=["input", "agent_scratchpad"],
                    template="""You are an expert assistant. Use the tools to respond to the user's request.
                    
                    {tools}
                    
                    Use the following format:
                    
                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question
                    
                    Question: {input}
                    {agent_scratchpad}
                    """
                )
                
                # Create the agent
                agent = create_react_agent(llm, tools, prompt)
                
                # Create and add agent component
                from langchain.agents import AgentExecutor
                
                agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True
                )
                
                agent_component = AgentComponent(agent_executor)
                sim_env.add_component(agent_component)
        
        # Run simulation
        results = sim_env.run_simulation(mode=mode, input_data=input_data)
        
        # Format and return the response
        return jsonify(format_response(
            run_id=run_id,
            output=results,
            metadata={
                "simulation_type": "orchestration",
                "mode": mode,
                "input_length": len(input_data),
                "num_components": len(sim_env.components)
            }
        ))
        
    except Exception as e:
        logger.error(f"Error in orchestration_sim: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)