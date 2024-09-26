import os
import json
from datetime import datetime
from dotenv import load_dotenv
import logging
import traceback
from langchain.agents import Tool, create_react_agent
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish
import json
from isopro.orchestration_simulation.orchestration_env import OrchestrationEnv
from isopro.orchestration_simulation.utils import setup_logging
from isopro.orchestration_simulation.components.base_component import BaseComponent

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file.")

# Set up logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"renewable_energy_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = setup_logging(log_file=log_file)

class ToolComponent(BaseComponent):
    def __init__(self, name, func, description):
        super().__init__(name)
        self.func = func
        self.description = description

    def run(self, input_data):
        logger.info(f"Running {self.name} with input: {input_data}")
        result = self.func(input_data)
        logger.info(f"{self.name} result: {result}")
        return result

# Create tool components
tech_research = ToolComponent("TechResearch", lambda x: "Latest advancements include improved efficiency in solar panels, advanced wind turbine designs, and breakthroughs in energy storage technologies.", "Research technological advancements")
economic_analysis = ToolComponent("EconomicAnalysis", lambda x: "Economic factors include decreasing costs of renewable technologies, increasing investment in clean energy, and the implementation of carbon pricing mechanisms.", "Analyze economic factors")
policy_review = ToolComponent("PolicyReview", lambda x: "Policy initiatives include renewable energy targets, feed-in tariffs, tax incentives for clean energy adoption, and stricter regulations on fossil fuel emissions.", "Review policy initiatives")

tools = [
    Tool(name=tech_research.name, func=tech_research.run, description=tech_research.description),
    Tool(name=economic_analysis.name, func=economic_analysis.run, description=economic_analysis.description),
    Tool(name=policy_review.name, func=policy_review.run, description=policy_review.description),
]

# Create agent
llm = OpenAI(temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""You are an expert in renewable energy analysis. Use the following tools to answer the question:

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

Answer: """
)

# Prepare the tool_names string
tool_names = ", ".join([tool.name for tool in tools])

# Create the agent
agent = create_react_agent(llm, tools, prompt)

class AgentComponent(BaseComponent):
    def __init__(self, agent, tools, max_iterations=5):
        super().__init__("AgentComponent")
        self.agent = agent
        self.tools = tools
        self.max_iterations = max_iterations
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, input_data):
        logger.info(f"AgentComponent running with input: {input_data}")
        try:
            result = self.agent_executor.invoke(
                {"input": input_data},
                {"max_iterations": self.max_iterations}
            )
            logger.info(f"AgentComponent raw result: {json.dumps(result, indent=2)}")
            
            final_result = {
                "result": result.get("output", "No output generated"),
                "intermediate_steps": [
                    {
                        "action": str(step[0]),
                        "observation": str(step[1])
                    } for step in result.get("intermediate_steps", [])
                ]
            }
            
            logger.info(f"AgentComponent processed result: {json.dumps(final_result, indent=2)}")
            return final_result
        except Exception as e:
            logger.error(f"Error in AgentComponent: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}


def save_output(result, mode):
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"result_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {mode} result to {filename}")

def main():
    # Create simulation environment
    sim_env = OrchestrationEnv()

    # Add agent component
    agent_component = AgentComponent(agent, tools)
    sim_env.add_component(agent_component)

    # Question
    question = "Analyze the current state of renewable energy adoption worldwide. Consider technological advancements, economic factors, and policy initiatives in your analysis. Then, based on this analysis, predict the most promising renewable energy source for widespread adoption in the next decade."

    # Run simulation in agent mode
    logger.info("Running agent simulation")
    agent_results = sim_env.run_simulation(mode='agent', input_data=question)
    logger.info(f"Agent results: {json.dumps(agent_results, indent=2)}")

    # Save output
    if agent_results:
        save_output(agent_results[0], "agent")
        print("\nSimulation Result:")
        print(json.dumps(agent_results[0], indent=2))
    else:
        logger.error("No results returned from simulation")
        print("Error: No results returned from simulation. Check the logs for details.")

if __name__ == "__main__":
    main()