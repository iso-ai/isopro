# main.py

import os
import logging
import time
import psutil
from .orchestration_env import OrchestrationEnv
from .components.llama_agent import LLaMAAgent
from .components.analysis_agent import AnalysisAgent
from .components.writing_agent import WritingAgent
from .evaluation import evaluate_results, measure_coherence

# Set up logging
logging.basicConfig(filename='logs/orchestration.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('output', exist_ok=True)

def run_simulation(env, mode, task):
    logger.info(f"Running in {mode} mode")
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # in MB
    
    result = env.run_simulation(mode=mode, input_data={'task': task, 'run_order': 'first'})
    
    final_memory = process.memory_info().rss / 1024 / 1024  # in MB
    memory_usage = final_memory - initial_memory
    execution_time = time.time() - start_time
    
    coherence = measure_coherence([r['result'] for r in result if 'result' in r])
    
    return result, execution_time, memory_usage, coherence

def save_output(mode, result):
    with open(f'output/{mode}_output.txt', 'w') as f:
        f.write(f"Mode: {mode}\n\n")
        for i, r in enumerate(result):
            f.write(f"Agent {i+1} output:\n")
            f.write(str(r.get('result', 'No result')) + '\n\n')

def main():
    env = OrchestrationEnv()

    # Add agents
    env.add_component(LLaMAAgent("Research", "conduct thorough research on the impact of artificial intelligence on job markets in the next decade"))
    env.add_component(AnalysisAgent("Analysis"))
    env.add_component(WritingAgent("Writing"))

    task = "Prepare a comprehensive report on the impact of artificial intelligence on job markets in the next decade."

    modes = ['parallel', 'sequence', 'node']
    results = {}
    evaluations = {}

    for mode in modes:
        result, execution_time, memory_usage, coherence = run_simulation(env, mode, task)
        results[mode] = result
        evaluations[mode] = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'coherence': coherence
        }
        save_output(mode, result)

        logger.info(f"{mode.capitalize()} mode results:")
        for i, r in enumerate(result):
            logger.info(f"Agent {i+1} output:\n{r.get('result', 'No result')}\n")

    best_mode = evaluate_results(evaluations)
    logger.info(f"\nBest mode for this task: {best_mode}")

if __name__ == "__main__":
    main()