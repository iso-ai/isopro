import logging
from concurrent.futures import ThreadPoolExecutor
import heapq
from typing import List
from tqdm import tqdm
from isopro.orchestration_simulation.components.base_component import BaseComponent

logger = logging.getLogger(__name__)

class OrchestrationEnv:
    def __init__(self):
        self.components: List[BaseComponent] = []

    def add_component(self, component: BaseComponent):
        if not isinstance(component, BaseComponent):
            raise ValueError(f"Only BaseComponent instances can be added, got {type(component)}")
        self.components.append(component)
        logger.info(f"Added component: {component.name}")

    def run_simulation(self, mode='agent', input_data=None):
        if not self.components:
            logger.warning("No components to run")
            return

        logger.info(f"Starting simulation in {mode} mode")
        if mode == 'agent':
            return self.run_agent_mode(input_data)
        elif mode == 'sequence':
            return self.run_in_sequence(input_data)
        elif mode == 'parallel':
            return self.run_in_parallel(input_data)
        elif mode == 'node':
            return self.run_as_node(input_data)
        else:
            raise ValueError("Invalid execution mode")

    def run_agent_mode(self, input_data):
        logger.info("Running in agent mode")
        agent_component = next((c for c in self.components if c.name == "AgentComponent"), None)
        if not agent_component:
            raise ValueError("AgentComponent not found")
        
        with tqdm(total=1, desc="Agent Progress") as pbar:
            result = agent_component.run(input_data)
            pbar.update(1)
        
        logger.info("Agent mode completed")
        return [result]  # Wrap the result in a list for consistency with other modes

    def run_in_sequence(self, input_data):
        logger.info("Running in sequence mode")
        results = []
        current_input = input_data
        with tqdm(total=len(self.components), desc="Sequence Progress") as pbar:
            for component in self.components:
                try:
                    logger.info(f"Running component: {component.name}")
                    result = component.run(current_input)
                    results.append(result)
                    if 'result' in result:
                        current_input = result['result']
                    else:
                        logger.warning(f"Component {component.name} did not return a 'result'. Using original input for next component.")
                except Exception as e:
                    logger.error(f"Error in component {component.name}: {e}")
                    results.append({"error": str(e)})
                finally:
                    pbar.update(1)
        
        logger.info("Sequence mode completed")
        return results

    def run_in_parallel(self, input_data):
        logger.info("Running in parallel mode")
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(component.run, input_data) for component in self.components]
            with tqdm(total=len(futures), desc="Parallel Progress") as pbar:
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        results.append({"error": str(e)})
                    finally:
                        pbar.update(1)
        
        logger.info("Parallel mode completed")
        return results

    def run_as_node(self, input_data):
        logger.info("Running in node mode (priority-based)")
        results = []
        priority_queue = [(i, component) for i, component in enumerate(self.components)]
        heapq.heapify(priority_queue)
        
        with tqdm(total=len(self.components), desc="Node Progress") as pbar:
            while priority_queue:
                _, component = heapq.heappop(priority_queue)
                try:
                    logger.info(f"Running component: {component.name}")
                    result = component.run(input_data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in component {component.name}: {e}")
                    results.append({"error": str(e)})
                finally:
                    pbar.update(1)
        
        logger.info("Node mode completed")
        return results