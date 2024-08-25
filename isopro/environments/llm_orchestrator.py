"""
LLM Orchestrator for managing and executing LLM components in various modes.
"""

import logging
import heapq
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Optional, Callable
from ..base.base_component import BaseComponent

logger = logging.getLogger(__name__)

class ComponentException(Exception):
    """Custom exception for component-related errors."""
    pass

class LLMOrchestrator:
    """
    LLMOrchestrator manages and executes LLM components in various modes:
    sequential, parallel, or priority-based node execution.
    """

    def __init__(self):
        """Initialize the LLMOrchestrator with an empty list of components."""
        self.components: List[BaseComponent] = []
        self.priority_function: Optional[Callable[[BaseComponent, Any], int]] = None

    def add_component(self, component: BaseComponent) -> None:
        """
        Add a component to the orchestrator.

        Args:
            component (BaseComponent): The component to be added.

        Raises:
            ValueError: If the component is None or not an instance of BaseComponent.
        """
        if component is None:
            raise ValueError("Cannot add None as a component")
        if not isinstance(component, BaseComponent):
            raise ValueError(f"Only BaseComponent instances can be added, got {type(component)}")
        self.components.append(component)

    def set_priority_function(self, priority_func: Callable[[BaseComponent, Any], int]) -> None:
        """
        Set the priority function for node-based execution.

        Args:
            priority_func (Callable[[BaseComponent, Any], int]): A function that takes a component
                and input data, and returns an integer priority value.
        """
        self.priority_function = priority_func

    def run_orchestration(self, mode: str = 'sequence', input_data: Optional[Any] = None) -> List[Any]:
        """
        Run the orchestration in the specified mode.

        Args:
            mode (str): The execution mode ('sequence', 'parallel', or 'node').
            input_data (Any, optional): The initial input data for the components.

        Returns:
            List[Any]: The results from all components.

        Raises:
            ValueError: If an invalid execution mode is specified.
        """
        if not self.components:
            logger.warning("No components to run")
            return []

        if mode == 'sequence':
            return self._run_in_sequence(input_data)
        elif mode == 'parallel':
            return self._run_in_parallel(input_data)
        elif mode == 'node':
            return self._run_as_node(input_data)
        else:
            raise ValueError("Invalid execution mode")

    def _run_in_sequence(self, input_data: Any) -> List[Any]:
        """
        Run components sequentially, passing the output of each as input to the next.

        Args:
            input_data (Any): The initial input data for the first component.

        Returns:
            List[Any]: The results from all components.
        """
        logger.info("Running in sequence mode")
        results = []
        current_input = input_data

        for component in self.components:
            try:
                result = self._run_component(component, current_input)
                results.append(result)
                current_input = result  # Use the output as input for the next component
            except ComponentException as e:
                logger.error(f"Error: {e}")
                results.append(str(e))

        return results

    def _run_in_parallel(self, input_data: Any) -> List[Any]:
        """
        Run components in parallel, providing the same input to all components.

        Args:
            input_data (Any): The input data for all components.

        Returns:
            List[Any]: The results from all components.
        """
        logger.info("Running in parallel mode")
        results = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._run_component, component, input_data) 
                       for component in self.components]
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except ComponentException as e:
                    logger.error(f"Error: {e}")
                    results.append(str(e))

        return results

    def _run_as_node(self, input_data: Any) -> List[Any]:
        """
        Run components in priority-based node mode.

        The priority is defined either by the LLM using reasoning on the best path
        of solving the problem or designated by the user through the priority_function.

        Args:
            input_data (Any): The input data for all components.

        Returns:
            List[Any]: The results from all components, ordered by priority.
        """
        logger.info("Running in node mode (priority-based)")
        results = []
        
        if self.priority_function is None:
            logger.warning("No priority function set. Using default priority (0) for all components.")
            priority_queue = [(0, i, component) for i, component in enumerate(self.components)]
        else:
            priority_queue = [(self.priority_function(component, input_data), i, component) 
                              for i, component in enumerate(self.components)]
        
        heapq.heapify(priority_queue)
        
        while priority_queue:
            priority, _, component = heapq.heappop(priority_queue)
            logger.info(f"Running component {component} with priority {priority}")
            try:
                result = self._run_component(component, input_data)
                results.append(result)
                
                # If the component changes the priority, we need to update the queue
                if self.priority_function:
                    new_priority = self.priority_function(component, result)
                    if new_priority != priority:
                        heapq.heappush(priority_queue, (new_priority, len(results), component))
                        logger.info(f"Updated priority for component {component}: {priority} -> {new_priority}")
                
            except ComponentException as e:
                logger.error(f"Error: {e}")
                results.append(str(e))

        return results

    def _run_component(self, component: BaseComponent, input_data: Any) -> Any:
        """
        Run a single component with the given input data.

        Args:
            component (BaseComponent): The component to run.
            input_data (Any): The input data for the component.

        Returns:
            Any: The result of running the component.

        Raises:
            ComponentException: If the component doesn't have a callable 'run' method.
        """
        if not hasattr(component, 'run') or not callable(component.run):
            raise ComponentException(f"Component {component} does not have a callable 'run' method")
        return component.run(input_data)