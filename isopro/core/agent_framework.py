"""
Core Agent Framework

Provides a unified interface for creating, managing, and improving AI agents
with support for various enhancement techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum
import uuid

from ..base.base_component import BaseComponent
from ..utils.logging_utils import setup_logger


class AgentType(Enum):
    """Supported agent types in the framework."""
    LLM = "llm"
    RL = "reinforcement_learning"
    HYBRID = "hybrid"
    MULTIMODAL = "multimodal"


class ImprovementTechnique(Enum):
    """Available improvement techniques."""
    REFLEXION = "reflexion"
    CONSTITUTIONAL_AI = "constitutional_ai"
    DQN = "dqn"
    DPO = "dpo"
    SELF_CONSISTENCY = "self_consistency"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class AgentConfig:
    """Configuration for agent creation and behavior."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.LLM
    model_name: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4096
    temperature: float = 0.7
    improvement_techniques: List[ImprovementTechnique] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=list)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.agent_type, AgentType):
            raise ValueError(f"agent_type must be an AgentType enum, got {type(self.agent_type)}")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


class EnhancedAgent(BaseComponent):
    """
    Enhanced agent with built-in improvement capabilities.
    
    This agent extends the base component architecture with support for
    various improvement techniques and comprehensive evaluation.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize enhanced agent with configuration.
        
        Args:
            config: Agent configuration specifying behavior and capabilities
        """
        super().__init__()
        self.config = config
        self.logger = setup_logger(f"agent_{config.agent_id}")
        
        # Core agent state
        self._conversation_history: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, float] = {}
        self._improvement_data: Dict[str, Any] = {}
        
        # Initialize improvement techniques
        self._improvement_engines = {}
        self._initialize_improvement_techniques()
        
        self.logger.info(f"Initialized enhanced agent {config.agent_id} with type {config.agent_type}")
    
    def _initialize_improvement_techniques(self):
        """Initialize configured improvement techniques."""
        for technique in self.config.improvement_techniques:
            try:
                engine = self._create_improvement_engine(technique)
                self._improvement_engines[technique] = engine
                self.logger.info(f"Initialized {technique.value} improvement engine")
            except Exception as e:
                self.logger.error(f"Failed to initialize {technique.value}: {e}")
    
    def _create_improvement_engine(self, technique: ImprovementTechnique):
        """Create improvement engine for specific technique."""
        # Import here to avoid circular dependencies
        from ..improvements.reflexion import ReflexionEngine
        from ..improvements.constitutional_ai import ConstitutionalAIEngine
        from ..improvements.dqn import DQNEngine
        from ..improvements.dpo import DPOEngine
        
        engine_map = {
            ImprovementTechnique.REFLEXION: ReflexionEngine,
            ImprovementTechnique.CONSTITUTIONAL_AI: ConstitutionalAIEngine,
            ImprovementTechnique.DQN: DQNEngine,
            ImprovementTechnique.DPO: DPOEngine,
        }
        
        engine_class = engine_map.get(technique)
        if not engine_class:
            raise ValueError(f"Unsupported improvement technique: {technique}")
        
        return engine_class(self.config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent with input data and return enhanced response.
        
        Args:
            input_data: Input data containing prompt, context, and other parameters
            
        Returns:
            Enhanced response with metadata and improvement information
        """
        try:
            # Pre-process input using improvement techniques
            enhanced_input = self._apply_input_improvements(input_data)
            
            # Core agent execution
            base_response = self._execute_core_logic(enhanced_input)
            
            # Post-process output using improvement techniques
            enhanced_response = self._apply_output_improvements(base_response, enhanced_input)
            
            # Update conversation history
            self._update_conversation_history(enhanced_input, enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            return {
                'error': str(e),
                'agent_id': self.config.agent_id,
                'status': 'failed'
            }
    
    def _apply_input_improvements(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvement techniques to input data."""
        enhanced_input = input_data.copy()
        
        for technique, engine in self._improvement_engines.items():
            try:
                enhanced_input = engine.enhance_input(enhanced_input)
                self.logger.debug(f"Applied {technique.value} to input")
            except Exception as e:
                self.logger.warning(f"Failed to apply {technique.value} to input: {e}")
        
        return enhanced_input
    
    def _apply_output_improvements(self, response: Dict[str, Any], 
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvement techniques to output response."""
        enhanced_response = response.copy()
        
        for technique, engine in self._improvement_engines.items():
            try:
                enhanced_response = engine.enhance_output(enhanced_response, input_data)
                self.logger.debug(f"Applied {technique.value} to output")
            except Exception as e:
                self.logger.warning(f"Failed to apply {technique.value} to output: {e}")
        
        return enhanced_response
    
    @abstractmethod
    def _execute_core_logic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core agent logic. Must be implemented by subclasses."""
        pass
    
    def _update_conversation_history(self, input_data: Dict[str, Any], 
                                   response: Dict[str, Any]):
        """Update conversation history with latest interaction."""
        interaction = {
            'timestamp': self._get_timestamp(),
            'input': input_data,
            'output': response,
            'agent_id': self.config.agent_id
        }
        
        self._conversation_history.append(interaction)
        
        # Keep history within reasonable bounds
        max_history = self.config.custom_parameters.get('max_history', 100)
        if len(self._conversation_history) > max_history:
            self._conversation_history = self._conversation_history[-max_history:]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics for the agent."""
        return self._performance_metrics.copy()
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update agent performance metrics."""
        self._performance_metrics.update(metrics)
        self.logger.info(f"Updated performance metrics: {metrics}")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self._conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()
        self.logger.info("Cleared conversation history")
    
    def get_improvement_data(self) -> Dict[str, Any]:
        """Get data from improvement techniques."""
        improvement_data = {}
        
        for technique, engine in self._improvement_engines.items():
            try:
                improvement_data[technique.value] = engine.get_improvement_data()
            except Exception as e:
                self.logger.warning(f"Failed to get improvement data for {technique.value}: {e}")
        
        return improvement_data
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()


class AgentFramework:
    """
    Main framework for managing enhanced AI agents.
    
    Provides high-level interface for creating, managing, and improving agents
    with various enhancement techniques.
    """
    
    def __init__(self):
        """Initialize the agent framework."""
        self.logger = setup_logger("agent_framework")
        self._agents: Dict[str, EnhancedAgent] = {}
        self._agent_registry: Dict[str, type] = {}
        
        # Register built-in agent types
        self._register_builtin_agents()
        
        self.logger.info("Agent framework initialized")
    
    def _register_builtin_agents(self):
        """Register built-in agent implementations."""
        # Import here to avoid circular dependencies
        from ..agents.enhanced_llm_agent import EnhancedLLMAgent
        from ..agents.enhanced_rl_agent import EnhancedRLAgent
        from ..agents.enhanced_hybrid_agent import EnhancedHybridAgent
        
        self.register_agent_type(AgentType.LLM, EnhancedLLMAgent)
        self.register_agent_type(AgentType.RL, EnhancedRLAgent)
        self.register_agent_type(AgentType.HYBRID, EnhancedHybridAgent)
    
    def register_agent_type(self, agent_type: AgentType, agent_class: type):
        """
        Register a new agent type with the framework.
        
        Args:
            agent_type: Type of agent being registered
            agent_class: Class implementing the agent
        """
        if not issubclass(agent_class, EnhancedAgent):
            raise ValueError("Agent class must inherit from EnhancedAgent")
        
        self._agent_registry[agent_type] = agent_class
        self.logger.info(f"Registered agent type {agent_type.value}")
    
    def create_agent(self, config: AgentConfig) -> EnhancedAgent:
        """
        Create a new enhanced agent with specified configuration.
        
        Args:
            config: Agent configuration
            
        Returns:
            Created enhanced agent instance
        """
        agent_class = self._agent_registry.get(config.agent_type)
        if not agent_class:
            raise ValueError(f"Unsupported agent type: {config.agent_type}")
        
        agent = agent_class(config)
        self._agents[config.agent_id] = agent
        
        self.logger.info(f"Created agent {config.agent_id} of type {config.agent_type.value}")
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[EnhancedAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self._agents.keys())
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent from framework.
        
        Args:
            agent_id: ID of agent to remove
            
        Returns:
            True if agent was removed, False if not found
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            self.logger.info(f"Removed agent {agent_id}")
            return True
        return False
    
    def batch_create_agents(self, configs: List[AgentConfig]) -> List[EnhancedAgent]:
        """
        Create multiple agents from a list of configurations.
        
        Args:
            configs: List of agent configurations
            
        Returns:
            List of created agent instances
        """
        agents = []
        for config in configs:
            try:
                agent = self.create_agent(config)
                agents.append(agent)
            except Exception as e:
                self.logger.error(f"Failed to create agent {config.agent_id}: {e}")
        
        return agents
    
    def get_framework_stats(self) -> Dict[str, Any]:
        """Get framework statistics and status."""
        agent_types = {}
        for agent in self._agents.values():
            agent_type = agent.config.agent_type.value
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        return {
            'total_agents': len(self._agents),
            'agent_types': agent_types,
            'registered_types': [t.value for t in self._agent_registry.keys()],
            'framework_version': "1.0.0"
        }