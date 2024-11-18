# isopro/__init__.py

"""
isopro: Intelligent Simulation Orchestration for LLMs

This package provides tools for creating, managing, and analyzing simulations
involving Large Language Models (LLMs), including reinforcement learning,
conversation simulations, and adversarial testing.
"""

__version__ = "0.1.6"

# Core components
from .environments.simulation_environment import SimulationEnvironment
from .environments.custom_environment import CustomEnvironment
from .environments.llm_orchestrator import LLMOrchestrator
from .agents.ai_agent import AI_Agent
from .base.base_component import BaseComponent
from .wrappers.simulation_wrapper import SimulationWrapper
from .rl.rl_environment import BaseRLEnvironment
from .rl.rl_agent import RLAgent
from .conversation_simulation import ConversationSimulator, ConversationEnvironment, ConversationAgent
from .adversarial_simulation import AdversarialSimulator, AdversarialEnvironment, AdversarialAgent
from .orchestration_simulation import LLaMAAgent, SubAgent, OrchestrationEnv, AI_AgentException, ComponentException, AI_Agent

# Workflow simulation components
from .workflow_simulation import (
    WorkflowSimulator,
    WorkflowEnvironment,
    WorkflowState,
    UIElement,
    UIElementDetector,
    MotionDetector,
    EpisodeMetrics,
    AgentConfig,
    VisualizationConfig,
    ValidationConfig,
    WorkflowAutomation
)

# Car RL components
from .car_simulator import CarRLEnvironment, LLMCarRLWrapper, CarVisualization

__all__ = [
    # Core components
    "LLaMAAgent", 
    "SubAgent", 
    "OrchestrationEnv", 
    "AI_AgentException", 
    "ComponentException", 
    "AI_Agent",
    "SimulationEnvironment",
    "CustomEnvironment",
    "LLMOrchestrator",
    "AI_Agent",
    "BaseComponent",
    "SimulationWrapper",
    "BaseRLEnvironment",
    "RLAgent",
    "ConversationSimulator",
    "ConversationEnvironment",
    "ConversationAgent",
    "AdversarialSimulator",
    "AdversarialEnvironment",
    "AdversarialAgent",
    
    # Workflow components
    "WorkflowSimulator",
    "WorkflowEnvironment",
    "WorkflowState",
    "UIElement",
    "UIElementDetector", 
    "MotionDetector",
    "EpisodeMetrics",
    "AgentConfig",
    "VisualizationConfig",
    "ValidationConfig",
    "WorkflowAutomation",
    
    # Car RL components
    "CarRLEnvironment",
    "LLMCarRLWrapper",
    "CarVisualization"
]