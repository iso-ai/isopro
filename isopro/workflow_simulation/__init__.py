"""
Workflow Simulator Package

A package for automating and learning UI workflows from video demonstrations.
Provides tools for training agents, validating workflows, and visualizing results.
"""

# Core components
from .workflow_simulator import WorkflowSimulator, EpisodeMetrics
from .workflow_environment import (
    WorkflowEnvironment,
    WorkflowState,
    UIElement,
    UIElementDetector,
    MotionDetector
)

# Configuration classes
from .agent_config import AgentConfig
from .workflow_visualizer import VisualizationConfig
from .workflow_validator import ValidationConfig

# Main automation
from .main import WorkflowAutomation

__version__ = "0.1.0"

__all__ = [
    # Core simulator and environment
    "WorkflowSimulator",
    "WorkflowEnvironment",
    
    # Environment components
    "WorkflowState",
    "UIElement",
    "UIElementDetector",
    "MotionDetector",
    
    # Metrics and tracking
    "EpisodeMetrics",
    
    # Configuration
    "AgentConfig",
    "VisualizationConfig",
    "ValidationConfig",
    
    # Main automation
    "WorkflowAutomation"
]
