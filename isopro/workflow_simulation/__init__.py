"""
Workflow Simulator Package

A package for automating and learning UI workflows from video demonstrations.
Provides tools for training agents, validating workflows, and visualizing results.
"""

# Core components — guard against stable_baselines3/protobuf breakage in
# environments with mismatched protobuf versions.
try:
    from .workflow_simulator import WorkflowSimulator, EpisodeMetrics
except Exception:  # noqa: BLE001
    WorkflowSimulator = None  # type: ignore[assignment,misc]
    EpisodeMetrics = None  # type: ignore[assignment,misc]
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
try:
    from .main import WorkflowAutomation
except Exception:  # noqa: BLE001
    WorkflowAutomation = None  # type: ignore[assignment,misc]

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
