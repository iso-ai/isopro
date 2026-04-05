"""
ISOPro Core Framework

A comprehensive SDK for building and evaluating high-performance AI agents
using advanced techniques including Reflexion, Constitutional AI, DQN, and DPO.
"""

from .agent_framework import (
    AgentFramework, 
    AgentConfig, 
    AgentType, 
    ImprovementTechnique,
    EnhancedAgent
)

__all__ = [
    'AgentFramework',
    'AgentConfig',
    'AgentType', 
    'ImprovementTechnique',
    'EnhancedAgent'
]