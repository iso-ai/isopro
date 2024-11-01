"""
Configuration module for workflow automation agents.

Defines the configuration parameters and validation for training and evaluating
workflow automation agents.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AgentConfig:
    """Configuration for workflow automation agents."""
    
    learning_rate: float = 3e-4
    pretrain_epochs: int = 10
    use_demonstration: bool = True
    use_reasoning: bool = True
    reward_threshold: float = 0.8
    batch_size: int = 64
    max_gradient_norm: float = 1.0
    update_frequency: int = 1
    buffer_size: int = 10000
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            AgentConfig instance with specified parameters
        """
        return cls(
            learning_rate=config_dict.get('learning_rate', 3e-4),
            pretrain_epochs=config_dict.get('pretrain_epochs', 10),
            use_demonstration=config_dict.get('use_demonstration', True),
            use_reasoning=config_dict.get('use_reasoning', True),
            reward_threshold=config_dict.get('reward_threshold', 0.8),
            batch_size=config_dict.get('batch_size', 64),
            max_gradient_norm=config_dict.get('max_gradient_norm', 1.0),
            update_frequency=config_dict.get('update_frequency', 1),
            buffer_size=config_dict.get('buffer_size', 10000)
        )
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.pretrain_epochs < 0:
            raise ValueError("Pretrain epochs must be non-negative")
        if not 0 <= self.reward_threshold <= 1:
            raise ValueError("Reward threshold must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_gradient_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if self.update_frequency <= 0:
            raise ValueError("Update frequency must be positive")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")