"""
Base Improvement Engine

Abstract base class for all agent improvement techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

from ..utils.logging_utils import setup_logger


@dataclass
class ImprovementMetrics:
    """Metrics for tracking improvement technique performance."""
    technique_name: str
    total_applications: int = 0
    successful_applications: int = 0
    average_improvement_score: float = 0.0
    error_count: int = 0
    last_updated: Optional[str] = None


class BaseImprovementEngine(ABC):
    """
    Abstract base class for all improvement techniques.
    
    This class defines the interface that all improvement engines must implement
    to integrate with the agent framework.
    """
    
    def __init__(self, agent_config: 'AgentConfig'):
        """
        Initialize improvement engine.
        
        Args:
            agent_config: Configuration of the agent this engine will improve
        """
        self.agent_config = agent_config
        self.logger = setup_logger(f"{self.__class__.__name__}_{agent_config.agent_id}")
        self.metrics = ImprovementMetrics(technique_name=self.__class__.__name__)
        self._improvement_history: List[Dict[str, Any]] = []
        
        # Initialize technique-specific parameters
        self._initialize_technique()
        
        self.logger.info(f"Initialized {self.__class__.__name__} for agent {agent_config.agent_id}")
    
    @abstractmethod
    def _initialize_technique(self):
        """Initialize technique-specific parameters and state."""
        pass
    
    @abstractmethod
    def enhance_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply improvement technique to input data.
        
        Args:
            input_data: Original input data
            
        Returns:
            Enhanced input data
        """
        pass
    
    @abstractmethod
    def enhance_output(self, output_data: Dict[str, Any], 
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply improvement technique to output data.
        
        Args:
            output_data: Original output data
            input_data: Corresponding input data
            
        Returns:
            Enhanced output data
        """
        pass
    
    def update_metrics(self, success: bool, improvement_score: float = 0.0):
        """
        Update metrics for this improvement technique.
        
        Args:
            success: Whether the improvement application was successful
            improvement_score: Quantitative measure of improvement (0-1 scale)
        """
        self.metrics.total_applications += 1
        
        if success:
            self.metrics.successful_applications += 1
            # Update running average of improvement score
            current_avg = self.metrics.average_improvement_score
            total_successful = self.metrics.successful_applications
            self.metrics.average_improvement_score = (
                (current_avg * (total_successful - 1) + improvement_score) / total_successful
            )
        else:
            self.metrics.error_count += 1
        
        self.metrics.last_updated = self._get_timestamp()
        
        self.logger.debug(f"Updated metrics: success={success}, score={improvement_score}")
    
    def get_improvement_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data about this improvement technique.
        
        Returns:
            Dictionary containing metrics, history, and other relevant data
        """
        return {
            'metrics': {
                'technique_name': self.metrics.technique_name,
                'total_applications': self.metrics.total_applications,
                'successful_applications': self.metrics.successful_applications,
                'success_rate': self._calculate_success_rate(),
                'average_improvement_score': self.metrics.average_improvement_score,
                'error_count': self.metrics.error_count,
                'last_updated': self.metrics.last_updated
            },
            'recent_history': self._improvement_history[-10:],  # Last 10 improvements
            'configuration': self._get_technique_config()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.metrics.total_applications == 0:
            return 0.0
        return (self.metrics.successful_applications / self.metrics.total_applications) * 100
    
    def _get_technique_config(self) -> Dict[str, Any]:
        """
        Get technique-specific configuration.
        Subclasses should override to provide their specific config.
        """
        return {
            'agent_config_id': self.agent_config.agent_id,
            'base_config': 'BaseImprovementEngine'
        }
    
    def _record_improvement(self, input_data: Dict[str, Any], 
                          output_data: Dict[str, Any], 
                          enhancement_type: str,
                          success: bool,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Record an improvement attempt for analysis.
        
        Args:
            input_data: Input that was enhanced
            output_data: Result of enhancement
            enhancement_type: Type of enhancement (e.g., 'input', 'output')
            success: Whether enhancement was successful
            metadata: Additional metadata about the improvement
        """
        record = {
            'timestamp': self._get_timestamp(),
            'enhancement_type': enhancement_type,
            'success': success,
            'input_sample': str(input_data)[:200],  # Sample for debugging
            'output_sample': str(output_data)[:200],
            'metadata': metadata or {}
        }
        
        self._improvement_history.append(record)
        
        # Keep history within reasonable bounds
        max_history = 1000
        if len(self._improvement_history) > max_history:
            self._improvement_history = self._improvement_history[-max_history:]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def reset_metrics(self):
        """Reset all metrics and history."""
        self.metrics = ImprovementMetrics(technique_name=self.__class__.__name__)
        self._improvement_history.clear()
        self.logger.info("Reset improvement metrics and history")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the improvement engine.
        
        Returns:
            Health status information including performance indicators
        """
        success_rate = self._calculate_success_rate()
        
        # Determine health based on success rate and error frequency
        if success_rate >= 90 and self.metrics.error_count < 5:
            health = "excellent"
        elif success_rate >= 75 and self.metrics.error_count < 10:
            health = "good"
        elif success_rate >= 50 and self.metrics.error_count < 20:
            health = "fair"
        else:
            health = "poor"
        
        return {
            'health_status': health,
            'success_rate': success_rate,
            'total_applications': self.metrics.total_applications,
            'error_count': self.metrics.error_count,
            'average_improvement': self.metrics.average_improvement_score,
            'last_updated': self.metrics.last_updated,
            'recommendations': self._get_health_recommendations(health)
        }
    
    def _get_health_recommendations(self, health: str) -> List[str]:
        """Get recommendations based on health status."""
        recommendations = []
        
        if health == "poor":
            recommendations.extend([
                "Consider reviewing technique configuration",
                "Check for compatibility with current agent type",
                "Review recent error logs for patterns"
            ])
        elif health == "fair":
            recommendations.extend([
                "Monitor performance trends",
                "Consider parameter tuning"
            ])
        elif health == "good":
            recommendations.append("Performance is good, continue monitoring")
        else:  # excellent
            recommendations.append("Performance is excellent, consider sharing configuration")
        
        return recommendations