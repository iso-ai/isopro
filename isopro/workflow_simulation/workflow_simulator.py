"""
WorkflowSimulator Module

Provides a simulator for workflow automation that handles training, evaluation,
and interaction between agents and the workflow environment.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import json
from datetime import datetime

from gymnasium import Env
from .workflow_agent import AgentConfig
from .workflow_visualizer import VisualizationConfig
from .workflow_validator import ValidationConfig
from .workflow_environment import WorkflowEnvironment
from .agent_config import AgentConfig

logger = logging.getLogger(__name__)

class EpisodeMetrics:
    """Tracks and analyzes episode metrics."""
    
    def __init__(self):
        self.rewards = []
        self.lengths = []
        self.success_rate = 0.0
        
    def add_episode(self, rewards: List[float], length: int, success: bool = False):
        """Add episode metrics."""
        self.rewards.append(sum(rewards))
        self.lengths.append(length)
        self.success_rate = (self.success_rate * len(self.rewards) + float(success)) / (len(self.rewards) + 1)
        
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of episodes."""
        return {
            'mean_reward': float(np.mean(self.rewards)),
            'std_reward': float(np.std(self.rewards)),
            'mean_length': float(np.mean(self.lengths)),
            'success_rate': self.success_rate,
            'total_episodes': len(self.rewards)
        }

class WorkflowSimulator(Env):
    """Simulator for workflow automation training and evaluation."""
    
    def __init__(
        self,
        video_path: str,
        agent_config: AgentConfig,
        viz_config: VisualizationConfig,
        validation_config: ValidationConfig,
        output_dir: str,
        anthropic_api_key: Optional[str] = None,
        max_episodes: int = 1000,
        eval_episodes: int = 10
    ):
        super().__init__()
        
        # Store configurations
        self.agent_config = agent_config
        self.viz_config = viz_config
        self.validation_config = validation_config
        self.output_dir = Path(output_dir)
        self.max_episodes = max_episodes
        self.eval_episodes = eval_episodes
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.env = WorkflowEnvironment(
            video_path=video_path,
            output_dir=str(self.output_dir / "env"),
            anthropic_api_key=anthropic_api_key,
            viz_enabled=viz_config.real_time_display
        )
        
        # Set spaces to match environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Initialize tracking
        self.current_episode = 0
        self.training_metrics = EpisodeMetrics()
        self.eval_metrics = EpisodeMetrics()
        self.best_reward = float('-inf')
        
        logger.info(f"Initialized WorkflowSimulator with output dir: {output_dir}")
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset simulator and environment."""
        self.current_episode += 1
        observation, info = self.env.reset(seed=seed)
        return self.convert_to_agent_input(observation), info
        
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action in environment."""
        env_action = self.convert_from_agent_output(action)
        observation, reward, terminated, truncated, info = self.env.step(env_action)
        
        agent_obs = self.convert_to_agent_input(observation)
        return agent_obs, reward, terminated, truncated, info
        
    def render(self):
        """Render current state."""
        return self.env.render()
        
    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
        
    def convert_to_agent_input(self, observation: Dict) -> Dict:
        """Convert environment observation to agent format."""
        return {
            'ui_elements': observation.get('ui_elements', []),
            'cursor_position': observation.get('cursor_pos', (0.0, 0.0)),
            'last_action': observation.get('last_action', ''),
            'progress': float(observation.get('progress', [0.0])[0])
        }
        
    def convert_from_agent_output(self, action: Dict) -> Dict:
        """Convert agent action to environment format."""
        return {
            'action_type': int(action.get('action_type', 0)),
            'target_element': np.array(action.get('target_element', [0, 0, 0, 0])),
            'parameters': {
                'text_input': str(action.get('text_input', '')),
                'drag_end': np.array(action.get('drag_end', [0, 0]))
            }
        }
        
    def train_agents(self) -> Dict[str, Any]:
        """Train agents on workflow demonstration."""
        logger.info("Starting agent training")
        
        episode_rewards = []
        best_episode_reward = float('-inf')
        
        for episode in range(self.max_episodes):
            # Run training episode
            episode_metrics = self._run_episode(training=True)
            episode_rewards.extend(episode_metrics['rewards'])
            
            # Track best performance
            episode_reward = sum(episode_metrics['rewards'])
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                self._save_checkpoint('best_model')
            
            # Log progress
            if (episode + 1) % 10 == 0:
                self._log_training_progress(episode + 1, episode_rewards[-10:])
            
            # Early stopping check
            if self._check_early_stopping(episode_rewards):
                logger.info("Early stopping criteria met")
                break
        
        # Save final model and results
        self._save_checkpoint('final_model')
        return self.training_metrics.get_summary()
        
    def evaluate_agents(self) -> Dict[str, Any]:
        """Evaluate trained agents."""
        logger.info("Starting agent evaluation")
        
        for episode in range(self.eval_episodes):
            # Run evaluation episode
            episode_metrics = self._run_episode(training=False)
            self.eval_metrics.add_episode(
                rewards=episode_metrics['rewards'],
                length=episode_metrics['length'],
                success=episode_metrics['success']
            )
            
            logger.info(f"Evaluation episode {episode + 1}/{self.eval_episodes} completed")
        
        # Save evaluation results
        results = self.eval_metrics.get_summary()
        self._save_results(results)
        
        return results
        
    def _run_episode(self, training: bool = True) -> Dict[str, Any]:
        """Run a single episode and return metrics."""
        observation, _ = self.reset()
        episode_rewards = []
        step_count = 0
        done = False
        
        while not done:
            # Get action (random for now - replace with actual policy)
            action = self.action_space.sample()
            
            # Take step in environment
            observation, reward, terminated, truncated, _ = self.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            step_count += 1
        
        metrics = {
            'rewards': episode_rewards,
            'length': step_count,
            'success': sum(episode_rewards) > self.agent_config.reward_threshold
        }
        
        if training:
            self.training_metrics.add_episode(**metrics)
        
        return metrics
        
    def _check_early_stopping(self, rewards: List[float], window: int = 100) -> bool:
        """Check if training should stop early."""
        if len(rewards) < window:
            return False
            
        recent_mean = np.mean(rewards[-window:])
        return recent_mean >= self.agent_config.reward_threshold
        
    def _log_training_progress(self, episode: int, recent_rewards: List[float]):
        """Log training progress."""
        mean_reward = np.mean(recent_rewards)
        logger.info(
            f"Episode {episode}/{self.max_episodes}, "
            f"Average Reward: {mean_reward:.2f}"
        )
        
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'episode': self.current_episode,
            'training_metrics': self.training_metrics.get_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = checkpoint_dir / f"{name}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        results_path = self.output_dir / "evaluation_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(
                {
                    'results': results,
                    'config': {
                        'max_episodes': self.max_episodes,
                        'eval_episodes': self.eval_episodes,
                        'agent_config': vars(self.agent_config),
                        'validation_config': vars(self.validation_config)
                    },
                    'timestamp': datetime.now().isoformat()
                },
                f,
                indent=2
            )
            
        logger.info(f"Saved evaluation results to {results_path}")