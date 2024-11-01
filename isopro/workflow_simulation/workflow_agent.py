"""
Workflow Agent Module

This module implements the WorkflowAgent class for learning and replicating UI workflows
using reinforcement learning and IsoZero reasoning capabilities.
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ..agents.ai_agent import AI_Agent
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm
from dataclasses import dataclass
import gymnasium as gym
from isozero import ClaudeAgent, QuestionAnswerer

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for WorkflowAgent."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    pretrain_epochs: int = 10
    use_demonstration: bool = True
    use_reasoning: bool = True

class WorkflowPolicy(nn.Module):
    """Custom policy network for learning UI workflows."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        
        # Input dimensions
        self.ui_element_dim = 128
        self.cursor_dim = 32
        self.context_dim = 64
        
        # UI element encoder
        self.ui_encoder = nn.Sequential(
            nn.Linear(4, 64),  # bbox features
            nn.ReLU(),
            nn.Linear(64, self.ui_element_dim),
            nn.ReLU()
        )
        
        # Cursor position encoder
        self.cursor_encoder = nn.Sequential(
            nn.Linear(2, self.cursor_dim),
            nn.ReLU()
        )
        
        # Context encoder for workflow state
        self.context_encoder = nn.Sequential(
            nn.Linear(observation_space.shape[0] - 6, self.context_dim),  # remaining features
            nn.ReLU()
        )
        
        # Attention mechanism for UI elements
        self.attention = nn.MultiheadAttention(
            embed_dim=self.ui_element_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Action predictor
        combined_dim = self.ui_element_dim + self.cursor_dim + self.context_dim
        self.action_predictor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.shape[0])
        )
        
        # Value function
        self.value_predictor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(
        self,
        ui_elements: torch.Tensor,
        cursor_pos: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the policy network."""
        # Encode UI elements
        ui_features = self.ui_encoder(ui_elements)
        
        # Apply attention to UI elements
        ui_features, _ = self.attention(ui_features, ui_features, ui_features)
        ui_features = torch.max(ui_features, dim=1)[0]  # max pooling
        
        # Encode cursor and context
        cursor_features = self.cursor_encoder(cursor_pos)
        context_features = self.context_encoder(context)
        
        # Combine features
        combined = torch.cat([ui_features, cursor_features, context_features], dim=1)
        
        # Predict actions and value
        action_logits = self.action_predictor(combined)
        value = self.value_predictor(combined)
        
        return action_logits, value

class WorkflowProgressCallback(BaseCallback):
    """Callback for tracking training progress."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.pbar = None
        
    def _on_training_start(self):
        """Initialize progress bar."""
        self.pbar = tqdm(total=self.locals['total_timesteps'],
                        desc="Training workflow agent")
        
    def _on_step(self) -> bool:
        """Update progress bar."""
        if self.pbar:
            self.pbar.update(1)
        return True
        
    def _on_training_end(self):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()
            self.pbar = None

class WorkflowAgent(AI_Agent):
    """Agent for learning and replicating UI workflows."""
    
    def __init__(
        self,
        name: str,
        env: Any,
        config: AgentConfig = None,
        anthropic_api_key: Optional[str] = None
    ):
        """Initialize the workflow agent.
        
        Args:
            name: Agent identifier
            env: The workflow environment
            config: Agent configuration
            anthropic_api_key: Optional API key for Claude agent
        """
        super().__init__(name)
        self.env = env
        self.config = config or AgentConfig()
        
        # Initialize policy
        policy_kwargs = {
            'policy_class': WorkflowPolicy,
            'features_extractor_class': None
        }
        
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            max_grad_norm=self.config.max_grad_norm,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        # Initialize IsoZero components if reasoning is enabled
        if self.config.use_reasoning and anthropic_api_key:
            self.claude_agent = ClaudeAgent(api_key=anthropic_api_key)
            self.qa_system = QuestionAnswerer(self.claude_agent)
            logger.info("Initialized IsoZero reasoning system")
            
        # Storage for demonstrations and reasoning
        self.demonstration_data = []
        self.reasoning_cache = {}
        
        logger.info(f"Initialized WorkflowAgent: {name}")
        
    def store_demonstration(
        self,
        states: List[Dict[str, Any]],
        actions: List[Dict[str, Any]]
    ):
        """Store demonstration data for imitation learning."""
        self.demonstration_data.extend(zip(states, actions))
        logger.info(f"Stored {len(states)} demonstration steps")
        
    def train(
        self,
        total_timesteps: int = 10000,
        callback: Optional[BaseCallback] = None
    ):
        """Train the agent using both imitation learning and RL.
        
        Args:
            total_timesteps: Number of environment steps for training
            callback: Optional callback for tracking progress
        """
        # First, pretrain on demonstrations if available
        if self.config.use_demonstration and self.demonstration_data:
            self._pretrain_on_demonstrations()
            
        # Setup training progress callback
        callbacks = [WorkflowProgressCallback()]
        if callback:
            callbacks.append(callback)
            
        # Train with RL
        logger.info(f"Training {self.name} with RL for {total_timesteps} steps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
    def _pretrain_on_demonstrations(self):
        """Pretrain the policy using demonstration data."""
        logger.info("Starting pretraining on demonstrations")
        
        pbar = tqdm(
            total=self.config.pretrain_epochs,
            desc="Pretraining on demonstrations"
        )
        
        for epoch in range(self.config.pretrain_epochs):
            total_loss = 0
            
            for state, action in self.demonstration_data:
                # Get policy prediction
                obs_tensor = self._preprocess_observation(state)
                action_tensor = torch.FloatTensor(action)
                
                policy_output = self.model.policy(obs_tensor)
                
                # Calculate loss
                action_loss = self._calculate_action_loss(
                    policy_output.actions,
                    action_tensor
                )
                
                total_loss += action_loss.item()
                
            avg_loss = total_loss / len(self.demonstration_data)
            logger.debug(f"Pretrain Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
            pbar.update(1)
            
        pbar.close()
        logger.info("Completed pretraining on demonstrations")
        
    def _preprocess_observation(
        self,
        observation: Dict[str, Any]
    ) -> torch.Tensor:
        """Preprocess observation for policy network."""
        # Extract and normalize UI elements
        ui_elements = torch.FloatTensor([
            elem['bbox'] for elem in observation['ui_elements']
        ])
        
        # Extract and normalize cursor position
        cursor_pos = torch.FloatTensor(observation['cursor_pos'])
        
        # Extract additional context features
        context = torch.FloatTensor([
            observation['sequence_progress']
        ])
        
        return {
            'ui_elements': ui_elements,
            'cursor_pos': cursor_pos,
            'context': context
        }
        
    def _calculate_action_loss(
        self,
        predicted_action: torch.Tensor,
        target_action: torch.Tensor
    ) -> torch.Tensor:
        """Calculate loss between predicted and target actions."""
        # MSE loss for continuous actions
        return torch.nn.functional.mse_loss(predicted_action, target_action)
        
    def predict(
        self,
        observation: Dict[str, Any],
        state: Optional[Any] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict next action based on observation."""
        # Apply reasoning if enabled
        if self.config.use_reasoning:
            observation = self._enhance_observation_with_reasoning(observation)
            
        # Get action from policy
        action, state = self.model.predict(
            observation,
            state=state,
            deterministic=deterministic
        )
        
        return action, state
    
    def _enhance_observation_with_reasoning(
        self,
        observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance observation with IsoZero reasoning for behavior replication."""
        if not hasattr(self, 'qa_system'):
            return observation
        
        # Create cache key from observation
        cache_key = str(observation)
    
        if cache_key in self.reasoning_cache:
            return {**observation, 'reasoning': self.reasoning_cache[cache_key]}
    
        # More specific questions for workflow understanding
        questions = [
            # Understand UI element relationships
            f"Given the UI elements {observation['ui_elements']}, what is their functional relationship to each other?",
        
            # Understand previous action context
            f"Based on the last action '{observation['last_action']}', what was likely the user's intention?",
        
            # Identify action prerequisites
            "What conditions need to be true before performing the next action?",
        
            # Predict next logical step
            f"Given the workflow progress is {observation['sequence_progress']}, what would be the next logical step in this workflow?",
        
            # Understand element states
            "Which UI elements are interactive in the current state and what interactions are they designed for?",
        
            # Validate action sequence
            "Does this sequence of actions align with common UI patterns and best practices?",
        
            # Identify potential errors
            "Are there any potential errors or invalid states that should be avoided in the next action?"
        ]
    
        # Create detailed context document for reasoning
        context = f"""
        Current UI State Analysis:
        1. Active UI Elements: {observation['ui_elements']}
        2. Current Cursor Position: {observation['cursor_pos']}
        3. Most Recent Action: {observation['last_action']}
        4. Workflow Progress: {observation['sequence_progress']}
        5. Element States: {self._get_element_states(observation)}
        6. Action History: {self._get_action_history()}
        7. Known Workflow Patterns: {self._get_workflow_patterns()}
        """
    
        # Get reasoning from IsoZero
        doc_pairs = [(q, context) for q in questions]
        reasoning = self.qa_system.answer_questions(doc_pairs)
    
        # Extract actionable insights from reasoning
        enhanced_reasoning = {
            'element_relationships': reasoning[questions[0]]['solution'],
            'user_intention': reasoning[questions[1]]['solution'],
            'prerequisites': reasoning[questions[2]]['solution'],
            'next_step': reasoning[questions[3]]['solution'],
            'interactive_elements': reasoning[questions[4]]['solution'],
            'pattern_validation': reasoning[questions[5]]['solution'],
            'potential_errors': reasoning[questions[6]]['solution'],
            'reasoning_steps': [
                step for q in questions 
                for step in reasoning[q]['reasoning']
            ]
        }
    
        # Cache the enhanced reasoning
        self.reasoning_cache[cache_key] = enhanced_reasoning
    
        # Add reasoning to observation
        enhanced_obs = {
            **observation,
            'reasoning': enhanced_reasoning,
            'suggested_action': self._extract_suggested_action(enhanced_reasoning)
        }
    
        return enhanced_obs

    def _get_element_states(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed states of UI elements."""
        element_states = {}
        for element in observation['ui_elements']:
            element_states[element['id']] = {
                'type': element['type'],
                'interactive': element.get('interactive', True),
                'state': element.get('state', 'default'),
                'enabled': element.get('enabled', True),
                'visible': element.get('visible', True)
            }
        return element_states

    def _get_action_history(self) -> List[str]:
        """Get recent action history for context."""
        return self.env.get_recent_actions() if hasattr(self.env, 'get_recent_actions') else []

    def _get_workflow_patterns(self) -> List[str]:
        """Get known workflow patterns for this type of interface."""
        return [
            "Form submission patterns",
            "Navigation patterns",
            "Selection patterns",
            "Confirmation patterns",
            "Error handling patterns"
        ]

    def _extract_suggested_action(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Extract concrete action suggestion from reasoning."""
        next_step = reasoning['next_step']
        return {
            'action_type': self._parse_action_type(next_step),
            'target_element': self._parse_target_element(next_step),
            'confidence': self._calculate_action_confidence(reasoning)
        }