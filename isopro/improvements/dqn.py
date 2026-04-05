"""
DQN (Deep Q-Network) Improvement Engine

Implements DQN-based reinforcement learning for continuous agent improvement
through action-value optimization.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import deque
import json

from .base_improvement import BaseImprovementEngine


@dataclass
class DQNConfig:
    """Configuration for DQN improvement engine."""
    state_dim: int = 128  # Dimension of state representation
    action_space: List[str] = field(default_factory=list)  # Available actions
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 32
    memory_size: int = 2000
    update_frequency: int = 10
    target_update_frequency: int = 100
    
    # Custom functions
    state_encoder: Optional[Callable] = None  # Encode input/output to state
    action_decoder: Optional[Callable] = None  # Decode action to modifications
    reward_function: Optional[Callable] = None  # Calculate reward
    q_network: Optional[Any] = None  # User-provided Q-network


class ReplayMemory:
    """Experience replay memory for DQN."""
    
    def __init__(self, capacity: int):
        """Initialize replay memory."""
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Save a transition."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of transitions."""
        if len(self.memory) < batch_size:
            return list(self.memory)
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def __len__(self) -> int:
        """Get current size of memory."""
        return len(self.memory)


class DQNEngine(BaseImprovementEngine):
    """
    DQN-based improvement engine that learns optimal enhancement strategies
    through reinforcement learning.
    """
    
    def _initialize_technique(self):
        """Initialize DQN-specific parameters."""
        # Load configuration
        dqn_params = self.agent_config.custom_parameters.get('dqn', {})
        
        # Initialize config with defaults or custom values
        self.config = DQNConfig(
            state_dim=dqn_params.get('state_dim', 128),
            action_space=dqn_params.get('action_space', self._get_default_actions()),
            learning_rate=dqn_params.get('learning_rate', 0.001),
            discount_factor=dqn_params.get('discount_factor', 0.95),
            epsilon=dqn_params.get('epsilon', 0.1),
            epsilon_decay=dqn_params.get('epsilon_decay', 0.995),
            epsilon_min=dqn_params.get('epsilon_min', 0.01),
            batch_size=dqn_params.get('batch_size', 32),
            memory_size=dqn_params.get('memory_size', 2000),
            update_frequency=dqn_params.get('update_frequency', 10),
            target_update_frequency=dqn_params.get('target_update_frequency', 100),
            state_encoder=dqn_params.get('state_encoder'),
            action_decoder=dqn_params.get('action_decoder'),
            reward_function=dqn_params.get('reward_function'),
            q_network=dqn_params.get('q_network')
        )
        
        # Initialize replay memory
        self.memory = ReplayMemory(self.config.memory_size)
        
        # Initialize Q-network if not provided
        if not self.config.q_network:
            self.q_network = self._create_default_q_network()
            self.target_network = self._create_default_q_network()
        else:
            self.q_network = self.config.q_network
            self.target_network = None  # User manages target network
        
        # Training state
        self.steps = 0
        self.episodes = 0
        self.current_epsilon = self.config.epsilon
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.action_counts: Dict[int, int] = {i: 0 for i in range(len(self.config.action_space))}
        
        self.logger.info(f"Initialized DQN with {len(self.config.action_space)} actions")
    
    def _get_default_actions(self) -> List[str]:
        """Get default action space."""
        return [
            "no_modification",
            "add_detail",
            "improve_clarity",
            "add_examples",
            "add_reasoning",
            "simplify",
            "add_structure",
            "add_confidence",
            "add_caveats",
            "focus_relevance"
        ]
    
    def _create_default_q_network(self) -> Dict[str, Any]:
        """
        Create a simple default Q-network representation.
        In production, users should provide their own neural network.
        """
        return {
            'type': 'tabular',
            'q_table': {},
            'default_value': 0.0
        }
    
    def enhance_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance input using learned strategies.
        
        Args:
            input_data: Original input data
            
        Returns:
            Enhanced input based on DQN policy
        """
        try:
            enhanced_input = input_data.copy()
            
            # Encode current state
            state = self._encode_state(input_data, {})
            
            # Select action based on current policy
            action = self._select_action(state, explore=False)
            
            # Apply action to input
            if action > 0:  # 0 is typically "no_modification"
                enhanced_input = self._apply_action_to_input(enhanced_input, action)
                
                enhanced_input['_dqn_metadata'] = {
                    'action_taken': self.config.action_space[action],
                    'action_id': action,
                    'exploration_rate': self.current_epsilon
                }
            
            self._record_improvement(input_data, enhanced_input, 'input', True)
            self.update_metrics(True, 0.7)
            
            return enhanced_input
            
        except Exception as e:
            self.logger.error(f"Failed to enhance input with DQN: {e}")
            self._record_improvement(input_data, input_data, 'input', False, {'error': str(e)})
            self.update_metrics(False)
            return input_data
    
    def enhance_output(self, output_data: Dict[str, Any], 
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance output using DQN policy and learn from the experience.
        
        Args:
            output_data: Original output data
            input_data: Corresponding input data
            
        Returns:
            Enhanced output based on DQN policy
        """
        try:
            # Encode current state
            state = self._encode_state(input_data, output_data)
            
            # Select action
            action = self._select_action(state, explore=True)
            self.action_counts[action] += 1
            
            # Apply action
            enhanced_output = self._apply_action_to_output(output_data, action)
            
            # Calculate reward
            reward = self._calculate_reward(output_data, enhanced_output, input_data)
            
            # Encode next state
            next_state = self._encode_state(input_data, enhanced_output)
            
            # Store transition
            self.memory.push(state, action, reward, next_state, False)
            
            # Update Q-network
            if self.steps % self.config.update_frequency == 0 and len(self.memory) >= self.config.batch_size:
                self._update_q_network()
            
            # Update target network
            if self.target_network and self.steps % self.config.target_update_frequency == 0:
                self._update_target_network()
            
            # Decay epsilon
            self.current_epsilon = max(
                self.config.epsilon_min,
                self.current_epsilon * self.config.epsilon_decay
            )
            
            self.steps += 1
            
            # Add metadata
            enhanced_output['_dqn_metadata'] = {
                'action_taken': self.config.action_space[action],
                'action_id': action,
                'reward': reward,
                'exploration_rate': self.current_epsilon,
                'step': self.steps
            }
            
            self._record_improvement(output_data, enhanced_output, 'output', True)
            self.update_metrics(True, reward)
            
            return enhanced_output
            
        except Exception as e:
            self.logger.error(f"Failed to enhance output with DQN: {e}")
            self._record_improvement(output_data, output_data, 'output', False, {'error': str(e)})
            self.update_metrics(False)
            return output_data
    
    def _encode_state(self, input_data: Dict[str, Any], 
                     output_data: Dict[str, Any]) -> np.ndarray:
        """Encode input/output pair into state representation."""
        if self.config.state_encoder:
            return self.config.state_encoder(input_data, output_data)
        
        # Default encoding: simple feature extraction
        features = []
        
        # Input features
        input_text = str(input_data.get('prompt', input_data.get('input', '')))
        features.extend([
            len(input_text),
            input_text.count(' '),
            input_text.count('?'),
            int('example' in input_text.lower()),
            int('explain' in input_text.lower()),
            int('create' in input_text.lower())
        ])
        
        # Output features
        if output_data:
            output_text = str(output_data.get('response', output_data.get('output', '')))
            features.extend([
                len(output_text),
                output_text.count(' '),
                output_text.count('.'),
                int('confidence' in output_data),
                int('reasoning' in output_data),
                int('error' in output_data)
            ])
        else:
            features.extend([0] * 6)
        
        # Pad or truncate to state_dim
        if len(features) < self.config.state_dim:
            features.extend([0] * (self.config.state_dim - len(features)))
        else:
            features = features[:self.config.state_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.current_epsilon:
            # Explore: random action
            return np.random.randint(0, len(self.config.action_space))
        
        # Exploit: best action according to Q-network
        if self.config.q_network:
            if hasattr(self.config.q_network, 'predict'):
                # Neural network interface
                q_values = self.config.q_network.predict(state.reshape(1, -1))[0]
                return int(np.argmax(q_values))
            elif isinstance(self.config.q_network, dict) and self.config.q_network['type'] == 'tabular':
                # Tabular Q-learning
                state_key = self._state_to_key(state)
                q_table = self.config.q_network['q_table']
                if state_key in q_table:
                    return int(np.argmax(q_table[state_key]))
        
        # Default: no modification
        return 0
    
    def _apply_action_to_input(self, input_data: Dict[str, Any], action: int) -> Dict[str, Any]:
        """Apply selected action to input data."""
        if self.config.action_decoder:
            return self.config.action_decoder(input_data, action, 'input')
        
        # Default action application
        enhanced = input_data.copy()
        action_name = self.config.action_space[action]
        
        if action_name == "add_detail" and 'prompt' in enhanced:
            enhanced['prompt'] += " Please provide detailed information."
        elif action_name == "add_examples" and 'prompt' in enhanced:
            enhanced['prompt'] += " Include specific examples."
        elif action_name == "focus_relevance" and 'prompt' in enhanced:
            enhanced['prompt'] += " Focus on the most relevant aspects."
        
        return enhanced
    
    def _apply_action_to_output(self, output_data: Dict[str, Any], action: int) -> Dict[str, Any]:
        """Apply selected action to output data."""
        if self.config.action_decoder:
            return self.config.action_decoder(output_data, action, 'output')
        
        # Default action application
        enhanced = output_data.copy()
        action_name = self.config.action_space[action]
        
        # Simple heuristic modifications
        if action_name == "no_modification":
            pass
        elif action_name == "add_confidence" and 'confidence' not in enhanced:
            enhanced['confidence'] = 0.8
        elif action_name == "add_reasoning" and 'reasoning' not in enhanced:
            enhanced['reasoning'] = "Based on the provided information."
        elif action_name == "add_structure":
            enhanced['structured'] = True
        
        enhanced['_action_applied'] = action_name
        return enhanced
    
    def _calculate_reward(self, original_output: Dict[str, Any],
                        enhanced_output: Dict[str, Any],
                        input_data: Dict[str, Any]) -> float:
        """Calculate reward for the action taken."""
        if self.config.reward_function:
            return self.config.reward_function(original_output, enhanced_output, input_data)
        
        # Default reward calculation
        reward = 0.0
        
        # Positive rewards
        if 'confidence' in enhanced_output and 'confidence' not in original_output:
            reward += 0.2
        if 'reasoning' in enhanced_output and 'reasoning' not in original_output:
            reward += 0.3
        if '_action_applied' in enhanced_output and enhanced_output['_action_applied'] != "no_modification":
            reward += 0.1
        
        # Negative rewards
        if 'error' in enhanced_output:
            reward -= 0.5
        
        # Clip reward
        return max(-1.0, min(1.0, reward))
    
    def _update_q_network(self):
        """Update Q-network using experience replay."""
        if len(self.memory) < self.config.batch_size:
            return
        
        batch = self.memory.sample(self.config.batch_size)
        
        if hasattr(self.config.q_network, 'train_on_batch'):
            # Neural network training
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            rewards = np.array([t[2] for t in batch])
            next_states = np.array([t[3] for t in batch])
            dones = np.array([t[4] for t in batch])
            
            # Calculate targets
            if self.target_network:
                next_q_values = self.target_network.predict(next_states)
            else:
                next_q_values = self.config.q_network.predict(next_states)
            
            targets = rewards + self.config.discount_factor * np.max(next_q_values, axis=1) * (1 - dones)
            
            # Train network
            self.config.q_network.train_on_batch(states, actions, targets)
            
        elif isinstance(self.config.q_network, dict) and self.config.q_network['type'] == 'tabular':
            # Tabular Q-learning update
            q_table = self.config.q_network['q_table']
            
            for state, action, reward, next_state, done in batch:
                state_key = self._state_to_key(state)
                next_state_key = self._state_to_key(next_state)
                
                # Initialize Q-values if needed
                if state_key not in q_table:
                    q_table[state_key] = np.zeros(len(self.config.action_space))
                if next_state_key not in q_table:
                    q_table[next_state_key] = np.zeros(len(self.config.action_space))
                
                # Q-learning update
                current_q = q_table[state_key][action]
                next_q = np.max(q_table[next_state_key]) if not done else 0
                
                q_table[state_key][action] = current_q + self.config.learning_rate * (
                    reward + self.config.discount_factor * next_q - current_q
                )
    
    def _update_target_network(self):
        """Update target network with current Q-network weights."""
        if hasattr(self.config.q_network, 'get_weights') and hasattr(self.target_network, 'set_weights'):
            self.target_network.set_weights(self.config.q_network.get_weights())
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to string key for tabular Q-learning."""
        # Discretize continuous values
        discretized = np.round(state, decimals=1)
        return json.dumps(discretized.tolist())
    
    def train_batch(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train DQN on a batch of experiences.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            Training metrics
        """
        total_reward = 0.0
        
        for exp in experiences:
            state = self._encode_state(exp['input'], exp['original_output'])
            enhanced_output = exp['enhanced_output']
            next_state = self._encode_state(exp['input'], enhanced_output)
            
            # Determine action taken
            action = self._infer_action(exp['original_output'], enhanced_output)
            
            # Calculate reward
            reward = self._calculate_reward(
                exp['original_output'], 
                enhanced_output, 
                exp['input']
            )
            
            # Store experience
            self.memory.push(state, action, reward, next_state, exp.get('done', False))
            total_reward += reward
        
        # Perform multiple updates
        update_steps = min(len(experiences), 10)
        for _ in range(update_steps):
            if len(self.memory) >= self.config.batch_size:
                self._update_q_network()
        
        return {
            'average_reward': total_reward / len(experiences),
            'memory_size': len(self.memory),
            'epsilon': self.current_epsilon
        }
    
    def _infer_action(self, original: Dict[str, Any], enhanced: Dict[str, Any]) -> int:
        """Infer which action was taken based on output differences."""
        if '_action_applied' in enhanced:
            action_name = enhanced['_action_applied']
            if action_name in self.config.action_space:
                return self.config.action_space.index(action_name)
        
        # Default inference based on changes
        if enhanced == original:
            return 0  # no_modification
        
        # Simple heuristics
        if 'confidence' in enhanced and 'confidence' not in original:
            return self.config.action_space.index('add_confidence') if 'add_confidence' in self.config.action_space else 0
        
        return 0  # Default
    
    def save_q_network(self, path: str):
        """Save Q-network to file."""
        if hasattr(self.config.q_network, 'save'):
            self.config.q_network.save(path)
        elif isinstance(self.config.q_network, dict):
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.config.q_network, f)
        
        self.logger.info(f"Saved Q-network to {path}")
    
    def load_q_network(self, path: str):
        """Load Q-network from file."""
        if hasattr(self.config.q_network, 'load'):
            self.config.q_network.load(path)
        elif isinstance(self.config.q_network, dict):
            import pickle
            with open(path, 'rb') as f:
                loaded = pickle.load(f)
                self.config.q_network.update(loaded)
        
        self.logger.info(f"Loaded Q-network from {path}")
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about action usage."""
        total_actions = sum(self.action_counts.values())
        
        action_stats = {}
        for action_id, count in self.action_counts.items():
            action_name = self.config.action_space[action_id]
            action_stats[action_name] = {
                'count': count,
                'percentage': (count / total_actions * 100) if total_actions > 0 else 0
            }
        
        return {
            'total_actions': total_actions,
            'action_distribution': action_stats,
            'most_used_action': max(self.action_counts, key=self.action_counts.get) if total_actions > 0 else None,
            'exploration_rate': self.current_epsilon
        }
    
    def _get_technique_config(self) -> Dict[str, Any]:
        """Get DQN-specific configuration."""
        return {
            'technique': 'DQN',
            'state_dim': self.config.state_dim,
            'action_space_size': len(self.config.action_space),
            'learning_rate': self.config.learning_rate,
            'discount_factor': self.config.discount_factor,
            'current_epsilon': self.current_epsilon,
            'memory_size': len(self.memory),
            'total_steps': self.steps,
            'q_network_type': (
                self.config.q_network['type'] 
                if isinstance(self.config.q_network, dict) 
                else 'custom'
            )
        }