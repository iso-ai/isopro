"""
DPO (Direct Preference Optimization) Improvement Engine

Implements Direct Preference Optimization for aligning agent outputs
with human preferences without explicit reward modeling.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .base_improvement import BaseImprovementEngine


@dataclass
class PreferencePair:
    """A pair of outputs with preference annotation."""
    input_data: Dict[str, Any]
    preferred_output: Dict[str, Any]
    rejected_output: Dict[str, Any]
    preference_strength: float = 1.0  # Strength of preference (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DPOConfig:
    """Configuration for DPO improvement engine."""
    beta: float = 0.1  # KL regularization coefficient
    learning_rate: float = 0.0001
    batch_size: int = 16
    preference_buffer_size: int = 1000
    min_preferences_to_update: int = 50
    update_frequency: int = 10
    
    # Preference collection
    implicit_preference_detection: bool = True
    preference_threshold: float = 0.2  # Min score difference for preference
    
    # Custom functions
    preference_scorer: Optional[Callable] = None  # Score outputs for preferences
    policy_updater: Optional[Callable] = None  # Update policy with preferences
    output_sampler: Optional[Callable] = None  # Sample alternative outputs
    llm_interface: Optional[Any] = None  # LLM for generating alternatives


class PreferenceBuffer:
    """Buffer for storing preference pairs."""
    
    def __init__(self, max_size: int):
        """Initialize preference buffer."""
        self.max_size = max_size
        self.preferences: List[PreferencePair] = []
        self.preference_stats = defaultdict(int)
    
    def add_preference(self, preference: PreferencePair):
        """Add a preference pair to the buffer."""
        self.preferences.append(preference)
        
        # Update statistics
        self.preference_stats['total'] += 1
        self.preference_stats[f'strength_{int(preference.preference_strength * 10)}'] += 1
        
        # Maintain size limit
        if len(self.preferences) > self.max_size:
            self.preferences = self.preferences[-self.max_size:]
    
    def sample_batch(self, batch_size: int) -> List[PreferencePair]:
        """Sample a batch of preferences."""
        if len(self.preferences) < batch_size:
            return self.preferences.copy()
        
        indices = np.random.choice(len(self.preferences), batch_size, replace=False)
        return [self.preferences[i] for i in indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return dict(self.preference_stats)
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.preferences)


class DPOEngine(BaseImprovementEngine):
    """
    DPO improvement engine that learns from preferences to align outputs
    with desired behavior patterns.
    """
    
    def _initialize_technique(self):
        """Initialize DPO-specific parameters."""
        # Load configuration
        dpo_params = self.agent_config.custom_parameters.get('dpo', {})
        
        self.config = DPOConfig(
            beta=dpo_params.get('beta', 0.1),
            learning_rate=dpo_params.get('learning_rate', 0.0001),
            batch_size=dpo_params.get('batch_size', 16),
            preference_buffer_size=dpo_params.get('preference_buffer_size', 1000),
            min_preferences_to_update=dpo_params.get('min_preferences_to_update', 50),
            update_frequency=dpo_params.get('update_frequency', 10),
            implicit_preference_detection=dpo_params.get('implicit_preference_detection', True),
            preference_threshold=dpo_params.get('preference_threshold', 0.2),
            preference_scorer=dpo_params.get('preference_scorer'),
            policy_updater=dpo_params.get('policy_updater'),
            output_sampler=dpo_params.get('output_sampler'),
            llm_interface=dpo_params.get('llm_interface')
        )
        
        # Initialize preference buffer
        self.preference_buffer = PreferenceBuffer(self.config.preference_buffer_size)
        
        # Initialize policy parameters (simplified representation)
        self.policy_params = {
            'preference_patterns': {},
            'feature_weights': defaultdict(float),
            'update_count': 0
        }
        
        # Tracking
        self.generation_count = 0
        self.preference_collection_count = 0
        
        self.logger.info("Initialized DPO improvement engine")
    
    def enhance_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance input based on learned preferences.
        
        Args:
            input_data: Original input data
            
        Returns:
            Enhanced input incorporating preference patterns
        """
        try:
            enhanced_input = input_data.copy()
            
            # Extract features from input
            input_features = self._extract_input_features(input_data)
            
            # Apply learned preference patterns to input
            preference_hints = self._get_preference_hints(input_features)
            
            if preference_hints:
                # Add preference context
                if 'context' in enhanced_input and isinstance(enhanced_input['context'], dict):
                    enhanced_input['context']['preference_hints'] = preference_hints
                else:
                    enhanced_input['preference_context'] = preference_hints
                
                enhanced_input['_dpo_metadata'] = {
                    'hints_applied': len(preference_hints),
                    'policy_version': self.policy_params['update_count']
                }
            
            self._record_improvement(input_data, enhanced_input, 'input', True)
            self.update_metrics(True, 0.7)
            
            return enhanced_input
            
        except Exception as e:
            self.logger.error(f"Failed to enhance input with DPO: {e}")
            self._record_improvement(input_data, input_data, 'input', False, {'error': str(e)})
            self.update_metrics(False)
            return input_data
    
    def enhance_output(self, output_data: Dict[str, Any], 
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance output using learned preferences and collect new preferences.
        
        Args:
            output_data: Original output data
            input_data: Corresponding input data
            
        Returns:
            Enhanced output aligned with preferences
        """
        try:
            self.generation_count += 1
            
            # Generate alternative outputs if possible
            alternatives = self._generate_alternatives(output_data, input_data)
            
            if alternatives:
                # Score all outputs (original + alternatives)
                all_outputs = [output_data] + alternatives
                scores = self._score_outputs(all_outputs, input_data)
                
                # Select best output based on learned preferences
                best_idx = np.argmax(scores)
                enhanced_output = all_outputs[best_idx].copy()
                
                # Collect preferences if there's a clear winner
                if self.config.implicit_preference_detection:
                    self._collect_implicit_preferences(
                        all_outputs, scores, input_data
                    )
                
                # Add metadata
                enhanced_output['_dpo_metadata'] = {
                    'alternatives_generated': len(alternatives),
                    'score': float(scores[best_idx]),
                    'score_difference': float(scores[best_idx] - scores[0]),
                    'selected_over_original': best_idx > 0
                }
            else:
                # No alternatives, use original
                enhanced_output = output_data.copy()
            
            # Update policy if enough preferences collected
            if (self.generation_count % self.config.update_frequency == 0 and 
                len(self.preference_buffer) >= self.config.min_preferences_to_update):
                self._update_policy()
            
            self._record_improvement(output_data, enhanced_output, 'output', True)
            self.update_metrics(True, 0.8)
            
            return enhanced_output
            
        except Exception as e:
            self.logger.error(f"Failed to enhance output with DPO: {e}")
            self._record_improvement(output_data, output_data, 'output', False, {'error': str(e)})
            self.update_metrics(False)
            return output_data
    
    def _extract_input_features(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from input for preference matching."""
        features = {}
        
        # Text-based features
        text = str(input_data.get('prompt', input_data.get('input', '')))
        
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['question'] = 1.0 if '?' in text else 0.0
        features['command'] = 1.0 if any(cmd in text.lower() for cmd in ['create', 'write', 'generate']) else 0.0
        features['analysis'] = 1.0 if any(word in text.lower() for word in ['analyze', 'explain', 'describe']) else 0.0
        
        # Task type features
        if 'task_type' in input_data:
            features[f'task_{input_data["task_type"]}'] = 1.0
        
        return features
    
    def _get_preference_hints(self, input_features: Dict[str, float]) -> List[str]:
        """Get preference hints based on input features."""
        hints = []
        
        # Apply learned patterns
        for pattern_name, pattern_data in self.policy_params['preference_patterns'].items():
            if self._matches_pattern(input_features, pattern_data):
                hints.extend(pattern_data.get('hints', []))
        
        # Apply feature-based preferences
        weighted_features = []
        for feature, value in input_features.items():
            weight = self.policy_params['feature_weights'].get(feature, 0)
            if weight > 0.1:  # Significant positive preference
                weighted_features.append((feature, weight * value))
        
        # Convert to hints
        weighted_features.sort(key=lambda x: x[1], reverse=True)
        for feature, _ in weighted_features[:3]:  # Top 3 features
            if feature == 'question':
                hints.append("Provide clear and direct answers")
            elif feature == 'command':
                hints.append("Focus on actionable results")
            elif feature == 'analysis':
                hints.append("Include detailed analysis and reasoning")
        
        return hints[:5]  # Limit hints
    
    def _matches_pattern(self, features: Dict[str, float], 
                        pattern: Dict[str, Any]) -> bool:
        """Check if features match a preference pattern."""
        required_features = pattern.get('required_features', {})
        
        for feature, min_value in required_features.items():
            if features.get(feature, 0) < min_value:
                return False
        
        return True
    
    def _generate_alternatives(self, output_data: Dict[str, Any], 
                             input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative outputs for comparison."""
        if self.config.output_sampler:
            # Use custom sampler
            return self.config.output_sampler(output_data, input_data)
        
        if not self.config.llm_interface:
            # No LLM available, can't generate alternatives
            return []
        
        alternatives = []
        
        # Generate variations with different strategies
        strategies = [
            "Make the response more concise",
            "Add more detail and examples",
            "Focus on practical applications",
            "Emphasize theoretical understanding"
        ]
        
        for strategy in strategies[:2]:  # Limit to 2 alternatives for efficiency
            try:
                # Create variation prompt
                variation_prompt = {
                    'task': 'create_variation',
                    'original_input': input_data,
                    'original_output': output_data,
                    'strategy': strategy,
                    'instruction': f"Create a variation of the output that: {strategy}"
                }
                
                # Generate variation
                response = self.config.llm_interface.generate(variation_prompt)
                
                if response and isinstance(response, dict):
                    alternatives.append(response)
                    
            except Exception as e:
                self.logger.debug(f"Failed to generate alternative with strategy '{strategy}': {e}")
        
        return alternatives
    
    def _score_outputs(self, outputs: List[Dict[str, Any]], 
                      input_data: Dict[str, Any]) -> np.ndarray:
        """Score outputs based on learned preferences."""
        if self.config.preference_scorer:
            # Use custom scorer
            return np.array([
                self.config.preference_scorer(output, input_data) 
                for output in outputs
            ])
        
        # Default scoring based on learned features
        scores = []
        
        for output in outputs:
            score = 0.5  # Base score
            
            # Extract output features
            features = self._extract_output_features(output)
            
            # Apply learned weights
            for feature, value in features.items():
                weight = self.policy_params['feature_weights'].get(feature, 0)
                score += weight * value
            
            # Normalize
            score = max(0, min(1, score))
            scores.append(score)
        
        return np.array(scores)
    
    def _extract_output_features(self, output_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from output for scoring."""
        features = {}
        
        # Basic features
        output_text = str(output_data.get('response', output_data.get('output', '')))
        
        features['length'] = min(len(output_text) / 500, 1.0)  # Normalized
        features['has_structure'] = 1.0 if '\n' in output_text or '•' in output_text else 0.0
        features['has_reasoning'] = 1.0 if 'reasoning' in output_data else 0.0
        features['has_confidence'] = 1.0 if 'confidence' in output_data else 0.0
        features['has_examples'] = 1.0 if 'example' in output_text.lower() else 0.0
        
        # Quality indicators
        features['no_errors'] = 0.0 if 'error' in output_data else 1.0
        features['complete'] = 1.0 if len(output_text) > 50 else 0.5
        
        return features
    
    def _collect_implicit_preferences(self, outputs: List[Dict[str, Any]], 
                                    scores: np.ndarray, 
                                    input_data: Dict[str, Any]):
        """Collect implicit preferences from scored outputs."""
        # Find pairs with significant score differences
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                score_diff = abs(scores[i] - scores[j])
                
                if score_diff >= self.config.preference_threshold:
                    # Create preference pair
                    if scores[i] > scores[j]:
                        preferred, rejected = outputs[i], outputs[j]
                    else:
                        preferred, rejected = outputs[j], outputs[i]
                    
                    preference = PreferencePair(
                        input_data=input_data,
                        preferred_output=preferred,
                        rejected_output=rejected,
                        preference_strength=min(score_diff / 0.5, 1.0),  # Normalize
                        metadata={
                            'score_difference': float(score_diff),
                            'collection_method': 'implicit'
                        }
                    )
                    
                    self.preference_buffer.add_preference(preference)
                    self.preference_collection_count += 1
    
    def add_explicit_preference(self, input_data: Dict[str, Any],
                              preferred_output: Dict[str, Any],
                              rejected_output: Dict[str, Any],
                              strength: float = 1.0):
        """
        Add an explicit preference from external feedback.
        
        Args:
            input_data: Input that generated the outputs
            preferred_output: The preferred output
            rejected_output: The rejected output
            strength: Strength of preference (0-1)
        """
        preference = PreferencePair(
            input_data=input_data,
            preferred_output=preferred_output,
            rejected_output=rejected_output,
            preference_strength=strength,
            metadata={'collection_method': 'explicit'}
        )
        
        self.preference_buffer.add_preference(preference)
        self.preference_collection_count += 1
        
        self.logger.info(f"Added explicit preference (strength: {strength})")
    
    def _update_policy(self):
        """Update policy based on collected preferences."""
        if self.config.policy_updater:
            # Use custom updater
            self.config.policy_updater(self.policy_params, self.preference_buffer)
            return
        
        # Default policy update
        batch = self.preference_buffer.sample_batch(self.config.batch_size)
        
        if not batch:
            return
        
        # Analyze preferences to update feature weights
        feature_importance = defaultdict(float)
        pattern_candidates = []
        
        for pref in batch:
            # Extract features from preferred and rejected
            preferred_features = self._extract_output_features(pref.preferred_output)
            rejected_features = self._extract_output_features(pref.rejected_output)
            
            # Update feature importance based on differences
            for feature in preferred_features:
                diff = preferred_features[feature] - rejected_features.get(feature, 0)
                feature_importance[feature] += diff * pref.preference_strength
            
            # Collect pattern candidates
            if pref.preference_strength > 0.7:  # Strong preferences
                pattern_candidates.append({
                    'input_features': self._extract_input_features(pref.input_data),
                    'preferred_features': preferred_features,
                    'strength': pref.preference_strength
                })
        
        # Update feature weights with momentum
        momentum = 0.9
        for feature, importance in feature_importance.items():
            current_weight = self.policy_params['feature_weights'][feature]
            new_weight = momentum * current_weight + (1 - momentum) * self.config.learning_rate * importance
            self.policy_params['feature_weights'][feature] = new_weight
        
        # Update patterns (simplified clustering)
        if pattern_candidates:
            self._update_preference_patterns(pattern_candidates)
        
        self.policy_params['update_count'] += 1
        
        self.logger.info(f"Updated policy (iteration {self.policy_params['update_count']})")
    
    def _update_preference_patterns(self, candidates: List[Dict[str, Any]]):
        """Update preference patterns from candidates."""
        # Simple pattern extraction (in production, use clustering)
        new_patterns = {}
        
        # Group by similar input features
        for candidate in candidates[:10]:  # Limit for efficiency
            pattern_key = self._get_pattern_key(candidate['input_features'])
            
            if pattern_key not in new_patterns:
                new_patterns[pattern_key] = {
                    'required_features': {},
                    'preferred_output_features': {},
                    'hints': [],
                    'support': 0
                }
            
            pattern = new_patterns[pattern_key]
            pattern['support'] += candidate['strength']
            
            # Update preferred features
            for feature, value in candidate['preferred_features'].items():
                if value > 0.5:  # Significant features
                    if feature not in pattern['preferred_output_features']:
                        pattern['preferred_output_features'][feature] = 0
                    pattern['preferred_output_features'][feature] += value
        
        # Merge with existing patterns
        for key, pattern in new_patterns.items():
            if pattern['support'] > 1.0:  # Sufficient support
                # Normalize and create hints
                pattern['hints'] = self._pattern_to_hints(pattern)
                self.policy_params['preference_patterns'][key] = pattern
    
    def _get_pattern_key(self, features: Dict[str, float]) -> str:
        """Generate pattern key from features."""
        # Simple discretization of key features
        key_parts = []
        
        if features.get('question', 0) > 0.5:
            key_parts.append('question')
        if features.get('command', 0) > 0.5:
            key_parts.append('command')
        if features.get('analysis', 0) > 0.5:
            key_parts.append('analysis')
        
        return '_'.join(key_parts) if key_parts else 'general'
    
    def _pattern_to_hints(self, pattern: Dict[str, Any]) -> List[str]:
        """Convert pattern to actionable hints."""
        hints = []
        
        pref_features = pattern['preferred_output_features']
        
        if pref_features.get('has_structure', 0) > 1:
            hints.append("Use structured formatting")
        if pref_features.get('has_examples', 0) > 1:
            hints.append("Include concrete examples")
        if pref_features.get('has_reasoning', 0) > 1:
            hints.append("Provide clear reasoning")
        
        return hints
    
    def get_preference_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected preferences."""
        buffer_stats = self.preference_buffer.get_statistics()
        
        # Analyze feature weights
        top_positive_features = sorted(
            [(f, w) for f, w in self.policy_params['feature_weights'].items() if w > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_negative_features = sorted(
            [(f, w) for f, w in self.policy_params['feature_weights'].items() if w < 0],
            key=lambda x: x[1]
        )[:5]
        
        return {
            'total_preferences': len(self.preference_buffer),
            'preference_collection_rate': (
                self.preference_collection_count / self.generation_count 
                if self.generation_count > 0 else 0
            ),
            'buffer_statistics': buffer_stats,
            'policy_updates': self.policy_params['update_count'],
            'top_positive_features': top_positive_features,
            'top_negative_features': top_negative_features,
            'pattern_count': len(self.policy_params['preference_patterns'])
        }
    
    def export_policy(self) -> Dict[str, Any]:
        """Export current policy for analysis or transfer."""
        return {
            'feature_weights': dict(self.policy_params['feature_weights']),
            'preference_patterns': self.policy_params['preference_patterns'],
            'update_count': self.policy_params['update_count'],
            'statistics': self.get_preference_statistics()
        }
    
    def import_policy(self, policy_data: Dict[str, Any]):
        """Import policy from exported data."""
        if 'feature_weights' in policy_data:
            self.policy_params['feature_weights'].update(policy_data['feature_weights'])
        
        if 'preference_patterns' in policy_data:
            self.policy_params['preference_patterns'].update(policy_data['preference_patterns'])
        
        if 'update_count' in policy_data:
            self.policy_params['update_count'] = policy_data['update_count']
        
        self.logger.info("Imported policy data")
    
    def _get_technique_config(self) -> Dict[str, Any]:
        """Get DPO-specific configuration."""
        return {
            'technique': 'DPO',
            'beta': self.config.beta,
            'learning_rate': self.config.learning_rate,
            'preference_buffer_size': self.config.preference_buffer_size,
            'current_buffer_size': len(self.preference_buffer),
            'policy_updates': self.policy_params['update_count'],
            'implicit_detection': self.config.implicit_preference_detection,
            'feature_count': len(self.policy_params['feature_weights']),
            'pattern_count': len(self.policy_params['preference_patterns'])
        }