"""
Reflexion Improvement Engine

Implements the Reflexion technique for improving agent performance through
self-reflection and iterative refinement.

Based on: "Reflexion: Language Agents with Verbal Reinforcement Learning"
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Protocol
import json
import re
from dataclasses import dataclass, field
from abc import abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .base_improvement import BaseImprovementEngine


class LLMInterface(Protocol):
    """
    Protocol for LLM interfaces that users must implement.
    This allows complete flexibility in model choice and orchestration.
    """
    
    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Dictionary containing prompt data (format depends on implementation)
            
        Returns:
            Response dictionary (format depends on implementation)
        """
        ...
    
    async def generate_async(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Optional async generation method."""
        ...


@dataclass
class ReflexionConfig:
    """Configuration for Reflexion technique."""
    max_reflections: int = 3
    reflection_threshold: float = 0.7  # Minimum score to trigger reflection
    use_external_feedback: bool = True
    memory_size: int = 50  # Number of past reflections to remember
    enable_async: bool = True
    batch_size: int = 10
    evaluation_function: Optional[Callable[[Dict[str, Any], Dict[str, Any]], float]] = None
    llm_interface: Optional[LLMInterface] = None  # User-provided LLM interface
    reflection_prompt_builder: Optional[Callable] = None  # Custom prompt builder
    response_parser: Optional[Callable] = None  # Custom response parser
    
    
class ReflexionMemory:
    """
    Manages reflection memory with efficient search capabilities.
    """
    
    def __init__(self, max_size: int = 50):
        """
        Initialize reflection memory.
        
        Args:
            max_size: Maximum number of reflections to store
        """
        self.max_size = max_size
        self.reflections: List[Dict[str, Any]] = []
        self._index: Dict[str, List[int]] = {}  # Keyword index for fast search
        
    def add_reflection(self, reflection: Dict[str, Any]):
        """Add a reflection to memory with indexing."""
        reflection_id = len(self.reflections)
        self.reflections.append(reflection)
        
        # Index keywords
        keywords = reflection.get('keywords', [])
        for keyword in keywords:
            if keyword not in self._index:
                self._index[keyword] = []
            self._index[keyword].append(reflection_id)
        
        # Maintain size limit
        if len(self.reflections) > self.max_size:
            # Remove oldest reflections
            removed = self.reflections[:-self.max_size]
            self.reflections = self.reflections[-self.max_size:]
            
            # Rebuild index
            self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild keyword index after removals."""
        self._index.clear()
        for i, reflection in enumerate(self.reflections):
            keywords = reflection.get('keywords', [])
            for keyword in keywords:
                if keyword not in self._index:
                    self._index[keyword] = []
                self._index[keyword].append(i)
    
    def search_similar(self, query: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar reflections using keyword matching.
        
        Args:
            query: Query data to match against
            top_k: Number of results to return
            
        Returns:
            List of most relevant reflections
        """
        if not self.reflections:
            return []
        
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)
        
        # Score reflections based on keyword overlap
        scores = {}
        for keyword in query_keywords:
            if keyword in self._index:
                for reflection_id in self._index[keyword]:
                    if reflection_id not in scores:
                        scores[reflection_id] = 0
                    scores[reflection_id] += 1
        
        # Get top reflections
        if not scores:
            # No keyword matches, return most recent
            return self.reflections[-top_k:]
        
        # Sort by score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [id for id, _ in sorted_ids[:top_k]]
        
        return [self.reflections[id] for id in top_ids]
    
    def _extract_keywords(self, data: Dict[str, Any]) -> List[str]:
        """Extract keywords from data for indexing/search."""
        text_parts = []
        
        # Extract text from common fields
        if isinstance(data, dict):
            for key in ['prompt', 'query', 'input', 'context', 'task', 'response', 'output']:
                if key in data:
                    text_parts.append(str(data[key]))
        
        # Combine and extract keywords
        text = ' '.join(text_parts).lower()
        
        # Simple keyword extraction: words > 3 chars, excluding stop words
        words = re.findall(r'\b[a-z]{4,}\b', text)
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'where'}
        
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))[:20]  # Limit to 20 keywords


class ReflexionEngine(BaseImprovementEngine):
    """
    Reflexion improvement engine that enhances agent performance through
    self-reflection and learning from past mistakes.
    
    This engine is model-agnostic and allows users to provide their own
    LLM interface and customization functions.
    """
    
    def _initialize_technique(self):
        """Initialize Reflexion-specific parameters."""
        # Load configuration from agent config or use defaults
        reflexion_params = self.agent_config.custom_parameters.get('reflexion', {})
        
        self.config = ReflexionConfig(
            max_reflections=reflexion_params.get('max_reflections', 3),
            reflection_threshold=reflexion_params.get('reflection_threshold', 0.7),
            use_external_feedback=reflexion_params.get('use_external_feedback', True),
            memory_size=reflexion_params.get('memory_size', 50),
            enable_async=reflexion_params.get('enable_async', True),
            batch_size=reflexion_params.get('batch_size', 10),
            evaluation_function=reflexion_params.get('evaluation_function'),
            llm_interface=reflexion_params.get('llm_interface'),
            reflection_prompt_builder=reflexion_params.get('reflection_prompt_builder'),
            response_parser=reflexion_params.get('response_parser')
        )
        
        # Validate LLM interface
        if not self.config.llm_interface:
            self.logger.warning("No LLM interface provided. Reflection will be limited.")
        
        # Initialize reflection memory
        self.memory = ReflexionMemory(max_size=self.config.memory_size)
        self._performance_history: List[float] = []
        
        # Thread pool for async operations
        if self.config.enable_async:
            self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info(f"Initialized Reflexion with max_reflections={self.config.max_reflections}")
    
    def enhance_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance input by adding relevant reflections and past learnings.
        
        Args:
            input_data: Original input data
            
        Returns:
            Enhanced input with reflection context
        """
        try:
            enhanced_input = input_data.copy()
            
            # Search for relevant reflections
            relevant_reflections = self.memory.search_similar(input_data, top_k=3)
            
            if relevant_reflections:
                # Build reflection context
                reflection_context = {
                    'past_learnings': [r.get('learning', '') for r in relevant_reflections],
                    'improvement_strategies': [r.get('strategy', '') for r in relevant_reflections],
                    'relevant_patterns': [r.get('pattern', '') for r in relevant_reflections]
                }
                
                # Add to input based on structure
                if 'context' in enhanced_input:
                    if isinstance(enhanced_input['context'], dict):
                        enhanced_input['context']['reflections'] = reflection_context
                    else:
                        enhanced_input['reflection_context'] = reflection_context
                else:
                    enhanced_input['reflection_context'] = reflection_context
                
                # Add metadata
                enhanced_input['_reflexion_metadata'] = {
                    'applied_reflections': len(relevant_reflections),
                    'reflection_ids': [r.get('id', '') for r in relevant_reflections]
                }
                
                self.logger.debug(f"Applied {len(relevant_reflections)} reflections to input")
            
            self._record_improvement(input_data, enhanced_input, 'input', True)
            self.update_metrics(True, 0.8)
            
            return enhanced_input
            
        except Exception as e:
            self.logger.error(f"Failed to enhance input with reflexion: {e}")
            self._record_improvement(input_data, input_data, 'input', False, {'error': str(e)})
            self.update_metrics(False)
            return input_data
    
    def enhance_output(self, output_data: Dict[str, Any], 
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance output by applying reflection if performance is below threshold.
        
        Args:
            output_data: Original output data
            input_data: Corresponding input data
            
        Returns:
            Enhanced output with potential reflection improvements
        """
        try:
            enhanced_output = output_data.copy()
            
            # Evaluate current performance
            performance_score = self._evaluate_performance(output_data, input_data)
            self._performance_history.append(performance_score)
            
            # Apply reflection if performance is below threshold and LLM is available
            if performance_score < self.config.reflection_threshold and self.config.llm_interface:
                reflection_result = self._perform_reflection(
                    output_data, input_data, performance_score
                )
                
                if reflection_result and reflection_result.get('improved_output'):
                    enhanced_output = reflection_result['improved_output']
                    enhanced_output['_reflexion_metadata'] = {
                        'reflection_applied': True,
                        'original_score': performance_score,
                        'improved_score': reflection_result.get('improved_score', performance_score),
                        'reflection_steps': reflection_result.get('steps', [])
                    }
                    
                    # Store reflection for future use
                    if 'reflection' in reflection_result:
                        self._store_reflection(reflection_result['reflection'])
            
            self._record_improvement(output_data, enhanced_output, 'output', True)
            self.update_metrics(True, performance_score)
            
            return enhanced_output
            
        except Exception as e:
            self.logger.error(f"Failed to enhance output with reflexion: {e}")
            self._record_improvement(output_data, output_data, 'output', False, {'error': str(e)})
            self.update_metrics(False)
            return output_data
    
    def _evaluate_performance(self, output_data: Dict[str, Any], 
                            input_data: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a response.
        
        Returns a score between 0 and 1.
        """
        # Use custom evaluation function if provided
        if self.config.evaluation_function:
            try:
                return self.config.evaluation_function(output_data, input_data)
            except Exception as e:
                self.logger.error(f"Custom evaluation function failed: {e}")
        
        # Default evaluation based on output structure
        score = 0.5  # Base score
        
        # Check for common quality indicators
        if output_data:
            # Has content
            if any(key in output_data for key in ['response', 'output', 'result', 'answer']):
                score += 0.2
            
            # Has metadata/confidence
            if any(key in output_data for key in ['confidence', 'metadata', 'reasoning']):
                score += 0.1
            
            # No errors
            if 'error' not in output_data and 'exception' not in output_data:
                score += 0.2
        
        return min(score, 1.0)
    
    def _perform_reflection(self, output_data: Dict[str, Any], 
                          input_data: Dict[str, Any], 
                          performance_score: float) -> Optional[Dict[str, Any]]:
        """
        Perform reflection to improve the response using the provided LLM interface.
        """
        if not self.config.llm_interface:
            return None
        
        reflection_steps = []
        current_output = output_data
        current_score = performance_score
        
        for step in range(self.config.max_reflections):
            # Generate reflection prompt
            if self.config.reflection_prompt_builder:
                reflection_prompt = self.config.reflection_prompt_builder(
                    input_data, current_output, current_score, step
                )
            else:
                reflection_prompt = self._default_reflection_prompt(
                    input_data, current_output, current_score, step
                )
            
            # Call LLM for reflection
            try:
                reflection_response = self.config.llm_interface.generate(reflection_prompt)
            except Exception as e:
                self.logger.error(f"LLM reflection call failed: {e}")
                break
            
            # Parse reflection response
            if self.config.response_parser:
                reflection_data = self.config.response_parser(reflection_response)
            else:
                reflection_data = self._default_response_parser(reflection_response)
            
            if reflection_data.get('improved_output'):
                current_output = reflection_data['improved_output']
                new_score = self._evaluate_performance(current_output, input_data)
                
                reflection_steps.append({
                    'step': step + 1,
                    'analysis': reflection_data.get('analysis', ''),
                    'improvements': reflection_data.get('improvements', []),
                    'score_change': new_score - current_score
                })
                
                current_score = new_score
                
                # Stop if we've reached acceptable performance
                if current_score >= self.config.reflection_threshold:
                    break
        
        if current_score > performance_score:
            # Extract keywords for indexing
            keywords = self.memory._extract_keywords({**input_data, **current_output})
            
            return {
                'improved_output': current_output,
                'improved_score': current_score,
                'steps': reflection_steps,
                'reflection': {
                    'id': f"refl_{self._get_timestamp()}",
                    'keywords': keywords,
                    'learning': self._summarize_learning(reflection_steps),
                    'strategy': f"Applied {len(reflection_steps)} reflection steps",
                    'pattern': self._identify_pattern(input_data)
                }
            }
        
        return None
    
    def _default_reflection_prompt(self, input_data: Dict[str, Any], 
                                 output_data: Dict[str, Any], 
                                 performance_score: float,
                                 step: int) -> Dict[str, Any]:
        """Default reflection prompt format."""
        return {
            'task': 'reflection',
            'input': input_data,
            'current_output': output_data,
            'performance_score': performance_score,
            'threshold': self.config.reflection_threshold,
            'step': step + 1,
            'max_steps': self.config.max_reflections,
            'instructions': (
                "Analyze the current output and provide improvements. "
                "Return analysis, list of improvements, and improved output."
            )
        }
    
    def _default_response_parser(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Default parser for reflection responses."""
        # Handle different response formats
        if isinstance(response, dict):
            # Already structured
            return {
                'analysis': response.get('analysis', ''),
                'improvements': response.get('improvements', []),
                'improved_output': response.get('improved_output', response.get('output', {}))
            }
        
        # If string response, try to extract structure
        if isinstance(response, str):
            # Look for JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
        
        # Fallback
        return {
            'analysis': str(response),
            'improvements': [],
            'improved_output': None
        }
    
    def _summarize_learning(self, reflection_steps: List[Dict[str, Any]]) -> str:
        """Summarize key learnings from reflection steps."""
        if not reflection_steps:
            return "No specific improvements identified"
        
        all_improvements = []
        for step in reflection_steps:
            all_improvements.extend(step.get('improvements', []))
        
        if all_improvements:
            # Take first 3 most important improvements
            summary = all_improvements[:3]
            return f"Key improvements: {'; '.join(summary)}"
        
        return "Iterative refinement applied"
    
    def _identify_pattern(self, input_data: Dict[str, Any]) -> str:
        """Identify task pattern from input."""
        # Extract text to analyze
        text_content = []
        if isinstance(input_data, dict):
            for key in ['prompt', 'query', 'task', 'input']:
                if key in input_data:
                    text_content.append(str(input_data[key]).lower())
        
        text = ' '.join(text_content)
        
        # Pattern detection
        patterns = []
        
        if '?' in text or any(q in text for q in ['what', 'why', 'how', 'when', 'where']):
            patterns.append('question')
        if any(cmd in text for cmd in ['create', 'generate', 'write', 'make']):
            patterns.append('generation')
        if any(cmd in text for cmd in ['analyze', 'explain', 'describe', 'evaluate']):
            patterns.append('analysis')
        if any(cmd in text for cmd in ['fix', 'correct', 'improve', 'debug']):
            patterns.append('correction')
        
        return '-'.join(patterns) if patterns else 'general'
    
    def _store_reflection(self, reflection: Dict[str, Any]):
        """Store a reflection for future use."""
        self.memory.add_reflection(reflection)
        self.logger.info(f"Stored reflection: {reflection.get('learning', '')[:50]}...")
    
    def _get_technique_config(self) -> Dict[str, Any]:
        """Get Reflexion-specific configuration."""
        return {
            'technique': 'Reflexion',
            'max_reflections': self.config.max_reflections,
            'reflection_threshold': self.config.reflection_threshold,
            'memory_size': self.config.memory_size,
            'current_memory_usage': len(self.memory.reflections),
            'performance_history_length': len(self._performance_history),
            'average_performance': (
                sum(self._performance_history) / len(self._performance_history) 
                if self._performance_history else 0.0
            ),
            'llm_interface_provided': self.config.llm_interface is not None,
            'custom_evaluation': self.config.evaluation_function is not None,
            'custom_prompt_builder': self.config.reflection_prompt_builder is not None,
            'custom_parser': self.config.response_parser is not None
        }
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights about reflexion performance."""
        insights = {
            'total_interactions': len(self._performance_history),
            'reflections_stored': len(self.memory.reflections),
            'memory_keywords': len(self.memory._index),
            'configuration': self._get_technique_config()
        }
        
        if self._performance_history:
            recent = self._performance_history[-10:]
            insights.update({
                'average_performance': sum(self._performance_history) / len(self._performance_history),
                'recent_average': sum(recent) / len(recent),
                'performance_trend': self._calculate_trend(),
                'below_threshold_count': sum(1 for s in self._performance_history if s < self.config.reflection_threshold)
            })
        
        return insights
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend."""
        if len(self._performance_history) < 10:
            return 'insufficient_data'
        
        # Compare first and last quartiles
        quarter_size = len(self._performance_history) // 4
        first_quarter = self._performance_history[:quarter_size]
        last_quarter = self._performance_history[-quarter_size:]
        
        first_avg = sum(first_quarter) / len(first_quarter)
        last_avg = sum(last_quarter) / len(last_quarter)
        
        if last_avg > first_avg + 0.05:
            return 'improving'
        elif first_avg > last_avg + 0.05:
            return 'declining'
        else:
            return 'stable'