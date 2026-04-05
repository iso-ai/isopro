"""
Constitutional AI Improvement Engine

Implements Constitutional AI for improving agent outputs through 
principle-based critique and revision.

Based on: "Constitutional AI: Harmlessness from AI Feedback"
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Protocol
import json
from dataclasses import dataclass, field
from enum import Enum

from .base_improvement import BaseImprovementEngine


class ConstitutionType(Enum):
    """Types of constitutional principles."""
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    ETHICS = "ethics"
    CUSTOM = "custom"


@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle."""
    id: str
    type: ConstitutionType
    description: str
    critique_prompt: str
    revision_prompt: str
    weight: float = 1.0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstitutionalAIConfig:
    """Configuration for Constitutional AI."""
    principles: List[ConstitutionalPrinciple] = field(default_factory=list)
    max_revisions: int = 2
    critique_threshold: float = 0.7  # Minimum score to pass critique
    enable_async: bool = True
    batch_critique: bool = True
    llm_interface: Optional['LLMInterface'] = None
    critique_parser: Optional[Callable] = None
    revision_parser: Optional[Callable] = None
    principle_selector: Optional[Callable] = None  # Custom principle selection


class ConstitutionalAIEngine(BaseImprovementEngine):
    """
    Constitutional AI improvement engine that enhances outputs through
    principle-based critique and revision.
    """
    
    def _initialize_technique(self):
        """Initialize Constitutional AI parameters."""
        # Load configuration
        const_params = self.agent_config.custom_parameters.get('constitutional_ai', {})
        
        self.config = ConstitutionalAIConfig(
            max_revisions=const_params.get('max_revisions', 2),
            critique_threshold=const_params.get('critique_threshold', 0.7),
            enable_async=const_params.get('enable_async', True),
            batch_critique=const_params.get('batch_critique', True),
            llm_interface=const_params.get('llm_interface'),
            critique_parser=const_params.get('critique_parser'),
            revision_parser=const_params.get('revision_parser'),
            principle_selector=const_params.get('principle_selector')
        )
        
        # Load principles
        if 'principles' in const_params:
            self.config.principles = self._load_principles(const_params['principles'])
        else:
            self.config.principles = self._get_default_principles()
        
        # Track principle usage
        self._principle_stats: Dict[str, Dict[str, Any]] = {}
        self._initialize_principle_stats()
        
        self.logger.info(f"Initialized Constitutional AI with {len(self.config.principles)} principles")
    
    def _load_principles(self, principles_data: List[Dict[str, Any]]) -> List[ConstitutionalPrinciple]:
        """Load principles from configuration."""
        principles = []
        
        for p_data in principles_data:
            principle = ConstitutionalPrinciple(
                id=p_data.get('id', f"principle_{len(principles)}"),
                type=ConstitutionType(p_data.get('type', 'custom')),
                description=p_data['description'],
                critique_prompt=p_data['critique_prompt'],
                revision_prompt=p_data['revision_prompt'],
                weight=p_data.get('weight', 1.0),
                active=p_data.get('active', True),
                metadata=p_data.get('metadata', {})
            )
            principles.append(principle)
        
        return principles
    
    def _get_default_principles(self) -> List[ConstitutionalPrinciple]:
        """Get default constitutional principles."""
        return [
            ConstitutionalPrinciple(
                id="harmlessness_1",
                type=ConstitutionType.HARMLESSNESS,
                description="Avoid harmful, offensive, or inappropriate content",
                critique_prompt=(
                    "Does this response contain any harmful, offensive, or inappropriate content? "
                    "Consider potential harms to individuals or groups. "
                    "Provide specific examples if found."
                ),
                revision_prompt=(
                    "Revise this response to remove any harmful content while maintaining "
                    "the helpful aspects of the answer."
                ),
                weight=2.0  # Higher weight for safety
            ),
            ConstitutionalPrinciple(
                id="helpfulness_1",
                type=ConstitutionType.HELPFULNESS,
                description="Provide helpful and constructive responses",
                critique_prompt=(
                    "Is this response helpful and constructive? "
                    "Does it address the user's needs effectively? "
                    "Identify areas for improvement."
                ),
                revision_prompt=(
                    "Improve this response to be more helpful and directly address "
                    "the user's needs."
                ),
                weight=1.5
            ),
            ConstitutionalPrinciple(
                id="honesty_1",
                type=ConstitutionType.HONESTY,
                description="Be honest about capabilities and limitations",
                critique_prompt=(
                    "Does this response honestly represent capabilities and limitations? "
                    "Are there any overstatements or misleading claims?"
                ),
                revision_prompt=(
                    "Revise to ensure honesty about capabilities and limitations, "
                    "adding appropriate caveats where needed."
                ),
                weight=1.5
            ),
            ConstitutionalPrinciple(
                id="clarity_1",
                type=ConstitutionType.CLARITY,
                description="Communicate clearly and concisely",
                critique_prompt=(
                    "Is this response clear and easy to understand? "
                    "Are there any ambiguous or confusing parts?"
                ),
                revision_prompt=(
                    "Revise for clarity and conciseness, ensuring the message "
                    "is easy to understand."
                ),
                weight=1.0
            ),
            ConstitutionalPrinciple(
                id="relevance_1",
                type=ConstitutionType.RELEVANCE,
                description="Stay relevant to the user's request",
                critique_prompt=(
                    "Does this response stay relevant to the user's request? "
                    "Are there unnecessary tangents or missing key points?"
                ),
                revision_prompt=(
                    "Focus the response on directly addressing the user's request, "
                    "removing irrelevant information."
                ),
                weight=1.0
            )
        ]
    
    def _initialize_principle_stats(self):
        """Initialize statistics tracking for principles."""
        for principle in self.config.principles:
            self._principle_stats[principle.id] = {
                'applications': 0,
                'critiques_triggered': 0,
                'revisions_made': 0,
                'average_improvement': 0.0
            }
    
    def enhance_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance input by adding constitutional context.
        
        Args:
            input_data: Original input data
            
        Returns:
            Enhanced input with constitutional guidelines
        """
        try:
            enhanced_input = input_data.copy()
            
            # Select relevant principles for this input
            relevant_principles = self._select_relevant_principles(input_data)
            
            if relevant_principles:
                # Add constitutional context
                const_context = {
                    'active_principles': [
                        {
                            'type': p.type.value,
                            'description': p.description
                        }
                        for p in relevant_principles
                    ],
                    'principle_count': len(relevant_principles)
                }
                
                # Add to input
                if 'context' in enhanced_input and isinstance(enhanced_input['context'], dict):
                    enhanced_input['context']['constitutional_guidelines'] = const_context
                else:
                    enhanced_input['constitutional_context'] = const_context
                
                enhanced_input['_constitutional_metadata'] = {
                    'principle_ids': [p.id for p in relevant_principles]
                }
            
            self._record_improvement(input_data, enhanced_input, 'input', True)
            self.update_metrics(True, 0.7)
            
            return enhanced_input
            
        except Exception as e:
            self.logger.error(f"Failed to enhance input with constitutional AI: {e}")
            self._record_improvement(input_data, input_data, 'input', False, {'error': str(e)})
            self.update_metrics(False)
            return input_data
    
    def enhance_output(self, output_data: Dict[str, Any], 
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance output through constitutional critique and revision.
        
        Args:
            output_data: Original output data
            input_data: Corresponding input data
            
        Returns:
            Enhanced output after constitutional review
        """
        try:
            if not self.config.llm_interface:
                # No LLM interface, return original
                self.logger.debug("No LLM interface provided for constitutional review")
                return output_data
            
            enhanced_output = output_data.copy()
            
            # Select principles to apply
            principles_to_apply = self._select_relevant_principles(input_data, output_data)
            
            if not principles_to_apply:
                return enhanced_output
            
            # Apply constitutional review
            review_result = self._apply_constitutional_review(
                enhanced_output, input_data, principles_to_apply
            )
            
            if review_result and review_result.get('revised_output'):
                enhanced_output = review_result['revised_output']
                enhanced_output['_constitutional_metadata'] = {
                    'review_applied': True,
                    'principles_applied': [p.id for p in principles_to_apply],
                    'revisions_made': review_result.get('revision_count', 0),
                    'critique_results': review_result.get('critiques', [])
                }
                
                # Update principle statistics
                self._update_principle_stats(principles_to_apply, review_result)
            
            self._record_improvement(output_data, enhanced_output, 'output', True)
            self.update_metrics(True, 0.85)
            
            return enhanced_output
            
        except Exception as e:
            self.logger.error(f"Failed to enhance output with constitutional AI: {e}")
            self._record_improvement(output_data, output_data, 'output', False, {'error': str(e)})
            self.update_metrics(False)
            return output_data
    
    def _select_relevant_principles(self, input_data: Dict[str, Any], 
                                  output_data: Optional[Dict[str, Any]] = None) -> List[ConstitutionalPrinciple]:
        """Select principles relevant to the current context."""
        # Use custom selector if provided
        if self.config.principle_selector:
            try:
                return self.config.principle_selector(
                    self.config.principles, input_data, output_data
                )
            except Exception as e:
                self.logger.error(f"Custom principle selector failed: {e}")
        
        # Default selection: all active principles
        selected = [p for p in self.config.principles if p.active]
        
        # Sort by weight
        selected.sort(key=lambda p: p.weight, reverse=True)
        
        return selected
    
    def _apply_constitutional_review(self, output_data: Dict[str, Any],
                                   input_data: Dict[str, Any],
                                   principles: List[ConstitutionalPrinciple]) -> Optional[Dict[str, Any]]:
        """Apply constitutional review process."""
        current_output = output_data
        critiques = []
        revision_count = 0
        
        for revision_round in range(self.config.max_revisions):
            # Critique phase
            critique_results = self._critique_output(
                current_output, input_data, principles
            )
            
            critiques.extend(critique_results)
            
            # Check if revision is needed
            needs_revision = any(
                c['score'] < self.config.critique_threshold 
                for c in critique_results
            )
            
            if not needs_revision:
                break
            
            # Revision phase
            revised_output = self._revise_output(
                current_output, input_data, critique_results
            )
            
            if revised_output and revised_output != current_output:
                current_output = revised_output
                revision_count += 1
            else:
                break
        
        if revision_count > 0:
            return {
                'revised_output': current_output,
                'revision_count': revision_count,
                'critiques': critiques
            }
        
        return None
    
    def _critique_output(self, output_data: Dict[str, Any],
                       input_data: Dict[str, Any],
                       principles: List[ConstitutionalPrinciple]) -> List[Dict[str, Any]]:
        """Critique output against principles."""
        critiques = []
        
        if self.config.batch_critique and len(principles) > 1:
            # Batch critique for efficiency
            batch_prompt = self._build_batch_critique_prompt(
                output_data, input_data, principles
            )
            
            try:
                response = self.config.llm_interface.generate(batch_prompt)
                parsed_critiques = self._parse_critique_response(response, principles)
                critiques.extend(parsed_critiques)
            except Exception as e:
                self.logger.error(f"Batch critique failed: {e}")
        else:
            # Individual critiques
            for principle in principles:
                critique_prompt = self._build_critique_prompt(
                    output_data, input_data, principle
                )
                
                try:
                    response = self.config.llm_interface.generate(critique_prompt)
                    critique = self._parse_single_critique(response, principle)
                    critiques.append(critique)
                except Exception as e:
                    self.logger.error(f"Critique failed for {principle.id}: {e}")
        
        return critiques
    
    def _revise_output(self, output_data: Dict[str, Any],
                      input_data: Dict[str, Any],
                      critique_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Revise output based on critiques."""
        # Filter critiques that need revision
        failed_critiques = [
            c for c in critique_results 
            if c['score'] < self.config.critique_threshold
        ]
        
        if not failed_critiques:
            return None
        
        # Build revision prompt
        revision_prompt = self._build_revision_prompt(
            output_data, input_data, failed_critiques
        )
        
        try:
            response = self.config.llm_interface.generate(revision_prompt)
            return self._parse_revision_response(response, output_data)
        except Exception as e:
            self.logger.error(f"Revision failed: {e}")
            return None
    
    def _build_batch_critique_prompt(self, output_data: Dict[str, Any],
                                   input_data: Dict[str, Any],
                                   principles: List[ConstitutionalPrinciple]) -> Dict[str, Any]:
        """Build prompt for batch critique."""
        principle_prompts = []
        for i, principle in enumerate(principles):
            principle_prompts.append(
                f"{i+1}. {principle.type.value.upper()}: {principle.critique_prompt}"
            )
        
        return {
            'task': 'constitutional_critique',
            'input': input_data,
            'output': output_data,
            'principles': principle_prompts,
            'instructions': (
                "Evaluate the output against each principle. "
                "For each principle, provide a score (0-1) and explanation. "
                "Format: [{principle_number, score, explanation}, ...]"
            )
        }
    
    def _build_critique_prompt(self, output_data: Dict[str, Any],
                             input_data: Dict[str, Any],
                             principle: ConstitutionalPrinciple) -> Dict[str, Any]:
        """Build prompt for single critique."""
        return {
            'task': 'constitutional_critique_single',
            'input': input_data,
            'output': output_data,
            'principle': {
                'type': principle.type.value,
                'description': principle.description,
                'prompt': principle.critique_prompt
            },
            'instructions': "Evaluate against this principle. Provide score (0-1) and explanation."
        }
    
    def _build_revision_prompt(self, output_data: Dict[str, Any],
                             input_data: Dict[str, Any],
                             failed_critiques: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build prompt for revision."""
        critique_summary = []
        revision_prompts = []
        
        for critique in failed_critiques:
            critique_summary.append(
                f"- {critique['principle_type']}: {critique['explanation']}"
            )
            if 'revision_prompt' in critique:
                revision_prompts.append(critique['revision_prompt'])
        
        return {
            'task': 'constitutional_revision',
            'input': input_data,
            'current_output': output_data,
            'critique_summary': critique_summary,
            'revision_guidance': revision_prompts,
            'instructions': (
                "Revise the output to address the critiques while maintaining "
                "all positive aspects. Return the complete revised output."
            )
        }
    
    def _parse_critique_response(self, response: Dict[str, Any], 
                               principles: List[ConstitutionalPrinciple]) -> List[Dict[str, Any]]:
        """Parse batch critique response."""
        if self.config.critique_parser:
            return self.config.critique_parser(response, principles)
        
        # Default parsing
        critiques = []
        
        # Try to extract structured data
        if isinstance(response, dict) and 'critiques' in response:
            for i, critique_data in enumerate(response['critiques']):
                if i < len(principles):
                    critiques.append({
                        'principle_id': principles[i].id,
                        'principle_type': principles[i].type.value,
                        'score': critique_data.get('score', 0.5),
                        'explanation': critique_data.get('explanation', ''),
                        'revision_prompt': principles[i].revision_prompt
                    })
        else:
            # Fallback: assume all pass with default score
            for principle in principles:
                critiques.append({
                    'principle_id': principle.id,
                    'principle_type': principle.type.value,
                    'score': 0.8,  # Default passing score
                    'explanation': 'No specific issues identified',
                    'revision_prompt': principle.revision_prompt
                })
        
        return critiques
    
    def _parse_single_critique(self, response: Dict[str, Any],
                             principle: ConstitutionalPrinciple) -> Dict[str, Any]:
        """Parse single critique response."""
        critique = {
            'principle_id': principle.id,
            'principle_type': principle.type.value,
            'revision_prompt': principle.revision_prompt
        }
        
        if isinstance(response, dict):
            critique['score'] = response.get('score', 0.5)
            critique['explanation'] = response.get('explanation', '')
        else:
            # Fallback
            critique['score'] = 0.8
            critique['explanation'] = str(response)[:200]
        
        return critique
    
    def _parse_revision_response(self, response: Dict[str, Any],
                               original_output: Dict[str, Any]) -> Dict[str, Any]:
        """Parse revision response."""
        if self.config.revision_parser:
            return self.config.revision_parser(response, original_output)
        
        # Default parsing
        if isinstance(response, dict):
            # Direct dict response
            if 'revised_output' in response:
                return response['revised_output']
            elif 'output' in response:
                return response['output']
            else:
                # Assume entire response is the revision
                return response
        
        # Fallback: wrap string response
        return {
            'response': str(response),
            '_revised': True
        }
    
    def _update_principle_stats(self, principles: List[ConstitutionalPrinciple],
                              review_result: Dict[str, Any]):
        """Update statistics for principles."""
        critiques = review_result.get('critiques', [])
        
        # Map critiques to principles
        critique_map = {c['principle_id']: c for c in critiques}
        
        for principle in principles:
            stats = self._principle_stats[principle.id]
            stats['applications'] += 1
            
            if principle.id in critique_map:
                critique = critique_map[principle.id]
                if critique['score'] < self.config.critique_threshold:
                    stats['critiques_triggered'] += 1
                    if review_result.get('revision_count', 0) > 0:
                        stats['revisions_made'] += 1
    
    def add_principle(self, principle: ConstitutionalPrinciple):
        """Add a new principle dynamically."""
        self.config.principles.append(principle)
        self._principle_stats[principle.id] = {
            'applications': 0,
            'critiques_triggered': 0,
            'revisions_made': 0,
            'average_improvement': 0.0
        }
        self.logger.info(f"Added principle: {principle.id}")
    
    def remove_principle(self, principle_id: str):
        """Remove a principle by ID."""
        self.config.principles = [
            p for p in self.config.principles 
            if p.id != principle_id
        ]
        if principle_id in self._principle_stats:
            del self._principle_stats[principle_id]
        self.logger.info(f"Removed principle: {principle_id}")
    
    def update_principle(self, principle_id: str, updates: Dict[str, Any]):
        """Update a principle's configuration."""
        for principle in self.config.principles:
            if principle.id == principle_id:
                for key, value in updates.items():
                    if hasattr(principle, key):
                        setattr(principle, key, value)
                self.logger.info(f"Updated principle: {principle_id}")
                break
    
    def get_principle_stats(self) -> Dict[str, Any]:
        """Get statistics for all principles."""
        stats = {}
        for principle_id, principle_stats in self._principle_stats.items():
            principle = next((p for p in self.config.principles if p.id == principle_id), None)
            if principle:
                stats[principle_id] = {
                    'type': principle.type.value,
                    'description': principle.description,
                    'weight': principle.weight,
                    'active': principle.active,
                    'stats': principle_stats
                }
        return stats
    
    def _get_technique_config(self) -> Dict[str, Any]:
        """Get Constitutional AI configuration."""
        return {
            'technique': 'Constitutional AI',
            'principle_count': len(self.config.principles),
            'active_principles': sum(1 for p in self.config.principles if p.active),
            'max_revisions': self.config.max_revisions,
            'critique_threshold': self.config.critique_threshold,
            'llm_interface_provided': self.config.llm_interface is not None,
            'custom_parsers': {
                'critique': self.config.critique_parser is not None,
                'revision': self.config.revision_parser is not None
            }
        }