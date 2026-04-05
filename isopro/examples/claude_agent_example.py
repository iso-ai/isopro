"""
Claude Agent Example

This example demonstrates how to use the ISOPro framework with Claude
to create enhanced AI agents using Reflexion, Constitutional AI, DQN, and DPO.
"""

import anthropic
from typing import Dict, Any
import asyncio
import os

from isopro.core import AgentFramework, AgentConfig, AgentType, ImprovementTechnique


class ClaudeInterface:
    """
    Claude LLM interface for ISOPro framework.
    """
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Claude interface.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        
        if not self.api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        print(f"🤖 Initialized Claude interface with model: {model}")
    
    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using Claude.
        
        Args:
            prompt: Prompt data with task-specific formatting
            
        Returns:
            Response data
        """
        try:
            # Handle different prompt types
            if prompt.get('task') == 'reflection':
                return self._handle_reflection_prompt(prompt)
            elif prompt.get('task') == 'constitutional_critique':
                return self._handle_constitutional_critique(prompt)
            elif prompt.get('task') == 'constitutional_revision':
                return self._handle_constitutional_revision(prompt)
            else:
                return self._handle_standard_prompt(prompt)
                
        except Exception as e:
            return {
                'error': str(e),
                'response': f"Error generating response: {e}"
            }
    
    def _handle_reflection_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Reflexion-style prompts."""
        input_data = prompt.get('input', {})
        current_output = prompt.get('current_output', {})
        score = prompt.get('performance_score', 0)
        
        reflection_text = f"""
Please analyze and improve the following response:

ORIGINAL INPUT: {input_data.get('prompt', '')}

CURRENT RESPONSE: {current_output.get('response', '')}

PERFORMANCE SCORE: {score:.2f} (threshold: {prompt.get('threshold', 0.7)})

The response scored below the quality threshold. Please:
1. Analyze what aspects need improvement
2. Identify specific weaknesses or missing elements  
3. Provide an improved version
4. List the key improvements made

Format your response as JSON:
{{
    "analysis": "Your analysis of issues",
    "improvements": ["List", "of", "improvements"],
    "improved_response": {{
        "response": "Your improved response",
        "confidence": 0.9,
        "reasoning": "Why this is better"
    }}
}}
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": reflection_text}]
        )
        
        # Parse JSON response
        try:
            import json
            import re
            
            content = response.content[0].text
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except:
            pass
        
        # Fallback parsing
        return {
            'analysis': 'Response needs improvement in clarity and detail',
            'improvements': ['Add more detail', 'Improve structure'],
            'improved_response': {
                'response': content,
                'confidence': 0.8
            }
        }
    
    def _handle_constitutional_critique(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Constitutional AI critique prompts."""
        input_data = prompt.get('input', {})
        output_data = prompt.get('output', {})
        principles = prompt.get('principles', [])
        
        critique_text = f"""
Please evaluate this response against constitutional principles:

INPUT: {input_data.get('prompt', '')}

RESPONSE TO EVALUATE: {output_data.get('response', '')}

PRINCIPLES TO CHECK:
"""
        
        for i, principle in enumerate(principles, 1):
            critique_text += f"{i}. {principle}\n"
        
        critique_text += """
For each principle, provide:
- Score (0.0 to 1.0, where 1.0 is perfect compliance)
- Explanation of your assessment

Format as JSON:
{
    "critiques": [
        {"principle": 1, "score": 0.8, "explanation": "Good but could be better"},
        {"principle": 2, "score": 0.9, "explanation": "Excellent compliance"}
    ]
}
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": critique_text}]
        )
        
        try:
            import json
            import re
            
            content = response.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback
        return {
            'critiques': [
                {'principle': i+1, 'score': 0.8, 'explanation': 'Generally good'}
                for i in range(len(principles))
            ]
        }
    
    def _handle_constitutional_revision(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Constitutional AI revision prompts."""
        input_data = prompt.get('input', {})
        current_output = prompt.get('current_output', {})
        critique_summary = prompt.get('critique_summary', [])
        
        revision_text = f"""
Please revise this response to address the identified issues:

ORIGINAL INPUT: {input_data.get('prompt', '')}

CURRENT RESPONSE: {current_output.get('response', '')}

ISSUES TO ADDRESS:
"""
        
        for issue in critique_summary:
            revision_text += f"- {issue}\n"
        
        revision_text += """
Provide a revised response that addresses these issues while maintaining helpfulness and accuracy.
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": revision_text}]
        )
        
        return {
            'response': response.content[0].text,
            'confidence': 0.85,
            '_revised': True
        }
    
    def _handle_standard_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard prompts."""
        # Extract prompt text
        if isinstance(prompt, str):
            prompt_text = prompt
        elif 'prompt' in prompt:
            prompt_text = prompt['prompt']
        elif 'input' in prompt:
            prompt_text = str(prompt['input'])
        else:
            prompt_text = str(prompt)
        
        # Add any context
        if isinstance(prompt, dict) and 'context' in prompt:
            prompt_text = f"Context: {prompt['context']}\n\nPrompt: {prompt_text}"
        
        # Add reflection context if present
        if isinstance(prompt, dict) and 'reflection_context' in prompt:
            context = prompt['reflection_context']
            if context.get('past_learnings'):
                prompt_text = f"Past learnings: {'; '.join(context['past_learnings'])}\n\n{prompt_text}"
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt_text}]
        )
        
        return {
            'response': response.content[0].text,
            'confidence': 0.8,
            'model': self.model
        }
    
    async def generate_async(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of generate."""
        # For now, just call sync version
        # In production, you'd use the async Anthropic client
        return self.generate(prompt)


def quality_evaluation_function(output_data: Dict[str, Any], 
                               input_data: Dict[str, Any]) -> float:
    """
    Quality evaluation function specifically for Claude responses.
    
    Args:
        output_data: Claude's output
        input_data: Original input
        
    Returns:
        Quality score between 0 and 1
    """
    score = 0.4  # Base score
    
    response_text = output_data.get('response', '')
    
    # Length and substance scoring
    if len(response_text) > 50:
        score += 0.1
    if len(response_text) > 200:
        score += 0.1
    
    # Structure and formatting
    if '\n' in response_text or '•' in response_text or '-' in response_text:
        score += 0.1  # Has structure
    
    # Quality indicators
    if 'confidence' in output_data:
        confidence = output_data.get('confidence', 0.5)
        score += confidence * 0.2
    
    if 'reasoning' in output_data:
        score += 0.1
    
    # Error checking
    if 'error' not in output_data:
        score += 0.1
    
    # Content quality heuristics
    input_text = input_data.get('prompt', '').lower()
    response_lower = response_text.lower()
    
    # Relevance check (simple keyword overlap)
    input_words = set(input_text.split())
    response_words = set(response_lower.split())
    overlap = len(input_words & response_words)
    if overlap > 2:
        score += min(overlap * 0.02, 0.1)
    
    # Question answering check
    if '?' in input_text and len(response_text) > 30:
        score += 0.1
    
    # Explanation quality
    if any(word in input_text for word in ['explain', 'describe', 'what', 'how', 'why']):
        if any(word in response_lower for word in ['because', 'since', 'therefore', 'due to']):
            score += 0.1
    
    return min(score, 1.0)


def main():
    """Main example function using Claude."""
    print("🚀 ISOPro Framework with Claude - Enhanced Agent Example")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        return
    
    # Initialize Claude interface
    claude_interface = ClaudeInterface(api_key=api_key)
    
    # Create enhanced agent configuration
    config = AgentConfig(
        agent_type=AgentType.LLM,
        model_name="claude-3-sonnet-20240229",
        improvement_techniques=[
            ImprovementTechnique.REFLEXION,
            ImprovementTechnique.CONSTITUTIONAL_AI
        ],
        custom_parameters={
            # Reflexion configuration
            'reflexion': {
                'llm_interface': claude_interface,
                'evaluation_function': quality_evaluation_function,
                'max_reflections': 2,
                'reflection_threshold': 0.75,  # Higher threshold for quality responses
                'memory_size': 30
            },
            # Constitutional AI configuration
            'constitutional_ai': {
                'llm_interface': claude_interface,
                'max_revisions': 2,
                'critique_threshold': 0.7,
                'principles': [
                    {
                        'id': 'helpfulness',
                        'type': 'helpfulness',
                        'description': 'Provide helpful, actionable, and comprehensive responses',
                        'critique_prompt': 'Is this response helpful and comprehensive? Does it fully address the user\'s needs?',
                        'revision_prompt': 'Make this response more helpful and comprehensive.'
                    },
                    {
                        'id': 'clarity',
                        'type': 'clarity',
                        'description': 'Communicate clearly with good structure and examples',
                        'critique_prompt': 'Is this response clear, well-structured, and easy to understand?',
                        'revision_prompt': 'Improve the clarity and structure of this response.'
                    },
                    {
                        'id': 'accuracy',
                        'type': 'accuracy',
                        'description': 'Provide accurate information with appropriate caveats',
                        'critique_prompt': 'Is this response accurate? Are there any claims that need caveats or verification?',
                        'revision_prompt': 'Ensure accuracy and add appropriate caveats where needed.'
                    }
                ]
            }
        }
    )
    
    # Create mock enhanced agent (in production, use the real EnhancedAgent)
    print("🔧 Creating enhanced Claude agent...")
    agent = MockClaudeAgent(config, claude_interface)
    
    # Test cases designed for Claude
    test_cases = [
        {
            'prompt': 'Explain the concept of machine learning in a way that a business executive would understand.',
            'context': 'Business explanation needed',
            'expected_improvements': ['business-friendly language', 'concrete examples', 'practical implications']
        },
        {
            'prompt': 'How do I implement a REST API authentication system using JWT tokens?',
            'context': 'Technical implementation guide',
            'expected_improvements': ['step-by-step guide', 'code examples', 'security considerations']
        },
        {
            'prompt': 'What are the environmental impacts of cryptocurrency mining?',
            'context': 'Environmental analysis',
            'expected_improvements': ['balanced perspective', 'specific data', 'multiple viewpoints']
        }
    ]
    
    print("\n🧪 Testing Enhanced Claude Agent:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"   Question: {test_case['prompt']}")
        print(f"   Context: {test_case['context']}")
        
        # Run the enhanced agent
        result = agent.run({
            'prompt': test_case['prompt'],
            'context': test_case['context']
        })
        
        # Display results
        response = result.get('response', 'No response generated')
        print(f"   📤 Response: {response[:150]}{'...' if len(response) > 150 else ''}")
        
        # Show improvement metadata
        if '_reflexion_metadata' in result:
            meta = result['_reflexion_metadata']
            print(f"   🔄 Reflexion Applied: {meta.get('reflection_applied', False)}")
            if meta.get('reflection_applied'):
                print(f"      Original Score: {meta.get('original_score', 0):.2f}")
                print(f"      Improved Score: {meta.get('improved_score', 0):.2f}")
        
        if '_constitutional_metadata' in result:
            meta = result['_constitutional_metadata']
            print(f"   ⚖️  Constitutional Review: {meta.get('review_applied', False)}")
            if meta.get('review_applied'):
                print(f"      Principles Applied: {len(meta.get('principles_applied', []))}")
                print(f"      Revisions Made: {meta.get('revisions_made', 0)}")
        
        # Quality assessment
        quality_score = quality_evaluation_function(result, {'prompt': test_case['prompt']})
        print(f"   📊 Quality Score: {quality_score:.2f}")
        
        print(f"   ✅ Expected improvements: {', '.join(test_case['expected_improvements'])}")
    
    # Show agent insights
    print(f"\n📈 Agent Performance Insights:")
    print("-" * 35)
    
    improvement_data = agent.get_improvement_data()
    
    # Reflexion insights
    if 'reflexion' in improvement_data:
        reflexion_data = improvement_data['reflexion']
        metrics = reflexion_data.get('metrics', {})
        print(f"🔄 Reflexion Engine:")
        print(f"   Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"   Applications: {metrics.get('total_applications', 0)}")
        print(f"   Average Improvement: {metrics.get('average_improvement_score', 0):.2f}")
    
    # Constitutional AI insights
    if 'constitutional_ai' in improvement_data:
        const_data = improvement_data['constitutional_ai']
        metrics = const_data.get('metrics', {})
        print(f"⚖️  Constitutional AI:")
        print(f"   Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"   Applications: {metrics.get('total_applications', 0)}")
        print(f"   Average Improvement: {metrics.get('average_improvement_score', 0):.2f}")
    
    print(f"\n🎉 Enhanced Claude agent demonstration complete!")
    print(f"💡 The agent automatically applied improvements based on quality thresholds and constitutional principles.")


class MockClaudeAgent:
    """Mock enhanced agent implementation for Claude."""
    
    def __init__(self, config, claude_interface):
        self.config = config
        self.claude_interface = claude_interface
        self._performance_metrics = {}
        
        # Initialize improvement engines
        self._improvement_engines = {}
        
        from isopro.improvements.reflexion import ReflexionEngine
        from isopro.improvements.constitutional_ai import ConstitutionalAIEngine
        
        if ImprovementTechnique.REFLEXION in config.improvement_techniques:
            self._improvement_engines[ImprovementTechnique.REFLEXION] = ReflexionEngine(config)
        
        if ImprovementTechnique.CONSTITUTIONAL_AI in config.improvement_techniques:
            self._improvement_engines[ImprovementTechnique.CONSTITUTIONAL_AI] = ConstitutionalAIEngine(config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the enhanced Claude agent."""
        # Apply input improvements
        enhanced_input = self._apply_input_improvements(input_data)
        
        # Get Claude's response
        claude_response = self.claude_interface.generate(enhanced_input)
        
        # Apply output improvements
        enhanced_response = self._apply_output_improvements(claude_response, enhanced_input)
        
        return enhanced_response
    
    def _apply_input_improvements(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvement techniques to input."""
        enhanced_input = input_data.copy()
        
        for technique, engine in self._improvement_engines.items():
            enhanced_input = engine.enhance_input(enhanced_input)
        
        return enhanced_input
    
    def _apply_output_improvements(self, response: Dict[str, Any], 
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvement techniques to Claude's output."""
        enhanced_response = response.copy()
        
        for technique, engine in self._improvement_engines.items():
            enhanced_response = engine.enhance_output(enhanced_response, input_data)
        
        return enhanced_response
    
    def get_improvement_data(self) -> Dict[str, Any]:
        """Get improvement data from all engines."""
        improvement_data = {}
        
        for technique, engine in self._improvement_engines.items():
            improvement_data[technique.value] = engine.get_improvement_data()
        
        return improvement_data


if __name__ == "__main__":
    main()