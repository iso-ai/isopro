"""
Basic Usage Example

This example demonstrates how to use the ISOPro framework with your own LLM
to create enhanced AI agents using Reflexion, Constitutional AI, DQN, and DPO.
"""

from typing import Dict, Any
import asyncio

from isopro.core import AgentFramework, AgentConfig, AgentType, ImprovementTechnique


# Step 1: Define your LLM interface
class MyLLMInterface:
    """
    Example LLM interface - replace with your own implementation.
    This could be OpenAI, Anthropic, Hugging Face, or any custom model.
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        # Initialize your LLM client here
        # self.client = your_llm_client
    
    def generate(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using your LLM.
        
        Args:
            prompt: Prompt data (format depends on your implementation)
            
        Returns:
            Response data (format depends on your implementation)
        """
        # Example implementation - replace with your LLM call
        if prompt.get('task') == 'reflection':
            # Handle reflection prompts
            return {
                'analysis': 'The response could be more detailed and include examples.',
                'improvements': ['Add specific examples', 'Improve clarity'],
                'improved_response': {
                    'response': 'Enhanced response with better details and examples...',
                    'confidence': 0.9
                }
            }
        elif prompt.get('task') == 'constitutional_critique':
            # Handle constitutional critique
            return {
                'critiques': [
                    {'score': 0.8, 'explanation': 'Good helpfulness'},
                    {'score': 0.6, 'explanation': 'Could be clearer'}
                ]
            }
        else:
            # Handle regular prompts
            input_text = prompt.get('input', prompt.get('prompt', ''))
            return {
                'response': f"Generated response for: {input_text}",
                'confidence': 0.8
            }
    
    async def generate_async(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of generate."""
        # For this example, just call sync version
        return self.generate(prompt)


# Step 2: Define custom evaluation function
def custom_evaluation_function(output_data: Dict[str, Any], 
                             input_data: Dict[str, Any]) -> float:
    """
    Custom evaluation function for Reflexion.
    
    Args:
        output_data: Generated output
        input_data: Original input
        
    Returns:
        Score between 0 and 1
    """
    score = 0.5  # Base score
    
    response_text = output_data.get('response', '')
    
    # Length scoring
    if len(response_text) > 100:
        score += 0.2
    
    # Quality indicators
    if 'confidence' in output_data:
        score += 0.1
    
    if 'reasoning' in output_data:
        score += 0.2
    
    # Check for errors
    if 'error' not in output_data:
        score += 0.1
    
    # Context relevance (simple keyword matching)
    input_text = input_data.get('prompt', '').lower()
    if any(word in response_text.lower() for word in input_text.split()[:5]):
        score += 0.1
    
    return min(score, 1.0)


def main():
    """Main example function."""
    print("🚀 ISOPro Framework - Basic Usage Example")
    print("=" * 50)
    
    # Initialize your LLM interface
    llm_interface = MyLLMInterface()
    
    # Create agent configuration
    config = AgentConfig(
        agent_type=AgentType.LLM,
        model_name="gpt-4",
        improvement_techniques=[
            ImprovementTechnique.REFLEXION,
            ImprovementTechnique.CONSTITUTIONAL_AI
        ],
        custom_parameters={
            # Reflexion configuration
            'reflexion': {
                'llm_interface': llm_interface,
                'evaluation_function': custom_evaluation_function,
                'max_reflections': 2,
                'reflection_threshold': 0.7
            },
            # Constitutional AI configuration
            'constitutional_ai': {
                'llm_interface': llm_interface,
                'principles': [
                    {
                        'id': 'helpfulness',
                        'type': 'helpfulness',
                        'description': 'Provide helpful and actionable responses',
                        'critique_prompt': 'Is this response helpful and actionable?',
                        'revision_prompt': 'Make this response more helpful and actionable.'
                    },
                    {
                        'id': 'clarity',
                        'type': 'clarity', 
                        'description': 'Communicate clearly and concisely',
                        'critique_prompt': 'Is this response clear and easy to understand?',
                        'revision_prompt': 'Make this response clearer and more concise.'
                    }
                ]
            }
        }
    )
    
    # Initialize framework and create agent
    framework = AgentFramework()
    
    # We'll need to implement the specific agent classes
    # For this example, let's create a mock agent
    print("📝 Creating enhanced agent...")
    agent = MockEnhancedAgent(config)
    
    # Test the agent with different inputs
    test_inputs = [
        {
            'prompt': 'Explain quantum computing in simple terms.',
            'context': 'Educational content for beginners'
        },
        {
            'prompt': 'How do I optimize my Python code for better performance?',
            'context': 'Technical guidance request'
        },
        {
            'prompt': 'What are the benefits of meditation?',
            'context': 'Health and wellness inquiry'
        }
    ]
    
    print("\n🧪 Testing agent with various inputs...")
    print("-" * 40)
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n💭 Test {i}: {test_input['prompt'][:50]}...")
        
        # Run the agent
        result = agent.run(test_input)
        
        print(f"📊 Result: {result.get('response', 'No response')[:100]}...")
        
        # Check if improvements were applied
        if '_reflexion_metadata' in result:
            metadata = result['_reflexion_metadata']
            print(f"🔄 Reflexion: {metadata}")
        
        if '_constitutional_metadata' in result:
            metadata = result['_constitutional_metadata']
            print(f"⚖️  Constitutional AI: {metadata}")
    
    # Show agent performance metrics
    print("\n📈 Performance Metrics:")
    print("-" * 25)
    metrics = agent.get_performance_metrics()
    print(f"Agent metrics: {metrics}")
    
    # Show improvement data
    print("\n🔧 Improvement Data:")
    print("-" * 20)
    improvement_data = agent.get_improvement_data()
    for technique, data in improvement_data.items():
        print(f"{technique}: {data.get('metrics', {})}")


class MockEnhancedAgent:
    """
    Mock implementation of EnhancedAgent for demonstration.
    In practice, you'd use the actual EnhancedAgent from the framework.
    """
    
    def __init__(self, config):
        self.config = config
        self._performance_metrics = {}
        
        # Initialize improvement engines
        self._improvement_engines = {}
        
        # Import and initialize improvement engines
        from isopro.improvements.reflexion import ReflexionEngine
        from isopro.improvements.constitutional_ai import ConstitutionalAIEngine
        
        if ImprovementTechnique.REFLEXION in config.improvement_techniques:
            self._improvement_engines[ImprovementTechnique.REFLEXION] = ReflexionEngine(config)
        
        if ImprovementTechnique.CONSTITUTIONAL_AI in config.improvement_techniques:
            self._improvement_engines[ImprovementTechnique.CONSTITUTIONAL_AI] = ConstitutionalAIEngine(config)
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent with improvements."""
        # Apply input improvements
        enhanced_input = self._apply_input_improvements(input_data)
        
        # Simulate core agent logic
        base_response = self._mock_agent_response(enhanced_input)
        
        # Apply output improvements
        enhanced_response = self._apply_output_improvements(base_response, enhanced_input)
        
        return enhanced_response
    
    def _apply_input_improvements(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvement techniques to input."""
        enhanced_input = input_data.copy()
        
        for technique, engine in self._improvement_engines.items():
            enhanced_input = engine.enhance_input(enhanced_input)
        
        return enhanced_input
    
    def _apply_output_improvements(self, response: Dict[str, Any], 
                                 input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvement techniques to output."""
        enhanced_response = response.copy()
        
        for technique, engine in self._improvement_engines.items():
            enhanced_response = engine.enhance_output(enhanced_response, input_data)
        
        return enhanced_response
    
    def _mock_agent_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock agent response for demonstration."""
        prompt = input_data.get('prompt', '')
        
        # Simple mock responses based on prompt
        if 'quantum' in prompt.lower():
            response = "Quantum computing uses quantum mechanics principles to process information."
        elif 'python' in prompt.lower():
            response = "To optimize Python code: use built-in functions, avoid loops where possible, profile your code."
        elif 'meditation' in prompt.lower():
            response = "Meditation can reduce stress, improve focus, and promote emotional well-being."
        else:
            response = f"Thank you for your question about: {prompt}"
        
        return {
            'response': response,
            'confidence': 0.75
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._performance_metrics
    
    def get_improvement_data(self) -> Dict[str, Any]:
        """Get improvement data from all engines."""
        improvement_data = {}
        
        for technique, engine in self._improvement_engines.items():
            improvement_data[technique.value] = engine.get_improvement_data()
        
        return improvement_data


if __name__ == "__main__":
    main()