import gymnasium as gym
from isopro.rl.rl_environment import LLMRLEnvironment
import numpy as np
import anthropic
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMCartPoleWrapper(LLMRLEnvironment):
    def __init__(self, agent_prompt, llm_call_limit: int, api_key: str):
        super().__init__(agent_prompt, None)
        self.cartpole_env = gym.make('CartPole-v1')
        self.action_space = self.cartpole_env.action_space
        self.observation_space = self.cartpole_env.observation_space
        self.client = anthropic.Anthropic(api_key=api_key)
        self.llm_call_count = 0
        self.llm_call_limit = llm_call_limit  # Set the maximum number of LLM calls allowed

    def reset(self, **kwargs):
        # Reset the environment and the LLM call count
        self.llm_call_count = 0
        return self.cartpole_env.reset(**kwargs)

    def step(self, action):
        if self.llm_call_count >= self.llm_call_limit:
            # If the LLM call limit is reached, take a default action (e.g., action = 0)
            logging.warning("LLM call limit reached, default action taken")
            return self.cartpole_env.step(0)  # Default action can be customized

        # Otherwise, proceed with the LLM call and increment the counter
        self.llm_call_count += 1
        return self.cartpole_env.step(action)


    def _llm_decision_to_cartpole_action(self, llm_decision):
        if isinstance(llm_decision, (int, np.integer)):
            return llm_decision
        elif isinstance(llm_decision, str):
            return 0 if "left" in llm_decision.lower() else 1
        else:
            raise ValueError(f"Unexpected action type: {type(llm_decision)}")

    def _update_llm(self, observation, reward, done):
        user_message = f"Observation: {observation}, Reward: {reward}, Done: {done}. What action should we take next?"

        messages = self.conversation_history + [
            {"role": "user", "content": user_message},
        ]

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=150,
            system=self.agent_prompt,
            messages=messages
        )

        ai_response = response.content[0].text
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        logger.debug(f"LLM updated. AI response: {ai_response}")