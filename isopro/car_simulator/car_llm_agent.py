import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import anthropic
import logging
from typing import List, Dict, Any
from .car_rl_environment import CarRLEnvironment
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMCarRLWrapper(CarRLEnvironment):
    def __init__(self, num_cars=1, time_of_day="12:00", is_rainy=False, is_weekday=True, 
                 agent_prompt="You are an expert driving instructor. Provide concise guidance to improve the RL agent's driving performance.",
                 llm_call_limit=100, llm_call_frequency=100):
        super().__init__(num_cars, time_of_day, is_rainy, is_weekday)
        self.agent_prompt = agent_prompt
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.llm_call_count = 0
        self.llm_call_limit = llm_call_limit
        self.llm_call_frequency = llm_call_frequency
        self.conversation_history: List[Dict[str, str]] = []
        self.step_count = 0
        self.current_guidance = {"action": "unknown"}

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.current_guidance = {"action": "unknown"}
        return super().reset(seed=seed)

    def step(self, action):
        self.step_count += 1
        
        if self.step_count % self.llm_call_frequency == 0 and self.llm_call_count < self.llm_call_limit:
            observation, reward, terminated, truncated, info = super().step(action)
            self.current_guidance = self._get_llm_guidance(observation, reward, terminated)
            self.llm_call_count += 1
        else:
            observation, reward, terminated, truncated, info = super().step(action)

        adjusted_action = self._adjust_action_based_on_guidance(action, self.current_guidance)
        
        return observation, reward, terminated, truncated, info

    def _get_llm_guidance(self, observation, reward, terminated):
        user_message = f"Current state: {observation}, Reward: {reward}, Terminated: {terminated}. Provide brief driving advice."

        messages = self.conversation_history + [
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=50,
                system=self.agent_prompt,
                messages=messages
            )

            ai_response = response.content[0].text
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            logger.debug(f"LLM guidance: {ai_response}")
            return self._parse_llm_guidance(ai_response)
        except Exception as e:
            logger.error(f"Error getting LLM guidance: {e}")
            return {"action": "unknown"}

    def _parse_llm_guidance(self, guidance):
        guidance_lower = guidance.lower()
        actions = {
            "increase speed": {"action": "increase_speed"},
            "decrease speed": {"action": "decrease_speed"},
            "slow down": {"action": "decrease_speed"},
            "turn left": {"action": "turn_left"},
            "turn right": {"action": "turn_right"},
            "stop": {"action": "stop"},
            "start raining": {"environment": "rain", "status": True},
            "increase traffic": {"environment": "traffic", "density": "high"}
        }
        
        for key, value in actions.items():
            if key in guidance_lower:
                return value
        
        return {"action": "unknown"}

    def _adjust_action_based_on_guidance(self, action, guidance):
        adjustments = {
            "increase_speed": (0, 0.1),
            "decrease_speed": (0, -0.1),
            "turn_left": (1, -0.1),
            "turn_right": (1, 0.1),
        }

        if guidance["action"] in adjustments:
            index, adjustment = adjustments[guidance["action"]]
            action[index] = np.clip(action[index] + adjustment, -1.0, 1.0)

        return action

def make_env(llm_call_limit):
    def _init():
        return LLMCarRLWrapper(num_cars=3, time_of_day="08:00", is_rainy=False, is_weekday=True, 
                               llm_call_limit=llm_call_limit)
    return _init

def train_and_evaluate(env, total_timesteps=100000, eval_episodes=10):
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, 
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model, mean_reward

def main():
    llm_call_limit = int(os.getenv('LLM_CALL_LIMIT', '10'))  # Default to 10 if not set

    env = DummyVecEnv([make_env(llm_call_limit)])

    model, mean_reward = train_and_evaluate(env)

    model.save("car_rl_llm_ppo_model")

    logger.info("Training and evaluation completed.")
    logger.info(f"Final mean reward: {mean_reward:.2f}")

if __name__ == "__main__":
    main()