import gymnasium as gym
from .rl_agent import RLAgent
from .rl_environment import LLMRLEnvironment
from stable_baselines3 import PPO
import numpy as np
import anthropic
import os
import logging
from typing import Optional, Dict, Any
from tqdm import tqdm
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMCartPoleWrapper(LLMRLEnvironment):
    def __init__(self, agent_prompt):
        super().__init__(agent_prompt, None)
        self.cartpole_env = gym.make('CartPole-v1')
        self.action_space = self.cartpole_env.action_space
        self.observation_space = self.cartpole_env.observation_space
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.agent_prompt = agent_prompt
        logger.info("LLMCartPoleWrapper initialized")

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.conversation_history = []
        obs, info = self.cartpole_env.reset(seed=seed, options=options)
        logger.debug(f"Environment reset. Initial observation: {obs}")
        return obs, info

    def step(self, action):
        cartpole_action = self._llm_decision_to_cartpole_action(action)
        observation, reward, terminated, truncated, info = self.cartpole_env.step(cartpole_action)
        self._update_llm(observation, reward, terminated or truncated)
        logger.debug(f"Step taken. Action: {cartpole_action}, Reward: {reward}, Done: {terminated or truncated}")
        return observation, reward, terminated, truncated, info

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

def main():
    # Create output folder
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a unique filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"cartpole_results_{timestamp}.json")
    
    agent_prompt = """You are an AI trained to play the CartPole game. 
    Your goal is to balance a pole on a moving cart for as long as possible. 
    You will receive observations about the cart's position, velocity, pole angle, and angular velocity. 
    Based on these, you should decide whether to move the cart left or right. 
    Respond with 'Move left' or 'Move right' for each decision."""

    env = LLMCartPoleWrapper(agent_prompt)
    rl_agent = RLAgent("LLM_CartPole_Agent", env, algorithm='PPO')

    logger.info("Starting training")
    rl_agent.train(total_timesteps=1)
    logger.info("Training completed")

    test_episodes = 1
    results = []
    
    logger.info("Starting test episodes")
    for episode in tqdm(range(test_episodes), desc="Test Episodes"):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        while not done:
            action, _ = rl_agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        logger.info(f"Episode {episode + 1} completed. Total reward: {total_reward}, Length: {episode_length}")
        results.append({"episode": episode + 1, "total_reward": total_reward, "length": episode_length})

    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Print summary
    average_reward = sum(r['total_reward'] for r in results) / len(results)
    average_length = sum(r['length'] for r in results) / len(results)
    logger.info(f"Test completed. Average reward: {average_reward:.2f}, Average length: {average_length:.2f}")

if __name__ == "__main__":
    main()