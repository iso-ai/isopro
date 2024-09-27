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
from .llm_cartpole_wrapper import LLMCartPoleWrapper
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    env = LLMCartPoleWrapper(agent_prompt, llm_call_limit=100, api_key=os.getenv("ANTHROPIC_API_KEY"))
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