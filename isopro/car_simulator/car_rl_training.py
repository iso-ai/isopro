import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from .car_rl_environment import CarRLEnvironment

def make_env():
    """Create and return an instance of the CarRLEnvironment."""
    return CarRLEnvironment(num_cars=3, time_of_day="08:00", is_rainy=False, is_weekday=True)

# Create a vectorized environment
env = DummyVecEnv([make_env])

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0)

# Train the agent
total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save the trained model
model.save("car_rl_ppo_model")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones.any():
        obs = env.reset()

env.close()