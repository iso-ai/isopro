import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .car_llm_agent import LLMCarRLWrapper
from .car_rl_environment import CarRLEnvironment 
from .carviz import CarVisualization
from stable_baselines3.common.evaluation import evaluate_policy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Car RL Simulation with LLM Integration and Visualization")
    parser.add_argument("--num_cars", type=int, default=3, help="Number of cars in the simulation")
    parser.add_argument("--time_of_day", type=str, default="08:00", help="Initial time of day (HH:MM format)")
    parser.add_argument("--is_rainy", action="store_true", help="Set initial weather to rainy")
    parser.add_argument("--is_weekday", action="store_true", help="Set initial day to weekday")
    parser.add_argument("--train_steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--visualize_episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--load_model", type=str, help="Path to a pre-trained model to load")
    parser.add_argument("--llm_call_limit", type=int, default=1000, help="Maximum number of LLM API calls")
    parser.add_argument("--llm_call_frequency", type=int, default=100, help="Frequency of LLM calls (in steps)")
    return parser.parse_args()

def make_env(num_cars, time_of_day, is_rainy, is_weekday, llm_call_limit, llm_call_frequency):
    def _init():
        return LLMCarRLWrapper(num_cars=num_cars, time_of_day=time_of_day, is_rainy=is_rainy, 
                               is_weekday=is_weekday, llm_call_limit=llm_call_limit, 
                               llm_call_frequency=llm_call_frequency)
    return _init

def train_and_evaluate(env, total_timesteps, eval_episodes=10):
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, 
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model, mean_reward

def main():
    args = parse_arguments()

    # Ensure the ANTHROPIC_API_KEY is set
    if not os.getenv('ANTHROPIC_API_KEY'):
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # Create the vectorized environment with LLM integration
    env = DummyVecEnv([make_env(args.num_cars, args.time_of_day, args.is_rainy, args.is_weekday, 
                                args.llm_call_limit, args.llm_call_frequency)])

    # Create or load the RL agent
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model from {args.load_model}")
        model = PPO.load(args.load_model, env=env)
    else:
        print("Creating and training a new model")
        model, mean_reward = train_and_evaluate(env, total_timesteps=args.train_steps)
        
        # Save the trained model
        model.save("car_rl_llm_model")
        print("Model saved as car_rl_llm_model")
        print(f"Final mean reward: {mean_reward:.2f}")

    # Run the visualization
    viz = CarVisualization(env, model)
    viz.run_visualization(num_episodes=args.visualize_episodes)

if __name__ == "__main__":
    main()