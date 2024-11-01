import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .car_rl_environment import CarRLEnvironment
from .carviz import CarVisualization

def parse_arguments():
    parser = argparse.ArgumentParser(description="Car RL Simulation and Visualization")
    parser.add_argument("--num_cars", type=int, default=3, help="Number of cars in the simulation")
    parser.add_argument("--time_of_day", type=str, default="08:00", help="Initial time of day (HH:MM format)")
    parser.add_argument("--is_rainy", action="store_true", help="Set initial weather to rainy")
    parser.add_argument("--is_weekday", action="store_true", help="Set initial day to weekday")
    parser.add_argument("--train_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--visualize_episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--load_model", type=str, help="Path to a pre-trained model to load")
    return parser.parse_args()

def make_env(num_cars, time_of_day, is_rainy, is_weekday):
    def _init():
        return CarRLEnvironment(num_cars=num_cars, time_of_day=time_of_day, is_rainy=is_rainy, is_weekday=is_weekday)
    return _init

def main():
    args = parse_arguments()

    # Create the vectorized environment
    env = DummyVecEnv([make_env(args.num_cars, args.time_of_day, args.is_rainy, args.is_weekday)])

    # Create or load the RL agent
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model from {args.load_model}")
        model = PPO.load(args.load_model, env=env)
    else:
        print("Creating and training a new model")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.train_steps)
        
        # Save the trained model
        model.save("car_rl_model")
        print("Model saved as car_rl_model")

    # Run the visualization
    viz = CarVisualization(env, model)
    viz.run_visualization(num_episodes=args.visualize_episodes)

if __name__ == "__main__":
    main()