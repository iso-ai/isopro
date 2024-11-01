import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Union

class CarRLEnvironment(gym.Env):
    def __init__(self, num_cars=1, time_of_day="12:00", is_rainy=False, is_weekday=True):
        super().__init__()
        self.num_cars = num_cars
        self.time_of_day = self.convert_time(time_of_day)
        self.is_rainy = is_rainy
        self.is_weekday = is_weekday
        self.friction = 0.4 if is_rainy else 0.8
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_cars * 2,), dtype=np.float32)
        
        # Observation space: [x, y, vx, vy, angle] for each car + [time_of_day, is_rainy, is_weekday]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_cars * 5 + 3,), 
            dtype=np.float32
        )

        self.cars = self.initialize_cars()

    def convert_time(self, time_of_day: Union[str, float]) -> float:
        """Convert time to a float between 0 and 24."""
        if isinstance(time_of_day, str):
            try:
                hours, minutes = map(int, time_of_day.split(':'))
                return float(hours + minutes / 60.0)
            except ValueError:
                print(f"Invalid time format: {time_of_day}. Using default value of 12:00.")
                return 12.0
        elif isinstance(time_of_day, (int, float)):
            return float(time_of_day) % 24.0
        else:
            print(f"Invalid time format: {time_of_day}. Using default value of 12:00.")
            return 12.0

    def initialize_cars(self) -> List[Dict[str, torch.Tensor]]:
        """Initialize car parameters."""
        return [
            {
                "position": torch.tensor([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=torch.float32),
                "velocity": torch.tensor([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)], dtype=torch.float32),
                "angle": torch.tensor([random.uniform(-np.pi, np.pi)], dtype=torch.float32)
            } for _ in range(self.num_cars)
        ]

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.cars = self.initialize_cars()
        return self.get_observation(), {}

    def get_observation(self) -> np.ndarray:
        """Get the current observation of the environment."""
        car_obs = np.concatenate([
            np.concatenate([
                car["position"].numpy(),
                car["velocity"].numpy(),
                car["angle"].numpy()
            ]) for car in self.cars
        ])
        env_obs = np.array([
            self.time_of_day,
            float(self.is_rainy),
            float(self.is_weekday)
        ], dtype=np.float32)
        return np.concatenate([car_obs, env_obs]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Array of actions for all cars [acceleration1, steering1, acceleration2, steering2, ...]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Ensure action is the correct shape
        action = np.array(action).flatten()
        if action.shape[0] != self.num_cars * 2:
            raise ValueError(f"Action shape {action.shape} does not match expected shape ({self.num_cars * 2},)")

        for i in range(self.num_cars):
            car_action = action[i*2:(i+1)*2]
            self.apply_action(self.cars[i], car_action)
            self.update_physics(self.cars[i])

        observation = self.get_observation()
        reward = self.calculate_reward()
        terminated = self.is_terminated()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def apply_action(self, car: Dict[str, torch.Tensor], action: np.ndarray):
        """Apply the RL agent's action to the car."""
        if len(action) != 2:
            raise ValueError(f"Expected action to have 2 values, got {len(action)}")
        
        acceleration, steering = action
        car["velocity"] += torch.tensor([acceleration, 0.0], dtype=torch.float32) * 0.1  # Scale down the acceleration
        car["angle"] += torch.tensor([steering], dtype=torch.float32) * 0.1  # Scale down the steering

    def update_physics(self, car: Dict[str, torch.Tensor], dt: float = 0.1):
        """Update car position and velocity using physics simulation."""
        # Update velocity (apply friction)
        car["velocity"] *= (1 - self.friction * dt)
        
        # Update position
        car["position"] += car["velocity"] * dt
        
        # Apply steering
        angle = car["angle"].item()
        rotation_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=torch.float32)
        car["velocity"] = torch.matmul(rotation_matrix, car["velocity"])
        
        # Bound the position to keep cars on the screen
        car["position"] = torch.clamp(car["position"], -1, 1)

    def calculate_reward(self) -> float:
        """Calculate the reward based on the current state."""
        reward = 0.0
        for car in self.cars:
            # Reward for moving
            speed = torch.norm(car["velocity"]).item()
            reward += speed * 0.1
            
            # Penalty for being close to the edge
            distance_from_center = torch.norm(car["position"]).item()
            reward -= distance_from_center * 0.1
        
        return reward

    def is_terminated(self) -> bool:
        """Check if the episode should be terminated."""
        for car in self.cars:
            if torch.any(torch.abs(car["position"]) > 1):
                return True
        return False

    def render(self):
        """Render the environment (placeholder for potential future implementation)."""
        pass