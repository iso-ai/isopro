"""Goal-directed car RL environment.

Improvements over the original:
  - Cars now have a goal position they must reach (reward is goal-directed).
  - Reward = progress toward goal minus collision/boundary penalty.
    The original rewarded raw speed and punished distance from center,
    which made spinning in place the optimal policy.
  - Proper physics: velocity is clamped to a max speed; steering modifies
    the heading angle; acceleration is applied along the heading vector.
  - Episode termination: car reaches goal (success) OR exceeds step budget.
  - Collision detection: hard boundary hit triggers a penalty and termination.
  - Observation now includes [x, y, vx, vy, angle, goal_dx, goal_dy] per car,
    giving the policy the information it needs to navigate.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..environments.base_env import BaseEnvironment, EpisodeResult


class CarRLEnvironment(BaseEnvironment, gym.Env):
    """Physics-based multi-car driving environment.

    Each car must navigate from a random start position to a goal position.
    The episode ends when all cars reach their goals, any car leaves the
    bounded arena, or max_steps is exceeded.

    Observation per car (7 values):
        [x, y, vx, vy, angle, goal_dx, goal_dy]
        where goal_dx/dy is the normalized vector from car to goal.

    Action per car (2 values, in [-1, 1]):
        [acceleration, steering]

    Args:
        num_cars: Number of cars in the simulation.
        time_of_day: Time string "HH:MM" or float hour. Affects friction.
        is_rainy: Reduces friction when True.
        is_weekday: Currently informational only; stored in observation.
        max_steps: Maximum steps before episode is truncated.
        goal_radius: Distance threshold for reaching the goal (arena units).
        max_speed: Maximum speed magnitude for any car.
    """

    # Per-car observation size: [x, y, vx, vy, angle, goal_dx, goal_dy]
    _OBS_PER_CAR: int = 7
    # Context vars appended after car observations: [time_of_day, is_rainy, is_weekday]
    _CONTEXT_DIM: int = 3

    def __init__(
        self,
        num_cars: int = 1,
        time_of_day: Union[str, float] = "12:00",
        is_rainy: bool = False,
        is_weekday: bool = True,
        max_steps: int = 200,
        goal_radius: float = 0.08,
        max_speed: float = 0.5,
    ) -> None:
        BaseEnvironment.__init__(self, backend=None)

        self.num_cars = num_cars
        self.time_of_day = self._convert_time(time_of_day)
        self.is_rainy = is_rainy
        self.is_weekday = is_weekday
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.max_speed = max_speed

        # Friction: lower when rainy (less grip). Also varies by time of day
        # (night → slightly reduced visibility/grip modelled as 5% lower).
        base_friction = 0.3 if is_rainy else 0.15
        night_penalty = 0.05 if (self.time_of_day < 6.0 or self.time_of_day > 20.0) else 0.0
        self.friction = min(base_friction + night_penalty, 0.5)

        # Gymnasium spaces.
        obs_dim = num_cars * self._OBS_PER_CAR + self._CONTEXT_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_cars * 2,), dtype=np.float32
        )

        self.cars: List[Dict[str, torch.Tensor]] = []
        self.goals: List[torch.Tensor] = []
        self._step_count: int = 0
        self._goals_reached: List[bool] = []
        self._prev_distances: List[float] = []

    # ------------------------------------------------------------------
    # gym.Env + BaseEnvironment interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """Reset all cars to random start positions with new random goals.

        Args:
            seed: Random seed for reproducibility.
            options: Unused; included for gym.Env compatibility.

        Returns:
            Tuple of (observation_array, info_dict).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._step_count = 0
        self.cars = self._initialize_cars()
        self.goals = self._initialize_goals()
        self._goals_reached = [False] * self.num_cars
        self._prev_distances = [
            self._distance_to_goal(car, goal)
            for car, goal in zip(self.cars, self.goals)
        ]

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple:
        """Apply actions, update physics, compute reward.

        Args:
            action: Flat array of [accel_1, steer_1, accel_2, steer_2, ...].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).

        Raises:
            ValueError: If action shape does not match num_cars.
        """
        action = np.array(action, dtype=np.float32).flatten()
        if action.shape[0] != self.num_cars * 2:
            raise ValueError(
                f"Action shape {action.shape} does not match "
                f"expected ({self.num_cars * 2},)."
            )

        self._step_count += 1
        total_reward = 0.0
        any_collision = False

        for i in range(self.num_cars):
            if self._goals_reached[i]:
                continue

            car_action = action[i * 2: (i + 1) * 2]
            self._apply_action(self.cars[i], car_action)
            self._update_physics(self.cars[i])

            # Check goal.
            dist = self._distance_to_goal(self.cars[i], self.goals[i])
            if dist < self.goal_radius:
                self._goals_reached[i] = True
                total_reward += 5.0  # Sparse success bonus.

            # Reward shaping: reward progress toward goal.
            progress = self._prev_distances[i] - dist
            total_reward += progress * 2.0

            # Penalty for boundary collision.
            pos = self.cars[i]["position"]
            if torch.any(torch.abs(pos) >= 1.0):
                total_reward -= 2.0
                any_collision = True

            self._prev_distances[i] = dist

        # Small time penalty to incentivize efficiency.
        total_reward -= 0.01

        observation = self._get_observation()
        terminated = all(self._goals_reached) or any_collision
        truncated = self._step_count >= self.max_steps

        info = {
            "goals_reached": sum(self._goals_reached),
            "step": self._step_count,
            "distances": [
                self._distance_to_goal(car, goal)
                for car, goal in zip(self.cars, self.goals)
            ],
        }
        return observation, float(total_reward), terminated, truncated, info

    def _get_action(self, observation: np.ndarray, config=None) -> np.ndarray:
        """Default policy: random actions from the action space.

        Override with a trained RL policy in subclasses.

        Args:
            observation: Current environment observation (unused).
            config: Ignored.

        Returns:
            Random action array.
        """
        return self.action_space.sample()

    def compute_metrics(self) -> dict:
        """Compute episode-level metrics.

        Returns:
            Dict with success rate, mean final distance, and step count.
        """
        rewards = [s.reward for s in self._step_results]
        return {
            "goals_reached": sum(self._goals_reached),
            "success_rate": sum(self._goals_reached) / max(self.num_cars, 1),
            "steps_taken": self._step_count,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "total_reward": float(sum(rewards)) if rewards else 0.0,
        }

    def render(self) -> None:
        """Print a simple ASCII representation of car and goal positions."""
        for i, (car, goal) in enumerate(zip(self.cars, self.goals)):
            pos = car["position"].numpy()
            vel = car["velocity"].numpy()
            goal_np = goal.numpy()
            dist = self._distance_to_goal(car, goal)
            reached = self._goals_reached[i]
            print(
                f"Car {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}) "
                f"vel=({vel[0]:.2f}, {vel[1]:.2f}) "
                f"goal=({goal_np[0]:.2f}, {goal_np[1]:.2f}) "
                f"dist={dist:.3f} {'[REACHED]' if reached else ''}"
            )

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _apply_action(self, car: Dict[str, torch.Tensor], action: np.ndarray) -> None:
        """Apply acceleration and steering to a car.

        Acceleration is applied along the car's current heading vector.
        Steering changes the heading angle.

        Args:
            car: Car state dict with 'position', 'velocity', 'angle'.
            action: [acceleration, steering] each in [-1, 1].
        """
        acceleration, steering = float(action[0]), float(action[1])

        # Update heading angle.
        car["angle"] += torch.tensor([steering * 0.1], dtype=torch.float32)

        # Apply acceleration in the heading direction.
        angle = car["angle"].item()
        heading = torch.tensor(
            [np.cos(angle), np.sin(angle)], dtype=torch.float32
        )
        car["velocity"] += heading * (acceleration * 0.05)

    def _update_physics(self, car: Dict[str, torch.Tensor], dt: float = 0.1) -> None:
        """Integrate car physics for one timestep.

        Args:
            car: Car state dict.
            dt: Timestep size in seconds.
        """
        # Friction decelerates the car.
        car["velocity"] *= (1.0 - self.friction * dt)

        # Clamp speed to max_speed.
        speed = torch.norm(car["velocity"]).item()
        if speed > self.max_speed:
            car["velocity"] = car["velocity"] / speed * self.max_speed

        # Integrate position.
        car["position"] += car["velocity"] * dt

        # Soft boundary: clamp position and zero velocity on wall hit.
        for dim in range(2):
            if car["position"][dim].abs() >= 1.0:
                car["position"][dim] = torch.clamp(car["position"][dim], -1.0, 1.0)
                car["velocity"][dim] = torch.tensor(0.0)

    # ------------------------------------------------------------------
    # Observation and initialization
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Construct the flat observation vector for all cars.

        Per-car features: [x, y, vx, vy, angle, goal_dx, goal_dy]
        Goal direction (goal_dx, goal_dy) is the unit vector from car to goal.

        Returns:
            Float32 array of shape (num_cars * 7 + 3,).
        """
        car_obs_parts = []
        for car, goal in zip(self.cars, self.goals):
            goal_vec = goal - car["position"]
            dist = goal_vec.norm().item()
            goal_dir = goal_vec / max(dist, 1e-6)

            car_obs_parts.append(np.concatenate([
                car["position"].numpy(),
                car["velocity"].numpy(),
                car["angle"].numpy(),
                goal_dir.numpy(),
            ]))

        context = np.array(
            [self.time_of_day / 24.0, float(self.is_rainy), float(self.is_weekday)],
            dtype=np.float32,
        )
        return np.concatenate(car_obs_parts + [context]).astype(np.float32)

    def _initialize_cars(self) -> List[Dict[str, torch.Tensor]]:
        """Initialize cars at random positions with zero velocity.

        Returns:
            List of car state dicts.
        """
        return [
            {
                "position": torch.tensor(
                    [random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)],
                    dtype=torch.float32,
                ),
                "velocity": torch.zeros(2, dtype=torch.float32),
                "angle": torch.tensor(
                    [random.uniform(-np.pi, np.pi)], dtype=torch.float32
                ),
            }
            for _ in range(self.num_cars)
        ]

    def _initialize_goals(self) -> List[torch.Tensor]:
        """Initialize goal positions, ensuring they differ from start positions.

        Returns:
            List of goal position tensors.
        """
        goals = []
        for car in self.cars:
            while True:
                goal = torch.tensor(
                    [random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)],
                    dtype=torch.float32,
                )
                dist = (goal - car["position"]).norm().item()
                if dist > 0.3:  # Ensure goal is not trivially close to start.
                    break
            goals.append(goal)
        return goals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_to_goal(car: Dict[str, torch.Tensor], goal: torch.Tensor) -> float:
        """Compute Euclidean distance from a car to its goal.

        Args:
            car: Car state dict.
            goal: Goal position tensor.

        Returns:
            Scalar distance.
        """
        return float((goal - car["position"]).norm().item())

    @staticmethod
    def _convert_time(time_of_day: Union[str, float]) -> float:
        """Convert time to a float hour in [0, 24).

        Args:
            time_of_day: "HH:MM" string or numeric hour.

        Returns:
            Float hour.
        """
        if isinstance(time_of_day, str):
            try:
                hours, minutes = map(int, time_of_day.split(":"))
                return float(hours + minutes / 60.0)
            except ValueError:
                return 12.0
        return float(time_of_day) % 24.0
