import pygame
import numpy as np
from .car_rl_environment import CarRLEnvironment
from stable_baselines3 import PPO
import math
import random
from datetime import datetime, timedelta

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
ROAD_WIDTH = 800
ROAD_HEIGHT = 600
CAR_WIDTH = 40
CAR_HEIGHT = 20
INFO_BOX_WIDTH = 200
INFO_BOX_HEIGHT = 120
UI_PANEL_WIDTH = 200

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class CarVisualization:
    def __init__(self, env, model):
        self.env = env
        self.unwrapped_env = env.envs[0]
        self.model = model
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Enhanced Car RL Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.rain = [self.RainDrop() for _ in range(100)]
        self.obstacles = [self.Obstacle() for _ in range(5)]
        self.time_of_day = self.float_to_datetime(self.unwrapped_env.time_of_day)

    def float_to_datetime(self, time_float):
        """Convert a float time (0-24) to a datetime object."""
        hours = int(time_float)
        minutes = int((time_float - hours) * 60)
        return datetime.min + timedelta(hours=hours, minutes=minutes)

    def datetime_to_string(self, dt):
        """Convert a datetime object to a string in HH:MM format."""
        return dt.strftime("%H:%M")

    def draw_road(self):
        road_rect = pygame.Rect((SCREEN_WIDTH - ROAD_WIDTH) // 2, (SCREEN_HEIGHT - ROAD_HEIGHT) // 2, ROAD_WIDTH, ROAD_HEIGHT)
        road_color = self.get_road_color()
        pygame.draw.rect(self.screen, road_color, road_rect)
        
        # Draw lane markings
        for i in range(1, 3):
            y = (SCREEN_HEIGHT - ROAD_HEIGHT) // 2 + i * (ROAD_HEIGHT // 3)
            pygame.draw.line(self.screen, WHITE, (road_rect.left, y), (road_rect.right, y), 2)

    def get_road_color(self):
        hour = self.time_of_day.hour
        if 6 <= hour < 18:  # Daytime
            return GRAY
        elif 18 <= hour < 20 or 4 <= hour < 6:  # Dawn/Dusk
            return (150, 150, 170)
        else:  # Night
            return (100, 100, 120)

    def draw_car(self, position, angle, color):
        x, y = position
        x = (x + 1) * ROAD_WIDTH / 2 + (SCREEN_WIDTH - ROAD_WIDTH) // 2
        y = (y + 1) * ROAD_HEIGHT / 2 + (SCREEN_HEIGHT - ROAD_HEIGHT) // 2
        
        car_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, color, (0, 0, CAR_WIDTH, CAR_HEIGHT))
        pygame.draw.polygon(car_surface, BLACK, [(0, 0), (CAR_WIDTH // 2, 0), (0, CAR_HEIGHT)])
        rotated_car = pygame.transform.rotate(car_surface, -math.degrees(angle))
        self.screen.blit(rotated_car, rotated_car.get_rect(center=(x, y)))

    def draw_info_box(self, car_index, position, action, reward):
        x, y = position
        x = (x + 1) * ROAD_WIDTH / 2 + (SCREEN_WIDTH - ROAD_WIDTH) // 2
        y = (y + 1) * ROAD_HEIGHT / 2 + (SCREEN_HEIGHT - ROAD_HEIGHT) // 2
        
        info_box = pygame.Surface((INFO_BOX_WIDTH, INFO_BOX_HEIGHT))
        info_box.fill(WHITE)
        pygame.draw.rect(info_box, BLACK, info_box.get_rect(), 2)
        
        texts = [
            f"Car {car_index + 1}",
            f"Acceleration: {action[0]:.2f}",
            f"Steering: {action[1]:.2f}",
            f"Reward: {reward:.2f}",
            f"Speed: {np.linalg.norm(self.unwrapped_env.cars[car_index]['velocity']):.2f}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, BLACK)
            info_box.blit(text_surface, (10, 10 + i * 25))
        
        self.screen.blit(info_box, (x - INFO_BOX_WIDTH // 2, y - INFO_BOX_HEIGHT - 30))


    def draw_rain(self):
        for drop in self.rain:
            pygame.draw.line(self.screen, (200, 200, 255), (drop.x, drop.y), (drop.x, drop.y + drop.size), drop.size)
            drop.fall()

    def draw_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, YELLOW, ((SCREEN_WIDTH - ROAD_WIDTH) // 2 + obstacle.x, 
                                                   (SCREEN_HEIGHT - ROAD_HEIGHT) // 2 + obstacle.y, 
                                                   obstacle.width, obstacle.height))

    def draw_ui_panel(self):
        panel = pygame.Surface((UI_PANEL_WIDTH, SCREEN_HEIGHT))
        panel.fill(WHITE)
        pygame.draw.rect(panel, BLACK, panel.get_rect(), 2)

        texts = [
            f"Time: {self.datetime_to_string(self.time_of_day)}",
            f"Rainy: {'Yes' if self.unwrapped_env.is_rainy else 'No'}",
            f"Weekday: {'Yes' if self.unwrapped_env.is_weekday else 'No'}",
            "Press keys to change:",
            "T: Time +1 hour",
            "R: Toggle Rain",
            "W: Toggle Weekday"
        ]

        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, BLACK)
            panel.blit(text_surface, (10, 10 + i * 30))

        self.screen.blit(panel, (SCREEN_WIDTH - UI_PANEL_WIDTH, 0))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.time_of_day += timedelta(hours=1)
                    self.unwrapped_env.time_of_day = (self.time_of_day.hour + self.time_of_day.minute / 60) % 24
                elif event.key == pygame.K_r:
                    self.unwrapped_env.is_rainy = not self.unwrapped_env.is_rainy
                elif event.key == pygame.K_w:
                    self.unwrapped_env.is_weekday = not self.unwrapped_env.is_weekday
        return True
    
    class RainDrop:
        def __init__(self):
            self.x = random.randint(0, SCREEN_WIDTH)
            self.y = random.randint(0, SCREEN_HEIGHT)
            self.speed = random.randint(5, 15)
            self.size = random.randint(1, 3)

        def fall(self):
            self.y += self.speed
            if self.y > SCREEN_HEIGHT:
                self.y = 0
                self.x = random.randint(0, SCREEN_WIDTH)

    class Obstacle:
        def __init__(self):
            self.width = random.randint(30, 60)
            self.height = random.randint(30, 60)
            self.x = random.randint(0, ROAD_WIDTH - self.width)
            self.y = random.randint(0, ROAD_HEIGHT - self.height)

    def run_visualization(self, num_episodes=5):
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                if not self.handle_events():
                    return

                self.screen.fill(WHITE)
                self.draw_road()
                self.draw_obstacles()
                if self.unwrapped_env.is_rainy:
                    self.draw_rain()

                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward[0]

                for i, car in enumerate(self.unwrapped_env.cars):
                    position = car["position"].numpy()
                    angle = car["angle"].item()
                    color = (RED, GREEN, BLUE)[i % 3]  # Cycle through colors for different cars
                    self.draw_car(position, angle, color)
                    self.draw_info_box(i, position, action[0][i*2:(i+1)*2], reward[0])

                self.draw_ui_panel()
                pygame.display.flip()
                self.clock.tick(30)
                step += 1

                if done[0]:
                    break

            print(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}")

        pygame.quit()


def main():
    # Create and train the model (you might want to load a pre-trained model instead)
    env = CarRLEnvironment(num_cars=3, time_of_day="08:00", is_rainy=False, is_weekday=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)  # Adjust as needed

    # Create and run the visualization
    viz = CarVisualization(env, model)
    viz.run_visualization()

if __name__ == "__main__":
    main()