import gymnasium as gym
import pygame

def main():
    env = gym.make('CarRacing-v3',
                   domain_randomize=False,
                   render_mode="human",
                   continuous=False)

    obs, _ = env.reset()
    
    terminate = False
    while not terminate:
        action = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_LEFT]:
            action = 2
        elif keys[pygame.K_UP]:
            action = 3
        elif keys[pygame.K_DOWN]:
            action = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        terminate = terminated or truncated

    env.close()
    
if __name__ == "__main__":
    main()