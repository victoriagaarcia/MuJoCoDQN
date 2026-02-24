import gymnasium as gym
import numpy as np
import cv2
from collections import deque

def main1():
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    frame = env.render()
    
    print("Observation type:", type(obs))
    print("Frame shape:", frame.shape)
    print("Frame dtype:", frame.dtype)
    
    env.close()
    print("Environment closed successfully.")

def preprocess(frame, size=84):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame

class PixelStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4, size=84):
        super().__init__(env)
        self.k = k
        self.size = size
        self.frames = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(k, size, size), dtype=np.float32)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.env.render()
        processed_frame = preprocess(frame, self.size)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(processed_frame)
        return np.stack(self.frames, axis=0), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        processed_frame = preprocess(frame, self.size)
        self.frames.append(processed_frame)
        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

def main():
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    env = PixelStackWrapper(env, k=4, size=84)
    obs, info = env.reset(seed=0)
    
    print("Stacked observation shape:", obs.shape, "dtype:", obs.dtype, "min:", obs.min(), "max:", obs.max())
    
    # pasos random para comprbar que no falla
    for i in range(3):
        action = env.unwrapped.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward}, terminated={terminated}, truncated={truncated}")
    
    env.close()
    print("Environment closed successfully.")
    
if __name__ == "__main__":
    # main1()
    main()