import gymnasium as gym
import torch
import numpy as np
import os

from src.dqn import QNetwork
from src.envs import (
    DiscreteActionWrapper, 
    ProgressWithSafetyShaping,
    PixelStackWrapper)

ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DATE = "Feb26_10_30_09"
LOAD_STEP = 14_250_000
MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d_step{LOAD_STEP}.pt"
VIDEO_DIR = f"runs/{MODEL_DATE}/"
N_EPISODES = 20

os.makedirs(VIDEO_DIR, exist_ok=True)


def main():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DiscreteActionWrapper(env)
    env = ProgressWithSafetyShaping(env)
    env = PixelStackWrapper(env)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,
        name_prefix=f"final_video_{LOAD_STEP}steps"
    )

    n_actions = env.action_space.n
    q_net = QNetwork(n_actions).to(DEVICE)
    q_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    q_net.eval()

    for ep in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        step = 0
        
        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action = q_net(s).argmax(dim=1).item()

            step += 1
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("----EPISODE END----")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("step:", step)

                z =  env.unwrapped.data.qpos[1]
                angle = env.unwrapped.data.qpos[2]
                
                print("torso height (z):", z)
                print("torso angle:", angle)
                print("is healthy:", env.unwrapped.is_healthy)
                
            done = terminated or truncated
            ep_return += reward

        print(f"Episode {ep} return: {ep_return:.2f}")

    env.close()
    print(f"Videos saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
