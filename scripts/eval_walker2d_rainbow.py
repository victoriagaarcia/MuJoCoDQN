import gymnasium as gym
import torch
import numpy as np
import os

from src.envs import (
    DiscreteActionWrapper,
    ProgressWithSafetyShaping,
    PixelStackWrapper
)

from src.rainbow import RainbowDQN
from .utils import to_uint8_stack


ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DATE = "Mar09_11_49_29"
CHECKPOINT_STEP = 5_000_000

MODEL_PATH = f"runs/{MODEL_DATE}/rainbow_walker2d.pt"
VIDEO_DIR = f"runs/{MODEL_DATE}/"
N_EPISODES = 5

# Debe coincidir con los parametros de entrenamiento C51.
N_ATOMS = 51
V_MIN = -80.0
V_MAX = 500.0

SIGMA_INIT = 0.017

os.makedirs(VIDEO_DIR, exist_ok=True)


def make_eval_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DiscreteActionWrapper(env)
    env = ProgressWithSafetyShaping(env)
    env = PixelStackWrapper(env, k=4, size=84)
    return env


@torch.no_grad()
def c51_q_values(dist: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    dist: (B, A, N) probs
    support: (N,)
    return: (B, A) Q(s,a) = E[Z]
    """
    return (dist * support.view(1, 1, -1)).sum(dim=-1)


def main():
    env = make_eval_env()

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,
        name_prefix=f"rainbow_video_{CHECKPOINT_STEP}steps"
    )

    n_actions = env.action_space.n

    q_net = RainbowDQN(
        num_actions=n_actions,
        n_atoms=N_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        sigma_init=SIGMA_INIT
    ).to(DEVICE)

    q_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    q_net.eval()

    if hasattr(q_net, "disable_noise"):
        q_net.disable_noise()

    support = q_net.support.to(DEVICE)

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=42 + 10_000 + ep)

        state = to_uint8_stack(obs[None, ...])

        ep_return = 0.0
        step = 0

        while True:
            s = state.to(DEVICE, non_blocking=True).float().div_(255.0)

            dist = q_net(s)
            q_vals = c51_q_values(dist, support)
            action = int(q_vals.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            step += 1

            state = to_uint8_stack(next_obs[None, ...])

            done = terminated or truncated
            if done:
                print("----EPISODE END----")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("step:", step)
                try:
                    z = env.unwrapped.data.qpos[1]
                    angle = env.unwrapped.data.qpos[2]
                    print("torso height (z):", float(z))
                    print("torso angle:", float(angle))
                    print("is healthy:", bool(env.unwrapped.is_healthy))
                except Exception:
                    pass
                break

        print(f"Episode {ep} return: {ep_return:.2f}")

    env.close()
    print(f"Videos saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()