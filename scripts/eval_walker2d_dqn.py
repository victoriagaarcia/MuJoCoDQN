import gymnasium as gym
import torch
import numpy as np
import os

from src.dqn import QNetwork
from src.envs import DiscreteActionWrapper, PixelStackWrapper

# -----------------------------
# Configuración
# -----------------------------
ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DATE = "Feb10_13_41_43"
MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d.pt"  # ← ajusta esto
VIDEO_DIR = f"runs/{MODEL_DATE}/"  # ← ajusta esto
N_EPISODES = 3

os.makedirs(VIDEO_DIR, exist_ok=True)


def main():
    # 1) Crear entorno base
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DiscreteActionWrapper(env)
    env = PixelStackWrapper(env)

    # 2) Envolver con RecordVideo
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,  # graba TODOS los episodios
        name_prefix="final_video"
    )

    # 3) Cargar modelo
    n_actions = env.action_space.n
    q_net = QNetwork(n_actions).to(DEVICE)
    q_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    q_net.eval()

    # 4) Ejecutar episodios (política greedy)
    for ep in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action = q_net(s).argmax(dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        print(f"Episode {ep} return: {ep_return:.2f}")

    env.close()
    print(f"Videos saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
