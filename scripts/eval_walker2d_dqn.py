import gymnasium as gym
import torch
import numpy as np
import os

from src.dqn import QNetwork
from src.envs import (
    DiscreteActionWrapper, 
    ForwardAliveSmoothReward, 
    IgnoreAngleTerminationWrapper,
    RGBObsWrapper
)
from .utils import (
    preprocess_rgb_batch_torch
)

# -----------------------------
# Configuración
# -----------------------------
ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DATE = "Feb20_10_17_11"
CHECKPOINT_STEP = "250000"
# MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d.pt"  # ← ajusta esto
MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d_step{CHECKPOINT_STEP}.pt"  # ← ajusta esto
VIDEO_DIR = f"runs/{MODEL_DATE}/"  # ← ajusta esto
N_EPISODES = 3

# Configuracion de reward
ALPHA_RW = 1.5
BETA_RW = 1.0
GAMMA_RW = 0.8
DELTA_RW = 1.0
LAM_RW = 0.05

os.makedirs(VIDEO_DIR, exist_ok=True)

def make_eval_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = ForwardAliveSmoothReward(env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
    env = IgnoreAngleTerminationWrapper(env)
    env = DiscreteActionWrapper(env)
    env = RGBObsWrapper(env)  
    return env

def main():
    # 1) Crear entorno base
    env = make_eval_env()

    # 2) Envolver con RecordVideo
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,  # graba TODOS los episodios
        name_prefix=f"final_video_{CHECKPOINT_STEP}steps"
    )

    # 3) Cargar modelo
    n_actions = env.action_space.n
    q_net = QNetwork(n_actions).to(DEVICE)
    q_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    q_net.eval()

    # 4) Ejecutar episodios (política greedy)
    for ep in range(N_EPISODES):
#        state, _ = env.reset()
#        done = False
#        ep_return = 0.0
#        step = 0
#        
#        while not done:
#            with torch.no_grad():
#                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#                action = q_net(s).argmax(dim=1).item()
#
#            step += 1
#            state, reward, terminated, truncated, info = env.step(action)
#            
#            if terminated or truncated:
#                print("----EPISODE END----")
#                print("terminated:", terminated)
#                print("truncated:", truncated)
#                print("step:", step)
#
#                z =  env.unwrapped.data.qpos[1]
#                angle = env.unwrapped.data.qpos[2]
#                
#                print("torso height (z):", z)
#                print("torso angle:", angle)
#                print("is healthy:", env.unwrapped.is_healthy)
#                
#            done = terminated or truncated
#            ep_return += reward
#
#        print(f"Episode {ep} return: {ep_return:.2f}")
#
#    env.close()
#    print(f"Videos saved in: {VIDEO_DIR}")

        obs, _ = env.reset(seed=42 + 10_000 + ep)   # obs: (H,W,3) uint8 (porque RGBObsWrapper)

        # preprocess + stack inicial
        obs = np.ascontiguousarray(obs)  # Aseguramos que la observación es contigua en memoria para evitar warnings de PyTorch
        frame = preprocess_rgb_batch_torch(obs[None, ...], out_size=84, device="cpu")  # (1,1,84,84) uint8
        state = frame.repeat(1, 4, 1, 1).contiguous()                                  # (1,4,84,84) uint8

        ep_return = 0.0
        step = 0

        while True:
            with torch.no_grad():
                s = state.to(DEVICE, non_blocking=True).float().div_(255.0)  # (1,4,84,84) float
                action = int(q_net(s).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            step += 1

            # update stack
            next_obs = np.ascontiguousarray(next_obs)  # Aseguramos que la observación es contigua en memoria para evitar warnings de PyTorch
            next_frame = preprocess_rgb_batch_torch(next_obs[None, ...], out_size=84, device="cpu")  # (1,1,84,84)
            state = torch.cat([state[:, 1:], next_frame], dim=1).contiguous()

            done = terminated or truncated
            if done:
                print("----EPISODE END----")
                print("terminated:", terminated)
                print("truncated:", truncated)
                print("step:", step)

                # info físico (si está disponible)
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