import gymnasium as gym
import torch
import numpy as np
import os

from src.envs_antiguo import (
    DiscreteActionWrapper,
    ProgressWithSafetyShaping,
    PixelStackWrapper
)

from src.rainbow import RainbowDQN  # <-- tu red rainbow (C51)
from .utils import to_uint8_stack


# -----------------------------
# Configuración
# -----------------------------
ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DATE = "Mar09_11_49_29"   # ajusta
CHECKPOINT_STEP = 5_000_000     # ajusta

# MODEL_PATH = f"runs/{MODEL_DATE}/rainbow_walker2d_step{CHECKPOINT_STEP}.pt"  # ajusta nombre si guardas distinto
MODEL_PATH = f"runs/{MODEL_DATE}/rainbow_walker2d.pt"  # carpeta separada para videos de evaluación
VIDEO_DIR = f"runs/{MODEL_DATE}/"
N_EPISODES = 5

# C51 params: deben coincidir con training
N_ATOMS = 51
V_MIN = -80.0
# V_MAX = 200.0
V_MAX = 500.0

# Noisy init (debe coincidir si tu ctor lo pide)
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
    # 1) Crear entorno base
    env = make_eval_env()

    # 2) Envolver con RecordVideo
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,
        name_prefix=f"rainbow_video_{CHECKPOINT_STEP}steps"
    )

    # 3) Cargar modelo
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

    # Evaluación determinista (si tu NoisyLinear lo soporta)
    if hasattr(q_net, "disable_noise"):
        q_net.disable_noise()

    support = q_net.support.to(DEVICE)  # (N,)

    # 4) Ejecutar episodios (política greedy)
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=42 + 10_000 + ep)

        # En vuestro DQN ya usáis to_uint8_stack con PixelStackWrapper
        state = to_uint8_stack(obs[None, ...])  # (1,4,84,84) uint8

        ep_return = 0.0
        step = 0

        while True:
            s = state.to(DEVICE, non_blocking=True).float().div_(255.0)  # (1,4,84,84)

            # forward distributional -> (1, A, N)
            dist = q_net(s)  # asumo que devuelve probs; si devuelve logits, abajo te digo el fix
            q_vals = c51_q_values(dist, support)  # (1,A)
            action = int(q_vals.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            step += 1

            state = to_uint8_stack(next_obs[None, ...])  # (1,4,84,84) uint8

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