import gymnasium as gym
import torch
import time
from src.dqn import QNetwork
from src.envs import PixelStackWrapper, DiscreteActionWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_ID = "Walker2d-v5"
MODEL_PATH = "runs/dqn_walker2d_final.pt"  # lo ajustamos luego

class IgnoreAngleTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accedemos al estado interno MuJoCo
        z = self.env.unwrapped.data.qpos[1]       # altura
        angle = self.env.unwrapped.data.qpos[2]   # ángulo torso

        # Rango saludable original de altura
        healthy_z_range = self.env.unwrapped._healthy_z_range

        # NUEVA condición: solo depende de altura
        healthy_z = healthy_z_range[0] < z < healthy_z_range[1]

        # Ignoramos condición del ángulo
        terminated = not healthy_z

        return obs, reward, terminated, truncated, info

def main():
    env = gym.make(ENV_ID, render_mode="human")
    env = IgnoreAngleTerminationWrapper(env)
    env = DiscreteActionWrapper(env)
    env = PixelStackWrapper(env)

    n_actions = env.action_space.n
    model = QNetwork(n_actions).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    state, _ = env.reset()

    while True:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = model(s).argmax(dim=1).item()

        state, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.03)  # para que no vaya demasiado rápido

        if terminated or truncated:
            state, _ = env.reset()

if __name__ == "__main__":
    main()
