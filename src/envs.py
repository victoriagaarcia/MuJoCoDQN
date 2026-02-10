import gymnasium as gym
import numpy as np
import cv2
from collections import deque

# =========================================================
# Preprocesado de píxeles
# =========================================================
def preprocess(frame, size=84):
    """
    RGB uint8 (H,W,3) -> grayscale float32 (84,84) en [0,1]
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame


class PixelStackWrapper(gym.Wrapper):
    """
    Convierte la observación en un stack de K frames preprocesados
    Shape final: (K, 84, 84)
    """
    def __init__(self, env, k=4, size=84):
        super().__init__(env)
        self.k = k
        self.size = size
        self.frames = deque(maxlen=k)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(k, size, size),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.env.render()
        p = preprocess(frame, self.size)

        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(p)

        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        p = preprocess(frame, self.size)

        self.frames.append(p)

        return np.stack(self.frames, axis=0), reward, terminated, truncated, info


# =========================================================
# Discretización de acciones para Walker2D
# =========================================================
def make_discrete_action_set(action_dim: int):
    """
    Conjunto reducido y justificable de acciones prototipo.
    Mantener este set fijo para DQN y Rainbow.
    """
    Z = np.zeros(action_dim, dtype=np.float32)
    P = np.ones(action_dim, dtype=np.float32)
    N = -np.ones(action_dim, dtype=np.float32)

    half = action_dim // 2

    P1 = Z.copy(); P1[:half] = 1.0
    P2 = Z.copy(); P2[half:] = 1.0
    N1 = Z.copy(); N1[:half] = -1.0
    N2 = Z.copy(); N2[half:] = -1.0

    actions = [
        Z,          # 0: idle
        0.5 * P,    # 1: forward suave
        1.0 * P,    # 2: forward fuerte
        0.5 * N,    # 3: backward suave
        1.0 * N,    # 4: backward fuerte
        P1,         # 5: empuje mitad 1
        P2,         # 6: empuje mitad 2
        N1,         # 7: freno mitad 1
        N2,         # 8: freno mitad 2
    ]

    return np.stack(actions, axis=0)


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Convierte acciones discretas (int) en acciones continuas (Box)
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)

        self._actions = make_discrete_action_set(env.action_space.shape[0])
        self.action_space = gym.spaces.Discrete(self._actions.shape[0])

    def action(self, act_idx):
        return self._actions[int(act_idx)]
