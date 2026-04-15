import gymnasium as gym
import numpy as np
import cv2
from collections import deque 


# Preprocesado de píxeles
def preprocess(frame, size=84):
    """
    RGB uint8 (H,W,3) -> grayscale float32 (84,84) en [0,1]
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Normaliza imagen a 84x84
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    # Normaliza a [0,1]
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
        _, info = self.env.reset(**kwargs)
        # Obtiene el frame RGB actual
        frame = self.env.render() 
        # Normaliza a grayscale 84x84
        p = preprocess(frame, self.size) 

        self.frames.clear()
        for _ in range(self.k):
            # Apilamos K frames idénticos al inicio (apilamos 4 para captar movimiento)
            self.frames.append(p) 

        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        p = preprocess(frame, self.size)
        # Apilamos el nuevo frame, descartando el más antiguo automáticamente por el maxlen=4
        self.frames.append(p) 

        return np.stack(self.frames, axis=0), reward, terminated, truncated, info

    
def make_discrete_action_set_legprototype(action_dim: int):
    Z = np.zeros(action_dim, dtype=np.float32)
    
    # Magnitudes 
    a = 1.0
    b = 0.5

    actions = []

    def add(vector):
        # Aseguramos que las acciones estén en el rango [-1, 1]
        actions.append(np.clip(vector, -1.0, 1.0))   
    
    # Acción de idle (ninguna acción)
    add(Z) 
    
    # # 1) dobla tobillo pierna 1 (empuje hacia adelante)
    add(np.array([0, 0, 0, 0, 0, +a], dtype=np.float32))
    add(np.array([0, 0, 0, 0, 0, -a], dtype=np.float32))
    
    # dobla tobillo pierna 2 (empuje hacia adelante)
    add(np.array([0, 0, +a, 0, 0, 0], dtype=np.float32))
    add(np.array([0, 0, -a, 0, 0, 0], dtype=np.float32))
    
    # 2) empuja pierna 1 (extiende rodilla + empuja tobillo + hip suave)
    add(np.array([0, -a, +a, 0, 0, 0], dtype=np.float32))
    add(np.array([0, +a, -a, 0, 0, 0], dtype=np.float32))
    
    # 3) empuja pierna 2
    add(np.array([0, 0, 0, 0, -a, +a], dtype=np.float32))
    add(np.array([0, 0, 0, 0, +a, -a], dtype=np.float32))
    
    # 4) recupera pierna 1 (flexiona rodilla)
    add(np.array([0, +a, 0, 0, 0, 0], dtype=np.float32))
    add(np.array([0, -a, 0, 0, 0, 0], dtype=np.float32))

    # 5) recupera pierna 2 (flexiona rodilla)
    add(np.array([0, 0, 0, 0, +a, 0], dtype=np.float32))
    add(np.array([0, 0, 0, 0, -a, 0], dtype=np.float32))

    # 6) estabiliza (hips hacia atrás suave para no “tirarse”)
    # Hips fuertes (1.0)
    add(np.array([0, 0, 0, +a, 0, 0], dtype=np.float32))
    add(np.array([0, 0, 0, -a, 0, 0], dtype=np.float32))
    add(np.array([+a, 0, 0, 0, 0, 0], dtype=np.float32))
    add(np.array([-a, 0, 0, 0, 0, 0], dtype=np.float32))
    # Hips suaves (0.5)
    add(np.array([0, 0, 0, +b, 0, 0], dtype=np.float32))
    add(np.array([0, 0, 0, -b, 0, 0], dtype=np.float32))
    add(np.array([+b, 0, 0, 0, 0, 0], dtype=np.float32))
    add(np.array([-b, 0, 0, 0, 0, 0], dtype=np.float32))
    
    return np.stack(actions, axis=0)
    

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Convierte acciones discretas (int) en acciones continuas (Box)
    """
    def __init__(self, env):
        super().__init__(env)
        # Comprobamos que el espacio de acciones original es continuo
        assert isinstance(env.action_space, gym.spaces.Box) 
        # Usamos el conjunto de acciones prototipo específico para Walker2D
        self._actions = make_discrete_action_set_legprototype(env.action_space.shape[0])
        # Redefinimos el espacio de acciones a discreto con el número de acciones prototipo
        self.action_space = gym.spaces.Discrete(self._actions.shape[0])

    def action(self, act_idx):
        # Convertimos el índice de acción discreta en la acción continua correspondiente
        return self._actions[int(act_idx)] 

class ProgressWithSafetyShaping(gym.Wrapper):
    """
    Wrapper de reward shaping PARA Walker2d (o similares MuJoCo) pensado para:
      - Mantener la reward por defecto como base (forward + survive - ctrl_cost)
      - Incentivar avance SIN forzar postura "bonita"
      - Penalizar solo situaciones que suelen acabar en caída (altura baja / inclinación extrema)
      - (Opcional) suavizar cambios bruscos de acción para estabilizar marcha

    Recomendación: úsalo junto a tu IgnoreAngleTerminationWrapper si quieres episodios menos binarios.
    """

    def __init__(
        self,
        env,
        z_ref: float = 1.10,          
        angle_ref: float = 0.7,      
        w_z: float = 0.7,            
        w_ang: float = 0.3,          
        w_smooth: float = 0.0,        
        alive_bonus: float = 0.2,    
        speed_bonus: float = 0.3,    
    ):
        super().__init__(env)
        self.z_ref = float(z_ref)
        self.angle_ref = float(angle_ref)
        self.w_z = float(w_z)
        self.w_ang = float(w_ang)
        self.w_smooth = float(w_smooth)
        self.alive_bonus = float(alive_bonus)
        self.speed_bonus = float(speed_bonus)

        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Base: reward por defecto del entorno
        shaped = float(reward)
        info["debug/base"] = shaped

        # Estado interno MuJoCo
        data = self.env.unwrapped.data
        z = float(data.qpos[1])       # altura torso
        ang = float(data.qpos[2])     # ángulo torso
        vx = float(data.qvel[0])      # velocidad x

        # 1) Avance: bonus suave y saturado
        shaped += self.speed_bonus * float(np.tanh(vx))
        info["debug/speed_bonus"] = self.speed_bonus * float(np.tanh(vx))

        # 2) Seguridad: penaliza solo si está "demasiado bajo" 
        shaped -= self.w_z * max(0.0, self.z_ref - z)
        info["debug/height_pen"] = self.w_z * max(0.0, self.z_ref - z)

        # 3) Seguridad: penaliza solo si está "demasiado inclinado"
        shaped -= self.w_ang * max(0.0, abs(ang) - self.angle_ref)
        info["debug/angle_pen"] = self.w_ang * max(0.0, abs(ang) - self.angle_ref)

        # 4) Alive bonus pequeño
        if not (terminated or truncated):
            shaped += self.alive_bonus
            info["debug/alive_bonus"] = self.alive_bonus
        else: 
            info["debug/alive_bonus"] = 0.0

        return obs, shaped, terminated, truncated, info