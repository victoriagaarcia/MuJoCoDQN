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
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) # Normaliza imagen a 84x84
    frame = frame.astype(np.float32) / 255.0 # Normaliza a [0,1]
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
        frame = self.env.render() # Obtiene el frame RGB actual
        p = preprocess(frame, self.size) # Normaliza a grayscale 84x84

        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(p) # Apilamos K frames idénticos al inicio (apilamos 4 para captar movimiento)

        return np.stack(self.frames, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        p = preprocess(frame, self.size)

        self.frames.append(p) # Apilamos el nuevo frame, descartando el más antiguo automáticamente por el maxlen=4

        return np.stack(self.frames, axis=0), reward, terminated, truncated, info


# =========================================================
# Discretización de acciones para Walker2D
# =========================================================
# def make_discrete_action_set(action_dim: int):
#     """
#     Conjunto reducido y justificable de acciones prototipo.
#     Mantener este set fijo para DQN y Rainbow.
#     """
#     Z = np.zeros(action_dim, dtype=np.float32) # acción de "idle" (ninguna acción)
#     P = np.ones(action_dim, dtype=np.float32) # acción de "forward" (empujar hacia adelante)
#     N = -np.ones(action_dim, dtype=np.float32) # acción de "backward" (empujar hacia atrás)

#     half = action_dim // 2

#     P1 = Z.copy(); P1[:half] = 1.0 # empuje mitad 1 (empujar hacia adelante solo la mitad de las articulaciones)
#     P2 = Z.copy(); P2[half:] = 1.0 # empuje mitad 2 (empujar hacia adelante solo la otra mitad de las articulaciones)
#     N1 = Z.copy(); N1[:half] = -1.0 # freno mitad 1 (frenar hacia atrás solo la mitad de las articulaciones)
#     N2 = Z.copy(); N2[half:] = -1.0 # freno mitad 2 (frenar hacia atrás solo la otra mitad de las articulaciones)

#     actions = [
#         Z,          # 0: idle (ninguna acción)
#         0.5 * P,    # 1: forward suave
#         1.0 * P,    # 2: forward fuerte
#         0.5 * N,    # 3: backward suave
#         1.0 * N,    # 4: backward fuerte
#         P1,         # 5: empuje mitad 1
#         P2,         # 6: empuje mitad 2
#         N1,         # 7: freno mitad 1
#         N2,         # 8: freno mitad 2
#     ]

#     return np.stack(actions, axis=0)

def make_discrete_action_set(action_dim: int):
    """
    Acciones discretas suaves y "controlables".
    Incluye:
    - idle
    - empujes globales suaves
    - ajustes individuales por articulación (+/-)
    """
    Z = np.zeros(action_dim, dtype=np.float32)

    # magnitudes suaves (evita 1.0 al inicio)
    # a1 = 0.2
    a1 = 0.4
    # a2 = 0.4
    a2 = 1.0  
    
    actions = [Z]

    # empujes globales suaves (a veces ayuda a avanzar, pero sin reventar)
    actions.append(np.full(action_dim, +a1, dtype=np.float32))
    actions.append(np.full(action_dim, -a1, dtype=np.float32))

    # ajustes por articulación (muy importantes para balance)
    for i in range(action_dim):
        v = Z.copy(); v[i] = +a2
        actions.append(v)
        v = Z.copy(); v[i] = -a2
        actions.append(v)

    return np.stack(actions, axis=0)

# def make_discrete_action_set_legprototype(action_dim: int):
    # Z = np.zeros(action_dim, dtype=np.float32)
    # 
    # # Magnitudes (suaves para evitar inestabilidad al inicio)
    # a = 0.25 
    # b = 0.15 
# 
    # actions = []
# 
    # def add(vector):
    #     actions.append(np.clip(vector, -1.0, 1.0)) # Aseguramos que las acciones estén en el rango [-1, 1]   
    # 
    # # Acción de idle (ninguna acción)
    # add(Z)
    # 
    # # 0) empuje global suave hacia adelante (arranque)
    # add(np.full(action_dim, +b, dtype=np.float32))
    # 
    # # 1) empuje global suave hacia atrás (freno)
    # add(np.full(action_dim, -b, dtype=np.float32))  
    # 
    # # 2) empuja pierna 1 (extiende rodilla + empuja tobillo + hip suave)
    # add(np.array([+b, -a, +a, 0, 0, 0], dtype=np.float32))
# 
    # # 3) empuja pierna 2
    # add(np.array([0, 0, 0, +b, -a, +a], dtype=np.float32))
# 
    # # 4) recupera pierna 1 (flexiona rodilla)
    # add(np.array([0, +a, 0, 0, 0, 0], dtype=np.float32))
# 
    # # 5) recupera pierna 2
    # add(np.array([0, 0, 0, 0, +a, 0], dtype=np.float32))
# 
    # # 6) estabiliza (hips hacia atrás suave para no “tirarse”)
    # add(np.array([-b, 0, 0, -b, 0, 0], dtype=np.float32))
# 
    # return np.stack(actions, axis=0)
    
def make_discrete_action_set_legprototype(action_dim: int):
    Z = np.zeros(action_dim, dtype=np.float32)
    
    # Magnitudes (suaves para evitar inestabilidad al inicio)
    # a = 0.25 
    a = 1.0
    # b = 0.15 
    # b = 0.25
    b = 0.5

    actions = []

    def add(vector):
        actions.append(np.clip(vector, -1.0, 1.0)) # Aseguramos que las acciones estén en el rango [-1, 1]   
    
    # Acción de idle (ninguna acción)
    add(Z)
    
    # # 0) empuje global suave hacia adelante (arranque)
    # add(np.full(action_dim, +b, dtype=np.float32))
    
    # # 1) dobla tobillo pierna 1 (empuje hacia adelante)
    add(np.array([0, 0, 0, 0, 0, +a], dtype=np.float32))
    add(np.array([0, 0, 0, 0, 0, -a], dtype=np.float32))
    
    # dobla tobillo pierna 2 (empuje hacia adelante)
    add(np.array([0, 0, +a, 0, 0, 0], dtype=np.float32))
    add(np.array([0, 0, a, 0, 0, 0], dtype=np.float32))
    
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
    # add(np.array([-b, 0, 0, -b, 0, 0], dtype=np.float32)
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
        assert isinstance(env.action_space, gym.spaces.Box) # Comprobamos que el espacio de acciones original es continuo

        # self._actions = make_discrete_action_set(env.action_space.shape[0]) # Creamos el conjunto de acciones discretas
        self._actions = make_discrete_action_set_legprototype(env.action_space.shape[0]) # Usamos el conjunto de acciones prototipo específico para Walker2D
        self.action_space = gym.spaces.Discrete(self._actions.shape[0]) # Redefinimos el espacio de acciones a discreto con el número de acciones prototipo

    def action(self, act_idx):
        return self._actions[int(act_idx)] # Convertimos el índice de acción discreta en la acción continua correspondiente

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
        z_ref: float = 1.10,          # altura "mínima de seguridad" (no obliga a ir alto, solo evita colapso)
        angle_ref: float = 0.7,      # umbral de inclinación permisivo (radianes aprox)
        w_z: float = 0.7,            # peso penalización altura
        w_ang: float = 0.3,          # peso penalización inclinación
        w_smooth: float = 0.0,        # 0.0 = desactivado (si quieres activarlo: 0.005–0.02)
        alive_bonus: float = 0.2,    # bonus pequeño por seguir vivo
        speed_bonus: float = 0.3,    # bonus suave por velocidad hacia delante (tanh)
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

        # 1) Avance: bonus suave y saturado (no clip duro)
        shaped += self.speed_bonus * float(np.tanh(vx))
        info["debug/speed_bonus"] = self.speed_bonus * float(np.tanh(vx))

        # 2) Seguridad: penaliza solo si está "demasiado bajo" (hinge)
        shaped -= self.w_z * max(0.0, self.z_ref - z)
        info["debug/height_pen"] = self.w_z * max(0.0, self.z_ref - z)

        # 3) Seguridad: penaliza solo si está "demasiado inclinado" (hinge)
        shaped -= self.w_ang * max(0.0, abs(ang) - self.angle_ref)
        info["debug/angle_pen"] = self.w_ang * max(0.0, abs(ang) - self.angle_ref)

        # 4) Suavidad (opcional): si action es vector continuo, penaliza jerk
        #    Si action es discreta (int), w_smooth debería estar a 0.0.
        if self.w_smooth > 0.0:
            a = np.array(action, dtype=np.float32)
            if self.prev_action is not None:
                shaped -= self.w_smooth * float(np.sum((a - self.prev_action) ** 2))
            self.prev_action = a

        # 5) Alive bonus pequeño (densifica sin dominar)
        if not (terminated or truncated):
            shaped += self.alive_bonus
            info["debug/alive_bonus"] = self.alive_bonus
        else: 
            info["debug/alive_bonus"] = 0.0

        return obs, shaped, terminated, truncated, info
