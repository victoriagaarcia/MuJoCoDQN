import gymnasium as gym
import numpy as np
import cv2
from collections import deque 

# =========================================================
# Preprocesado de píxeles
# =========================================================
import cv2
import numpy as np
import gymnasium as gym
from scripts.utils import preprocess

class Gray84ObsWrapper(gym.ObservationWrapper):
    """
    Devuelve observación grayscale 84x84 uint8.
    output: (84,84) uint8
    """
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(size, size), dtype=np.uint8
        )

    def observation(self, obs):
        frame = self.env.render()  # (H,W,3) uint8
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return gray.astype(np.uint8)


class RGBObsWrapper(gym.ObservationWrapper):
    """
    Convierte la observación en un frame RGB preprocesado
    """
    def __init__(self, env):
        super().__init__(env)
        # self.env.reset()
        # frame = self.env.render() # Renderizamos un frame para obtener su tamaño original
        h, w, c = 480, 480, 3 # Asumimos que el renderizado es RGB con tamaño 480x480 (ajustar si es diferente)
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return self.env.render()


class ReduceAngleTerminationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accedemos al estado interno MuJoCo
        z = self.env.unwrapped.data.qpos[1]       # altura
        angle = self.env.unwrapped.data.qpos[2]   # ángulo torso

        # Rango saludable original de altura
        healthy_z_range = self.env.unwrapped._healthy_z_range
        
        # Bajar el umbral inferior de altura para permitir más flexibilidad 
        lower_z = healthy_z_range[0] * 0.9  # por ejemplo, un 20% más bajo
        
        # Cambiar el rango saludable para el ángulo a algo más permisivo 
        healthy_angle_range = (-1.5, 1.5)

        # NUEVA condición:
        healthy_z = lower_z < z < healthy_z_range[1] and healthy_angle_range[0] < angle < healthy_angle_range[1]

        # Ignoramos condición del ángulo
        terminated = not healthy_z

        return obs, reward, terminated, truncated, info


class ForwardAliveSmoothReward(gym.Wrapper):
    def __init__(self, env, alpha=2.0, beta=1.0, gamma=0.6, delta=1.0, lam=0.05):
        super().__init__(env)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lam = lam
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        # Intentamos leer términos del info (si tu gym los expone)
        forward = info.get("reward_forward", None)
        ctrl = info.get("reward_ctrl", None)

        # Fallbacks robustos
        vx = float(self.env.unwrapped.data.qvel[0])  # velocidad x torso
        if forward is None:
            forward = vx
        if ctrl is None:
            # ctrl = 0.0
            ctrl = 0.001 * float(np.sum(np.square(a)))

        # Healthy: usamos el criterio interno del entorno
        healthy = 1.0 if getattr(self.env.unwrapped, "is_healthy", True) else 0.0
        # En algunas versiones is_healthy es @property
        try:
            healthy = 1.0 if self.env.unwrapped.is_healthy else 0.0
        except Exception:
            pass

        # Penaliza ir hacia atrás
        backward_pen = max(0.0, -vx)

        # Penaliza cambios bruscos de acción (suavidad)
        a = np.array(action, dtype=np.float32)
        smooth_pen = 0.0
        if self.prev_action is not None:
            smooth_pen = float(np.sum((a - self.prev_action) ** 2))
        self.prev_action = a

        new_reward = (
            # self.alpha * 1.0 * (vx > 0.0)
            self.alpha * max(0.0, vx)
            + self.beta * float(healthy)
            - self.gamma * float(ctrl)
            - self.delta * float(backward_pen)
            - self.lam * float(smooth_pen)
        )

        # Debug en info (útil para análisis)
        info["shaping/forward"] = float(forward)
        info["shaping/healthy"] = float(healthy)
        info["shaping/ctrl"] = float(ctrl)
        info["shaping/backward_pen"] = float(backward_pen)
        info["shaping/smooth_pen"] = float(smooth_pen)
        info["shaping/reward"] = float(new_reward)

        return obs, new_reward, terminated, truncated, info

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
        angle_ref: float = 1.20,      # umbral de inclinación permisivo (radianes aprox)
        w_z: float = 0.07,            # peso penalización altura
        w_ang: float = 0.03,          # peso penalización inclinación
        w_smooth: float = 0.0,        # 0.0 = desactivado (si quieres activarlo: 0.005–0.02)
        alive_bonus: float = 0.02,    # bonus pequeño por seguir vivo
        speed_bonus: float = 0.03,    # bonus suave por velocidad hacia delante (tanh)
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

        # Estado interno MuJoCo
        data = self.env.unwrapped.data
        z = float(data.qpos[1])       # altura torso
        ang = float(data.qpos[2])     # ángulo torso
        vx = float(data.qvel[0])      # velocidad x

        # 1) Avance: bonus suave y saturado (no clip duro)
        shaped += self.speed_bonus * float(np.tanh(vx))

        # 2) Seguridad: penaliza solo si está "demasiado bajo" (hinge)
        shaped -= self.w_z * max(0.0, self.z_ref - z)

        # 3) Seguridad: penaliza solo si está "demasiado inclinado" (hinge)
        shaped -= self.w_ang * max(0.0, abs(ang) - self.angle_ref)

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

        return obs, shaped, terminated, truncated, info

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
            low=0,
            high=255,
            shape=(k, size, size),
            dtype=np.uint8,
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
    # a1 = 0.25
    a1 = 0.4
    
    # a2 = 0.85
    a2 = 1.0  
    # chat sugiere a1=0.25 y a2=0.35 ¿?
    
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
    
    add(np.array([0, 0, 0, +a, 0, 0], dtype=np.float32))
    add(np.array([0, 0, 0, -a, 0, 0], dtype=np.float32))
    add(np.array([+a, 0, 0, 0, 0, 0], dtype=np.float32))
    add(np.array([-a, 0, 0, 0, 0, 0], dtype=np.float32))
    
    # add(np.array([0, 0, 0, +b, 0, 0], dtype=np.float32))
    # add(np.array([0, 0, 0, -b, 0, 0], dtype=np.float32))
    # add(np.array([+b, 0, 0, 0, 0, 0], dtype=np.float32))
    # add(np.array([-b, 0, 0, 0, 0, 0], dtype=np.float32))
    
    return np.stack(actions, axis=0)
    

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Convierte acciones discretas (int) en acciones continuas (Box)
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box) # Comprobamos que el espacio de acciones original es continuo

        self._actions = make_discrete_action_set(env.action_space.shape[0]) # Creamos el conjunto de acciones discretas
        # self._actions = make_discrete_action_set_legprototype(env.action_space.shape[0]) # Usamos el conjunto de acciones prototipo específico para Walker2D
        self.action_space = gym.spaces.Discrete(self._actions.shape[0]) # Redefinimos el espacio de acciones a discreto con el número de acciones prototipo

    def action(self, act_idx):
        return self._actions[int(act_idx)] # Convertimos el índice de acción discreta en la acción continua correspondiente
