import numpy as np
import torch
from collections import deque

# Prioritized Experience Replay: SumTree implementation
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity # Número de hojas (experiencias)
        self.tree = np.zeros(2 * capacity - 1) # Árbol binario completo
        self.data = np.zeros(capacity, dtype=object) # Almacena experiencias
        self.write = 0 # Índice para escribir la próxima experiencia
        self.n_entries = 0 # Número de experiencias almacenadas
    
    def _propagate(self, idx: int, change: float):
        # Propaga el cambio hacia arriba en el árbol
        parent = (idx - 1) // 2
        # Actualiza el nodo padre con el cambio
        self.tree[parent] += change
        if parent != 0:
            # Si no hemos llegado a la raíz, seguimos propagando (recursividad)
            self._propagate(parent, change)
    
    @property
    def total(self):
        # El valor total de prioridades es el valor de la raíz del árbol
        return self.tree[0]
    
    def update(self, data_idx: int, priority: float):
        # Actualiza la prioridad de una experiencia dada su posición en el buffer
        # data_idx es el índice de la experiencia en el buffer (0 a capacity-1)
        tree_idx = data_idx + self.capacity - 1 # Índice en el árbol correspondiente a la experiencia
        change = priority - self.tree[tree_idx] # Calcula el cambio en prioridad
        self.tree[tree_idx] = priority # Actualiza la prioridad en el nodo hoja
        self._propagate(tree_idx, change) # Propaga el cambio hacia arriba
    
    def add(self, priority: float, data): # DATA?
        # Agrega una nueva experiencia con su prioridad
        data_idx = self.write # Índice donde se escribirá la nueva experiencia
        self.data[data_idx] = data # Almacena la experiencia en el buffer ¡!
        self.update(data_idx, priority) # Actualiza el árbol con la nueva prioridad
        self.write = (self.write + 1) % self.capacity # Mueve el índice de escritura (circular)
        self.n_entries = min(self.n_entries + 1, self.capacity) # Actualiza el conteo de experiencias
        return data_idx # Devuelve el índice de la experiencia agregada (útil para actualizaciones futuras)
    
    def _retrieve(self, idx: int, s: float):
        # Busca en el árbol la experiencia correspondiente a la prioridad acumulada s
        left = 2 * idx + 1 # Índice del hijo izquierdo
        right = left + 1 # Índice del hijo derecho
        
        if left >= len(self.tree): # Si llegamos a una hoja (no hay hijos)
            return idx # Devolvemos el índice de la hoja
        
        if s <= self.tree[left]: # Si s es menor o igual a la prioridad acumulada del hijo izquierdo
            return self._retrieve(left, s) # Buscamos en el hijo izquierdo
        else: # Si s es mayor, buscamos en el hijo derecho, ajustando s restando la prioridad del hijo izquierdo
            return self._retrieve(right, s - self.tree[left]) # Buscamos en el hijo derecho, ajustando s
        
    def get(self, s: float):
        # Devuelve la experiencia correspondiente a la prioridad acumulada s
        idx = self._retrieve(0, s) # Comienza la búsqueda desde la raíz
        data_idx = idx - self.capacity + 1 # Calcula el índice de la experiencia en el buffer
        return data_idx
        # return (idx, self.tree[idx], self.data[data_idx]) # Devuelve el índice en el árbol, la prioridad y la experiencia
        # ¿QUÉ TIENE QUE DEVOLVER ESTO?


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_shape = (4, 84, 84), alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha # Exponente para controlar la prioridad (0 = uniforme, 1 = prioridad total)
        self.tree = SumTree(capacity) # Árbol para gestionar prioridades
        self.max_priority = 1.0 # Prioridad máxima inicial (para nuevas experiencias)

        self.states = np.zeros((capacity, *obs_shape), dtype=torch.uint8, device="cpu") # Buffer para estados
        self.actions = np.zeros(capacity, dtype=np.int64, device="cpu") # Buffer para acciones
        self.rewards = np.zeros(capacity, dtype=np.float32, device="cpu") # Buffer para recompensas
        self.next_states = np.zeros((capacity, *obs_shape), dtype=torch.uint8, device="cpu") # Buffer para próximos estados
        self.dones = np.zeros(capacity, dtype=np.float32, device="cpu") # Buffer para dones (finalización de episodios)

        self.idx = 0 # Índice para escribir la próxima experiencia
        self.size = 0 # Número de experiencias almacenadas
    
    def __len__(self):
        return self.size
    
    # COMPARAR FLUJO CON EL BUFFER DE LA DQN!
    def push(self, state, action, reward, next_state, done):
        # Agrega una nueva experiencia al buffer con su prioridad
        i = self.idx # Índice donde se escribirá la nueva experiencia

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state)
        
        self.states[i].copy_(state.to(dtype=torch.uint8, device="cpu")) # Almacena el estado (convertido a uint8 para ahorrar memoria)
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i].copy_(next_state.to(dtype=torch.uint8, device="cpu")) # Almacena el próximo estado (convertido a uint8 para ahorrar memoria)
        self.dones[i] = done

        self.idx = (self.idx + 1) % self.capacity # Mueve el índice de escritura (circular)
        self.size = min(self.size + 1, self.capacity) # Actualiza el tamaño del buffer

        # Prioridad inicial = máxima prioridad
        p = self.max_priority ** self.alpha # Calcula la prioridad ajustada por alpha
        if self.size < self.capacity:
            self.tree.add(p, None) # Agrega la nueva experiencia al árbol con su prioridad
        else:
            # Si estamos sobrescribiendo una experiencia existente, actualizamos su prioridad en el árbol
            self.tree.update(i, p) # Actualiza la prioridad de la experiencia sobrescrita en el árbol
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(next_states, np.ndarray):
            next_states = torch.from_numpy(next_states)
        
        batch_size = states.shape[0]
        for i in range(batch_size):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
    def sample(self, batch_size: int, beta: float = 0.4):
        # Muestra un batch de experiencias con prioridad
        indices = []
        priorities = []
        segment = self.tree.total / batch_size # Divide el rango total de prioridades en segmentos iguales
        for i in range(batch_size):
            a = segment * i # Límite inferior del segmento
            b = segment * (i + 1) # Límite superior del segmento
            s = np.random.uniform(a, b) # Muestra un valor aleatorio dentro del segmento
            idx = self.tree.get(s) # Obtiene el índice de la experiencia correspondiente a la prioridad acumulada s
            indices.append(idx) # Almacena el índice de la experiencia muestreada
            leaf_priority = self.tree.tree[idx + self.tree.capacity - 1] # Obtiene la prioridad de la experiencia muestreada
            priorities.append(leaf_priority / self.tree.total) # Almacena la prioridad normalizada de la experiencia muestreada
        
        # Importance sampling weights: 
        # w_i = (1 / N * 1 / P(i))^beta, donde P(i) es la probabilidad de muestreo de la experiencia i
        weights = (self.size * np.array(priorities)) ** (-beta) # Calcula los pesos de importancia para corregir el sesgo de muestreo
        weights /= weights.max() # Normaliza los pesos para evitar valores muy grandes
        weights = torch.tensor(weights, dtype=torch.float32, device="cpu") # Convierte los pesos a tensor de PyTorch
        idxs = torch.tensor(indices, dtype=torch.long, device="cpu") # Convierte los índices a tensor de PyTorch

        s = self.states[idxs].to(device="cpu", dtype=torch.float32) / 255.0 # Normaliza los estados (de uint8 a float32 en [0,1])
        a = self.actions[idxs].to(device="cpu") # Acciones muestreadas
        r = self.rewards[idxs].to(device="cpu") # Recompensas muestreadas
        ns = self.next_states[idxs].to(device="cpu", dtype=torch.float32) / 255.0 # Normaliza los próximos estados (de uint8 a float32 en [0,1])
        d = self.dones[idxs].to(device="cpu") # Dones muestreados

        return s, a, r, ns, d, weights, idxs # Devuelve el batch de experiencias muestreadas junto con sus pesos de importancia y sus índices en el buffer
    
    def update_priorities(self, indices, priorities):
        idxs = indices.detach().cpu().numpy().astype(np.int32)
        priorities = priorities.detach().cpu().numpy().astype(np.float32)

        priorities = np.clip(priorities, a_min=1e-6, a_max=None) # Evita prioridades demasiado pequeñas
        self.max_priority = max(self.max_priority, priorities.max()) # Actualiza la prioridad máxima si es necesario

        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority ** self.alpha) # Actualiza la prioridad de la experiencia en el árbol


class NStepAccumulator:
    """
    Mantiene una deque por env_id para acumular transiciones n-step:
    (s0, a0, R_n, s_n, done_n)
    """

    def __init__(self, n: int, gamma: float, n_envs: int):
        self.n = n
        self.gamma = gamma
        self.n_envs = n_envs
        self.buffers = [deque(maxlen=n) for _ in range(n_envs)] # Una deque por cada entorno para acumular transiciones

    def reset_env(self, env_id: int):
        self.buffers[env_id].clear() # Limpia la deque del entorno específico al resetearlo
    
    def _compute_n_step(self, buffer: deque):
        R_n = 0.0
        for i, (_, _, r, _, d) in enumerate(buffer):
            R_n += (self.gamma ** i) * r # Calcula la recompensa acumulada con descuento
            if d: # Si encontramos un done, dejamos de acumular
                break
        s0, a0, _, _, _ = buffer[0] # Estado y acción iniciales
        _, _, _, s_n, done_n = buffer[-1] # Estado y done finales
    
        # Bootstrapping: devolvemos next step del último elemento considerado
        # Si el episodio terminó antes de n pasos, devolvemos el último estado y done=True
        last_idx = 0
        for i, (_, _, _, s, d) in enumerate(buffer):
            if d:
                last_idx = i
                break
        _, _, _, s_n, done_n = buffer[last_idx] # Estado y done del último paso considerado
        return s0, a0, R_n, s_n, done_n # Devuelve la transición n-step acumulada

    def add(self, env_id: int, state, action, reward, next_state, done):
        buffer = self.buffers[env_id] # Obtiene la deque correspondiente al entorno
        buffer.append((state, action, reward, next_state, done)) # Agrega la nueva transición a la deque

        out = []
        if len(buffer) == self.n: # Si hemos acumulado n transiciones
            out.append(self._compute_n_step(buffer)) # Añade la transición n-step acumulada a la lista de salidas
            buffer.popleft() # Elimina la transición más antigua para hacer espacio para nuevas transiciones
        
        if done: # Si la transición actual es un done, vaciamos la deque y procesamos las transiciones restantes
            while len(buffer) > 0:
                out.append(self._compute_n_step(buffer)) # Añade la transición n-step acumulada a la lista de salidas
                buffer.popleft() # Elimina la transición más antigua para procesar la siguiente
            self.reset_env(env_id) # Limpia la deque del entorno al finalizar el episodio
        
        return out # Devuelve la lista de transiciones n-step acumuladas (puede contener