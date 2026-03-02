import numpy as np
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, capacity: int, device="cuda", obs_shape = (4, 84, 84), alpha: float = 0.6):
        self.device = device
        self.capacity = capacity
        self.alpha = alpha # Exponente para controlar la prioridad (0 = uniforme, 1 = prioridad total)
        self.tree = SumTree(capacity) # Árbol para gestionar prioridades
        self.max_priority = 1.0 # Prioridad máxima inicial (para nuevas experiencias)

        self.states = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device="cpu") # Buffer para estados
        self.actions = torch.empty(capacity, dtype=torch.int64, device="cpu") # Buffer para acciones
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device="cpu") # Buffer para recompensas
        self.next_states = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device="cpu") # Buffer para próximos estados
        self.dones = torch.zeros(capacity, dtype=torch.float32, device="cpu") # Buffer para dones (finalización de episodios)

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
        # for i in range(batch_size):
        #     self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # En vez de pushear uno a uno, insertamos el batch entero
        idxs = (torch.arange(batch_size) + self.idx) % self.capacity # Índices donde se escribirán las nuevas experiencias
        
        self.states[idxs].copy_(states.to(dtype=torch.uint8, device="cpu")) # Almacena los estados (convertidos a uint8 para ahorrar memoria)
        self.next_states[idxs].copy_(next_states.to(dtype=torch.uint8, device="cpu")) # Almacena los próximos estados (convertidos a uint8 para ahorrar memoria)
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones

        self.idx = (self.idx + batch_size) % self.capacity # Mueve el índice de escritura (circular)
        self.size = min(self.size + batch_size, self.capacity) # Actualiza el tamaño del buffer

        # Prioridad inicial = máxima prioridad
        p = self.max_priority ** self.alpha # Calcula la prioridad ajustada por alpha
        if self.size < self.capacity:
            for i in range(batch_size):
                self.tree.add(p, None) # Agrega las nuevas experiencias al árbol con su prioridad
        else:
            # Si estamos sobrescribiendo experiencias existentes, actualizamos sus prioridades en el árbol
            for i in range(batch_size):
                idx = (self.idx - batch_size + i) % self.capacity
                self.tree.update(idx, p) # Actualiza la prioridad de las experiencias sobrescritas en el árbol


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
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device) # Convierte los pesos a tensor de PyTorch
        idxs = torch.tensor(indices, dtype=torch.long, device=self.device) # Convierte los índices a tensor de PyTorch
        
        # idxs debe estar en CPU para indexar tensores CPU
        if torch.is_tensor(idxs):
            idxs = idxs.to("cpu")
        else:
            idxs = torch.as_tensor(idxs, dtype=torch.long, device="cpu")
        
        s = self.states[idxs].to(device=self.device, dtype=torch.float32) / 255.0 # Normaliza los estados (de uint8 a float32 en [0,1])
        a = self.actions[idxs].to(device=self.device) # Acciones muestreadas
        r = self.rewards[idxs].to(device=self.device) # Recompensas muestreadas
        ns = self.next_states[idxs].to(device=self.device, dtype=torch.float32) / 255.0 # Normaliza los próximos estados (de uint8 a float32 en [0,1])
        d = self.dones[idxs].to(device=self.device) # Dones muestreados

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
        last_idx = len(buffer) - 1 # Si no hay 'done', nos quedamos con el último 
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
        
        return out # Devuelve la lista de transiciones n-step acumuladas


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init # Copilot pone 0.017 por defecto

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.noise_enabled = True 

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Inicialización de los parámetros mu y sigma
        mu_range = 1.0 / np.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.sigma_init / np.sqrt(self.in_features))
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.bias_sigma, self.sigma_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        device = self.weight_mu.device
        x = torch.randn(size, device=device) # Genera ruido gaussiano
        # Multiplicamos el ruido por la raíz de su valor absoluto para obtener una distribución de ruido 
        # que favorece valores pequeños pero permite valores grandes ocasionalmente (heavy-tailed)
        return x.sign().mul(x.abs().sqrt()) 

    def reset_noise(self):
        # Genera nuevos valores de ruido para pesos y sesgos
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in)) # Genera la matriz de ruido para los pesos (outer product)
        self.bias_epsilon.copy_(epsilon_out) # Genera el vector de ruido para los sesgos (solo depende de las salidas)

    def disable_noise(self):
        self.noise_enabled = False
    
    def enable_noise(self):
        self.noise_enabled = True

    def forward(self, input):
        # Si el ruido está habilitado, calculamos los pesos y sesgos ruidosos; 
        # si no, usamos los pesos y sesgos deterministas
        if self.training and self.noise_enabled: 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)


class NoisyandDuelingDQN(nn.Module):
    def __init__(self, num_actions: int, sigma_init: float = 0.017):
        super().__init__()
        self.num_actions = num_actions

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84) 
            n_flat = self.encoder(dummy).shape[1]

        # Dividimos la red en dos "streams": uno para el valor del estado y otro para las ventajas de cada acción
        self.value_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, 1, sigma_init) # Salida: valor del estado V(s)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions, sigma_init) # Salida: ventajas de cada acción A(s,a)
        )
    
    def reset_noise(self):
        # Resetea el ruido en todas las capas NoisyLinear
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def enable_noise(self): 
        # Activa el ruido en todas las capas NoisyLinear
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.enable_noise()
    
    def disable_noise(self):
        # Desactiva el ruido en todas las capas NoisyLinear 
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.disable_noise()
    
    def forward(self, x):
        conv_out = self.encoder(x).view(x.size()[0], -1) # Extrae características con la parte convolucional
        value = self.value_stream(conv_out) # Calcula el valor del estado V(s)
        advantages = self.advantage_stream(conv_out) # Calcula las ventajas de cada acción A(s,a)

        # Combina el valor y las ventajas para obtener Q(s,a) usando la fórmula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True)) 
        return q_values # Devuelve los valores Q para cada acción


class RainbowDQN(nn.Module):
    def __init__(self, num_actions: int, n_atoms : int = 51, v_min: float = -10.0, v_max: float = 10.0, sigma_init: float = 0.017):
        super().__init__()
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Soporte de la distribución de valores
        support = torch.linspace(v_min, v_max, n_atoms) # dim: (n_atoms,) con los valores z_i del soporte
        self.register_buffer('support', support) # Registramos el soporte como buffer para que se

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84) 
            n_flat = self.encoder(dummy).shape[1]
        
        # Value stream para la distribución de valores del estado
        self.value_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, n_atoms, sigma_init) # Salida: distribución de valores del estado (n_atoms)
        )

        # Advantage stream para la distribución de ventajas de cada acción
        self.advantage_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions * n_atoms, sigma_init) # Salida: distribución de ventajas de cada acción (num_actions * n_atoms)
        )

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def enable_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.enable_noise()

    def disable_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.disable_noise()
    
    def forward(self, x, return_probs : bool = True):
        # Extrae características con la parte convolucional
        conv_out = self.encoder(x).view(x.size()[0], -1) # 
        # Calcula la distribución de valores del estado 
        value = self.value_stream(conv_out) # (batch_size, n_atoms)
        # Calcula la distribución de ventajas de cada acción 
        advantages = self.advantage_stream(conv_out) # (batch_size, num_actions * n_atoms)

        # Reshape para broadcasting (batch_size, 1, n_atoms)
        value = value.unsqueeze(1)
        # Reshape para separar acciones (batch_size, num_actions, n_atoms)
        advantages = advantages.view(-1, self.num_actions, self.n_atoms) 

        logits = value + (advantages - advantages.mean(dim=1, keepdim=True)) # Combina valor y ventajas para obtener la distribución de Q(s,a)
        if return_probs:
            probs = F.softmax(logits, dim=-1) # Convierte los logits en probabilidades usando softmax
            return probs # Devuelve las distribuciones de probabilidad para cada acción (batch_size, num_actions, n_atoms)
        else:
            return logits # Devuelve los logits sin normalizar (útil para el cálculo de la pérdida con CrossEntropyLoss)
    
    @torch.no_grad()
    def get_q_values(self, x):
        # Calcula los valores Q esperados para cada acción a partir de las distribuciones de probabilidad
        probs = self.forward(x, return_probs=True) # Obtiene las distribuciones de probabilidad para cada acción
        q_values = torch.sum(probs * self.support.view(1, 1, -1), dim=-1) # Calcula el valor esperado Q(s,a) = sum(p_i * z_i)
        return q_values # Devuelve los valores Q para cada acción


@torch.no_grad()
def projection_distribution(
    next_dist: torch.Tensor, # (batch_size, num_actions, n_atoms) distribución de probabilidad de los próximos estados para cada acción
    rewards: torch.Tensor, # (batch_size,) recompensas obtenidas al tomar la acción
    dones: torch.Tensor, # (batch_size,) indicadores de si el episodio terminó después de tomar la acción
    gamma: float,
    n_step: int,
    support: torch.Tensor,
    v_min: float,
    v_max: float
):
    # batch_size, n_actions, n_atoms = next_dist.shape
    # Device = next_dist.device
    
    # Acepta (B,1,N) y lo convierte a (B,N)
    if next_dist.dim() == 3:
        if next_dist.size(1) != 1:
            raise ValueError(f"projection_distribution: esperaba A=1, got {next_dist.shape}")
        next_dist = next_dist.squeeze(1)  # (B,N)
    elif next_dist.dim() != 2:
        raise ValueError(f"projection_distribution: esperaba (B,N) o (B,1,N), got {next_dist.shape}")

    batch_size, n_atoms = next_dist.shape
    device = next_dist.device

    # Calcula el soporte proyectado Tz para cada transición del batch
    # Tz = r + (gamma^n) * (1 - done) * z_i
    Tz = rewards.unsqueeze(1) + (gamma ** n_step) * (1 - dones.unsqueeze(1)) * support.view(1, n_atoms) # ()
    Tz = Tz.clamp(v_min, v_max) # Limita el soporte proyectado al rango [v_min, v_max]

    b = (Tz - v_min) / (v_max - v_min) * (n_atoms - 1) # (batch_size, n_atoms) Índices flotantes del soporte proyectado en el espacio de átomos
    l = b.floor().long() # (batch_size, n_atoms) Índices inferiores
    u = b.ceil().long() # (batch_size, n_atoms) Índices superiores

    # Distribución proyectada inicializada a cero
    projected_dist = torch.zeros((batch_size, n_atoms), device=device, dtype=torch.float32)

    # Distribución entre l y u (vectorizado con scatter_add)
    offset = (torch.arange(batch_size, device=device) * n_atoms).unsqueeze(1) # (batch_size, 1) Offset para indexar en la distribución proyectada

    # Índices planos para scatter_add
    l_idx = (l + offset).view(-1) # Índices planos para l
    u_idx = (u + offset).view(-1) # Índices planos para u
    next_dist_flat = next_dist.view(-1) # Aplanamos la distribución de los próximos estados para indexar fácilmente

    # Agregamos la probabilidad de cada átomo a los índices l y u correspondientes
    proj_dist_flat = projected_dist.view(-1) # Aplanamos la distribución proyectada para scatter_add
    # proj_dist_flat.scatter_add_(0, l_idx, (next_dist_flat * (u.float() - b)).view(-1)) 
    # proj_dist_flat.scatter_add_(0, u_idx, (next_dist_flat * (b - l.float())).view(-1))
    proj_dist_flat.scatter_add_(0, l_idx, (next_dist * (u.float() - b)).view(-1))
    proj_dist_flat.scatter_add_(0, u_idx, (next_dist * (b - l.float())).view(-1))

    # Corregimos el caso l==u (cuando b es un entero, toda la probabilidad va a un solo átomo)
    eq_mask = (u == l).view(-1) # Máscara para identificar dónde l y u son iguales
    if eq_mask.any():
        idx = l_idx[eq_mask] # Índices donde l == u
        # En este caso, toda la probabilidad va al mismo átomo, así que sumamos ambas contribuciones
        proj_dist_flat.scatter_add_(0, idx, next_dist_flat[eq_mask])
    
    projected_dist = projected_dist.view(batch_size, n_atoms) # Reshape de vuelta a (batch_size, n_atoms)
    # Normalizamos por si acaso
    projected_dist = projected_dist / projected_dist.sum(dim=1, keepdim=True) # Normaliza la distribución proyectada para que sume 1
    return projected_dist # Devuelve la distribución proyectada para cada transición del batch
