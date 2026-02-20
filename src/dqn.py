import numpy as np
import torch
import torch.nn as nn

# QNetwork (CNN+MLP)
class QNetwork(nn.Module): # Aproxima Q(s,a), es decir, el valor esperado si hago la acción a en el estado s
    def __init__(self, num_actions): 
        super(QNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # 4 es el número de frames apilados
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
    
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)  # Assuming input shape is (4, 84, 84)
            n_flat = self.encoder(dummy).shape[1]
            
    
        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions) # Cada componente es Q(s,a_i) para cada acción a_i del espacio de acciones discretas
        )

    def forward(self, x):
        conv_out = self.encoder(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer: # Memoria en la que guardamos transiciones (s,a,r,s',done) para luego muestrear aleatoriamente y romper la correlación temporal entre muestras
    def __init__(self, capacity, obs_shape=(4,84,84), device="cpu"):
        self.capacity = int(capacity)
        self.device = device
        
        # CPU pinned
        self.states      = torch.empty((capacity, *obs_shape), dtype=torch.uint8,  pin_memory=True)
        self.next_states = torch.empty((capacity, *obs_shape), dtype=torch.uint8,  pin_memory=True)
        self.actions     = torch.empty((capacity,),            dtype=torch.int64,  pin_memory=True)
        self.rewards     = torch.empty((capacity,),            dtype=torch.float32,pin_memory=True)
        self.dones       = torch.empty((capacity,),            dtype=torch.float32,pin_memory=True)

        self.idx = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        # state/next_state: torch uint8 (4,84,84) o np.uint8 compatible

        i = self.idx
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state)

        self.states[i].copy_(state.to(dtype=torch.uint8, device="cpu"))
        self.next_states[i].copy_(next_state.to(dtype=torch.uint8, device="cpu"))

        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.dones[i]   = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        states/next_states: torch uint8 (B,4,84,84) en CPU (ideal) o numpy.
        actions: np.int64 o torch.int64 (B,)
        rewards: np.float32 o torch.float32 (B,)
        dones:   np.bool/float o torch (B,)
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(next_states, np.ndarray):
            next_states = torch.from_numpy(next_states)

        b = states.shape[0]
        idxs = (self.idx + torch.arange(b)) % self.capacity

        # aseguramos CPU + uint8
        self.states[idxs].copy_(states.to(dtype=torch.uint8, device="cpu"))
        self.next_states[idxs].copy_(next_states.to(dtype=torch.uint8, device="cpu"))

        self.actions[idxs] = torch.as_tensor(actions, dtype=torch.int64)
        self.rewards[idxs] = torch.as_tensor(rewards, dtype=torch.float32)
        # dones lo guardo float32 para target (1-done)
        self.dones[idxs]   = torch.as_tensor(dones, dtype=torch.float32)

        self.idx = int((self.idx + b) % self.capacity)
        self.size = int(min(self.size + b, self.capacity))

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device="cpu")

        # Copia no-bloqueante a GPU + normalización a [0,1]
        s  = self.states[idxs].to(self.device, non_blocking=True).float().div_(255.0)
        ns = self.next_states[idxs].to(self.device, non_blocking=True).float().div_(255.0)
        a  = self.actions[idxs].to(self.device, non_blocking=True)
        r  = self.rewards[idxs].to(self.device, non_blocking=True)
        d  = self.dones[idxs].to(self.device, non_blocking=True)
        return s, a, r, ns, d

    def __len__(self):
        return self.size