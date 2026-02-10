import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)