import numpy as np
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    @property
    def total(self):
        return self.tree[0]
    
    def update(self, data_idx: int, priority: float):
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def add(self, priority: float, data):
        data_idx = self.write
        self.data[data_idx] = data
        self.update(data_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return data_idx
    
    def _retrieve(self, idx: int, s: float):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
        
    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, device="cuda", obs_shape = (4, 84, 84), alpha: float = 0.6):
        self.device = device
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.states = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device="cpu")
        self.actions = torch.empty(capacity, dtype=torch.int64, device="cpu")
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device="cpu")
        self.next_states = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device="cpu")
        self.dones = torch.zeros(capacity, dtype=torch.float32, device="cpu")

        self.idx = 0
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def push(self, state, action, reward, next_state, done):
        i = self.idx

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state)
        
        self.states[i].copy_(state.to(dtype=torch.uint8, device="cpu"))
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i].copy_(next_state.to(dtype=torch.uint8, device="cpu"))
        self.dones[i] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        p = self.max_priority ** self.alpha
        if self.size < self.capacity:
            self.tree.add(p, None)
        else:
            self.tree.update(i, p)
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(next_states, np.ndarray):
            next_states = torch.from_numpy(next_states)
        
        batch_size = states.shape[0]
        idxs = (torch.arange(batch_size) + self.idx) % self.capacity
        
        self.states[idxs].copy_(states.to(dtype=torch.uint8, device="cpu"))
        self.next_states[idxs].copy_(next_states.to(dtype=torch.uint8, device="cpu"))
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones

        p = self.max_priority ** self.alpha

        for idx in idxs:
            self.tree.update(int(idx), p)

        self.idx = (self.idx + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)


    def sample(self, batch_size: int, beta: float = 0.4):
        indices = []
        priorities = []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx = self.tree.get(s)
            indices.append(idx)
            leaf_priority = self.tree.tree[idx + self.tree.capacity - 1]
            priorities.append(leaf_priority / self.tree.total)
        
        # Importance-sampling weights:
        # w_i = (1 / N * 1 / P(i))^beta
        weights = (self.size * np.array(priorities)) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        idxs = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        if torch.is_tensor(idxs):
            idxs = idxs.to("cpu")
        else:
            idxs = torch.as_tensor(idxs, dtype=torch.long, device="cpu")
        
        s = self.states[idxs].to(device=self.device, dtype=torch.float32) / 255.0
        a = self.actions[idxs].to(device=self.device)
        r = self.rewards[idxs].to(device=self.device)
        ns = self.next_states[idxs].to(device=self.device, dtype=torch.float32) / 255.0
        d = self.dones[idxs].to(device=self.device)

        return s, a, r, ns, d, weights, idxs
    
    def update_priorities(self, indices, priorities):
        idxs = indices.detach().cpu().numpy().astype(np.int32)
        priorities = priorities.detach().cpu().numpy().astype(np.float32)

        priorities = np.clip(priorities, a_min=1e-6, a_max=None)
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority ** self.alpha)


class NStepAccumulator:
    """
    Mantiene una deque por env_id para acumular transiciones n-step:
    (s0, a0, R_n, s_n, done_n)
    """

    def __init__(self, n: int, gamma: float, n_envs: int):
        self.n = n
        self.gamma = gamma
        self.n_envs = n_envs
        self.buffers = [deque(maxlen=n) for _ in range(n_envs)]

    def reset_env(self, env_id: int):
        self.buffers[env_id].clear()
    
    def _compute_n_step(self, buffer: deque): 
        R_n = 0.0
        for i, (_, _, r, _, d) in enumerate(buffer):
            R_n += (self.gamma ** i) * r
            if d:
                break
        s0, a0, _, _, _ = buffer[0]
        _, _, _, s_n, done_n = buffer[-1]
    
        # Si aparece un done antes de n pasos, se usa ese punto de corte para el bootstrapping.
        last_idx = len(buffer) - 1
        for i, (_, _, _, s, d) in enumerate(buffer):
            if d:
                last_idx = i
                break
        _, _, _, s_n, done_n = buffer[last_idx]
        return s0, a0, R_n, s_n, done_n

    def add(self, env_id: int, state, action, reward, next_state, done):
        buffer = self.buffers[env_id]
        buffer.append((state, action, reward, next_state, done))

        out = []
        if len(buffer) == self.n:
            out.append(self._compute_n_step(buffer))
            buffer.popleft()
        
        if done:
            while len(buffer) > 0:
                out.append(self._compute_n_step(buffer))
                buffer.popleft()
            self.reset_env(env_id)
        
        return out


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

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
        mu_range = 1.0 / np.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.sigma_init / np.sqrt(self.in_features))
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.bias_sigma, self.sigma_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        device = self.weight_mu.device
        x = torch.randn(size, device=device)
        # Factorized Gaussian noise de NoisyNet.
        return x.sign().mul(x.abs().sqrt()) 

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def disable_noise(self):
        self.noise_enabled = False
    
    def enable_noise(self):
        self.noise_enabled = True

    def forward(self, input):
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

        # Dueling head: stream de valor y stream de ventajas.
        self.value_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, 1, sigma_init)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions, sigma_init)
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
    
    def forward(self, x):
        conv_out = self.encoder(x).view(x.size()[0], -1)
        value = self.value_stream(conv_out)
        advantages = self.advantage_stream(conv_out)

        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True)) 
        return q_values


class RainbowDQN(nn.Module):
    def __init__(self, num_actions: int, n_atoms : int = 51, v_min: float = -10.0, v_max: float = 10.0, sigma_init: float = 0.017):
        super().__init__()
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Soporte discreto de C51.
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer('support', support)

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
        
        self.value_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, n_atoms, sigma_init)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(n_flat, 512, sigma_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions * n_atoms, sigma_init)
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
        conv_out = self.encoder(x).view(x.size()[0], -1)
        value = self.value_stream(conv_out)
        advantages = self.advantage_stream(conv_out)

        value = value.unsqueeze(1)
        advantages = advantages.view(-1, self.num_actions, self.n_atoms) 

        logits = value + (advantages - advantages.mean(dim=1, keepdim=True))
        if return_probs:
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            return logits
    
    @torch.no_grad()
    def get_q_values(self, x):
        probs = self.forward(x, return_probs=True)
        q_values = torch.sum(probs * self.support.view(1, 1, -1), dim=-1)
        return q_values


@torch.no_grad()
def projection_distribution(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    n_step: int,
    support: torch.Tensor,
    v_min: float,
    v_max: float
):
    # Acepta (B,1,N) o (B,N).
    if next_dist.dim() == 3:
        if next_dist.size(1) != 1:
            raise ValueError(f"projection_distribution: esperaba A=1, got {next_dist.shape}")
        next_dist = next_dist.squeeze(1)
    elif next_dist.dim() != 2:
        raise ValueError(f"projection_distribution: esperaba (B,N) o (B,1,N), got {next_dist.shape}")

    batch_size, n_atoms = next_dist.shape
    device = next_dist.device

    # Tz = r + gamma^n * (1 - done) * z_i
    Tz = rewards.unsqueeze(1) + (gamma ** n_step) * (1 - dones.unsqueeze(1)) * support.view(1, n_atoms)
    Tz = Tz.clamp(v_min, v_max)

    b = (Tz - v_min) / (v_max - v_min) * (n_atoms - 1)
    l = b.floor().long()
    u = b.ceil().long()

    projected_dist = torch.zeros((batch_size, n_atoms), device=device, dtype=torch.float32)

    offset = (torch.arange(batch_size, device=device) * n_atoms).unsqueeze(1)

    l_idx = (l + offset).view(-1)
    u_idx = (u + offset).view(-1)
    next_dist_flat = next_dist.view(-1)

    # Proyección lineal entre vecinos l y u.
    proj_dist_flat = projected_dist.view(-1)
    proj_dist_flat.scatter_add_(0, l_idx, (next_dist * (u.float() - b)).view(-1))
    proj_dist_flat.scatter_add_(0, u_idx, (next_dist * (b - l.float())).view(-1))

    # Si b cae exacto en un átomo, toda su masa va a ese índice.
    eq_mask = (u == l).view(-1)
    if eq_mask.any():
        idx = l_idx[eq_mask]
        proj_dist_flat.scatter_add_(0, idx, next_dist_flat[eq_mask])
    
    projected_dist = projected_dist.view(batch_size, n_atoms)
    projected_dist = projected_dist / projected_dist.sum(dim=1, keepdim=True)
    return projected_dist
