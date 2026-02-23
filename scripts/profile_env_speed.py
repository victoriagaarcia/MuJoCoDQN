"""
Script para perfilar la velocidad de los entornos con AsyncVectorEnv vs SyncVectorEnv
y medir el overhead de comunicación entre procesos.
"""
import gymnasium as gym
import numpy as np
import torch
import time
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from MuJoCoDQN.src.dqn_copy import QNetwork
from MuJoCoDQN.src.envs_copy import DiscreteActionWrapper, PixelStackWrapper, ForwardAliveSmoothReward, IgnoreAngleTerminationWrapper

ENV_ID = "Walker2d-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_RW, BETA_RW, GAMMA_RW, DELTA_RW, LAM_RW = 1.5, 1.0, 0.8, 1.0, 0.05
SEED = 42

def make_env(rank: int):
    def _thunk():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = ForwardAliveSmoothReward(env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
        env = IgnoreAngleTerminationWrapper(env)
        env = DiscreteActionWrapper(env)
        env = PixelStackWrapper(env)
        return env
    return _thunk

def benchmark_envs(num_envs, num_steps=1000, env_type="async"):
    """Mide el tiempo de ejecución de env.step() con diferentes configuraciones."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {env_type.upper()} con {num_envs} entornos")
    print(f"{'='*60}")
    
    # Crear entornos
    if env_type == "async":
        env = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    elif env_type == "sync":
        env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
    else:
        raise ValueError("env_type debe ser 'async' o 'sync'")
    
    q_net = QNetwork(env.single_action_space.n).to(DEVICE)
    q_net.eval()
    
    seeds = [SEED + i for i in range(num_envs)]
    state, _ = env.reset(seed=seeds)
    
    # Calentar GPU
    for _ in range(10):
        with torch.no_grad():
            _ = q_net(torch.randn(num_envs, 4, 84, 84).to(DEVICE))
    
    # Profiling
    times = {
        "inference": 0.0,
        "env_step": 0.0,
        "action_selection": 0.0,
        "other": 0.0,
    }
    
    start_total = time.perf_counter()
    
    for step in range(num_steps):
        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            actions = q_net(s).argmax(dim=1).cpu().numpy()
        t1 = time.perf_counter()
        times["inference"] += (t1 - t0)
        
        # Environment step
        t0 = time.perf_counter()
        next_state, reward, terminated, truncated, _ = env.step(actions)
        t1 = time.perf_counter()
        times["env_step"] += (t1 - t0)
        
        done = np.logical_or(terminated, truncated)
        if done.any():
            state, _ = env.reset(seed=seeds)
        else:
            state = next_state
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    print(f"Total steps: {num_steps}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per step: {(total_time/num_steps)*1000:.2f}ms")
    print(f"Steps per second: {num_steps/total_time:.1f}")
    print(f"\nBreakdown:")
    print(f"  Inference:       {times['inference']:.2f}s ({times['inference']/total_time*100:.1f}%)")
    print(f"  Env step:        {times['env_step']:.2f}s ({times['env_step']/total_time*100:.1f}%)")
    
    env.close()
    
    return total_time / num_steps  # tiempo promedio por step

# Benchmark con diferentes números de entornos
print("\n" + "="*60)
print("COMPARACIÓN: AsyncVectorEnv vs SyncVectorEnv")
print("="*60)

num_steps = 500  # Reducir para que sea rápido

for num_envs in [1, 2, 4, 8]:
    print(f"\n{'-'*60}")
    print(f"NUM_ENVS = {num_envs}")
    print(f"{'-'*60}")
    
    time_sync = benchmark_envs(num_envs, num_steps=num_steps, env_type="sync")
    time_async = benchmark_envs(num_envs, num_steps=num_steps, env_type="async")
    
    speedup = time_sync / time_async
    print(f"\nSpeedup (Sync/Async): {speedup:.2f}x")
    if speedup < 1:
        print("⚠️  AsyncVectorEnv es MÁS LENTO")
    else:
        print("✓ AsyncVectorEnv es más rápido")
