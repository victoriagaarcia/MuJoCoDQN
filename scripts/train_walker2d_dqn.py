import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv

from src.dqn import QNetwork, ReplayBuffer
from src.envs import (
    DiscreteActionWrapper, 
    ForwardAliveSmoothReward, 
    IgnoreAngleTerminationWrapper,
    RGBObsWrapper
)
from .utils import (
    epsilon,
    preprocess_rgb_batch_torch,
    should_update_target,
    save_experiment_to_excel
)

# -----------------------------
# Hiperparámetros
# -----------------------------

ENV_ID = "Walker2d-v5"

TOTAL_STEPS = 8_000_000 # Número total de pasos de interacción con el entorno (no episodios)
BUFFER_SIZE = 200_000 # Capacidad máxima del replay buffer (número de transiciones almacenadas)
BATCH_SIZE = 64 # Tamaño del batch para el entrenamiento de la red Q
GAMMA = 0.99 # Ponderación del valor futuro en la actualización de Q (factor de descuento)
LR = 1e-4
TARGET_UPDATE = 40_000 # Frecuencia de actualización de la red objetivo (en pasos de interacción)
START_TRAINING = 50_000 # Número de pasos de interacción antes de empezar a entrenar (para llenar el buffer con experiencias iniciales)

EPS_START = 1.0 # Valor inicial de epsilon para la política epsilon-greedy (probabilidad de acción aleatoria)
# EPS_START = 0.1
EPS_END = 0.1 # Valor final de epsilon después de la fase de decaimiento (probabilidad mínima de acción aleatoria)
EPS_DECAY = 4_000_000 # Número de pasos durante los cuales epsilon decae linealmente desde EPS_START hasta EPS_END
START_DECAY = 50_000 # Número de pasos antes de empezar a decaer epsilon 

SEED = 42 # Semilla para reproducibilidad
NUM_ENVS = 4 # Número de entornos paralelos para entrenamiento 

# Configuracion de reward
ALPHA_RW = 1.5
BETA_RW = 1.0
GAMMA_RW = 0.8
DELTA_RW = 1.0
LAM_RW = 0.05

# Train con saltos
TRAIN_FREQ = 4 # Como hay 4 envs, poner 4 es como hacer 1 update por iteración
LOG_EVERY = 5_000
CHECKPOINT_EVERY = 250_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "runs/" + datetime.now().strftime("%b%d_%H_%M_%S") # Directorio para guardar el modelo entrenado y los logs de TensorBoard
EXPERIMENT_XLSX = "runs/experiments.xlsx" # Archivo Excel para guardar los resultados de los experimentos

# MODEL_DIR = f"runs/Feb14_20_38_13" # Directorio para guardar el modelo entrenado y los logs de TensorBoard (ajusta esto)
# MODEL_DATE = "Feb14_20_38_13"
# MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d.pt"  # ← ajusta esto
# MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d_step3000000.pt"  # ← ajusta esto

def make_env(rank:int):
    def _thunk():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        # env = ForwardAliveSmoothReward(env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
        env = IgnoreAngleTerminationWrapper(env)
        env = DiscreteActionWrapper(env)
        env = RGBObsWrapper(env)
        return env
    return _thunk

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    writer = SummaryWriter(MODEL_DIR) # Creamos un escritor de TensorBoard para registrar métricas durante el entrenamiento

    env = AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)]) # Creamos un entorno vectorizado con múltiples instancias en paralelo para acelerar el entrenamiento
    n_actions = env.single_action_space.n

    # Creamos la red Q (online: para seleccionar acciones) y la red objetivo (target: para calcular los objetivos de entrenamiento)
    q_net = QNetwork(n_actions).to(DEVICE)
    # q_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    target_net = QNetwork(n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict()) # Inicializamos la red objetivo con los mismos pesos que la red online

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE, obs_shape=(4,84,84), device=DEVICE) 

    seeds = [SEED + i for i in range(NUM_ENVS)] # Semillas diferentes para cada entorno paralelo para mayor diversidad de experiencias
    # state, _ = env.reset(seed=seeds) # Reiniciamos el entorno y obtenemos el estado inicial (stack de frames)
    # episode_reward = 0.0
    # n_episodes = 0

    obs, info = env.reset(seed=seeds) # Reiniciamos el entorno y obtenemos el estado inicial (batch de stacks de frames)
    
    # Frame inicial -> stack 4
    frame = preprocess_rgb_batch_torch(obs, out_size=84, device="cpu") # (B,1,84,84) uint8
    state = frame.repeat(1, 4, 1, 1).contiguous()
    
    # obs_chw = transpose_obs_batch(obs) # Transponemos las observaciones al formato (B, C, H, W) 
    # state = np.repeat(obs_chw, 4, axis=1) # Creamos el stack inicial de 4 frames repitiendo la misma observación 4 veces

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    n_episodes = 0
    updates_done = 0  # Contador de updates
    avg_test_reward = np.nan  # para que exista incluso si no llegas a guardar checkpoint

    # Contador transiciones reales
    global_step = 0

    pbar = tqdm(total=TOTAL_STEPS, desc="train_steps")
    while global_step < TOTAL_STEPS:
    # for it in tqdm(range(total_iters)):
        eps = epsilon(global_step, EPS_END, EPS_START, 
                      START_DECAY, EPS_DECAY) # Calculamos el valor de epsilon para esta etapa del entrenamiento (decay lineal)

        # epsilon-greedy (batch)
        if np.random.rand() < eps:
            actions = np.array([env.single_action_space.sample() for _ in range(NUM_ENVS)], dtype=np.int64)
        else:
            with torch.no_grad():
                s = state.to(DEVICE, non_blocking=True).float().div_(255.0)  # (B,4,84,84) float
                q = q_net(s)
                actions = q.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

        # Ejecutamos la acción en el entorno vectorizado
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        done = np.logical_or(terminated, truncated)

        episode_rewards+= rewards.astype(np.float32)
        episode_lengths += 1
        
        # preprocess next frames (batch)
        next_frame = preprocess_rgb_batch_torch(next_obs, out_size=84, device="cpu") # (B,1,84,84) uint8
        next_state = torch.cat([state[:, 1:], next_frame], dim=1).contiguous()        # (B,4,84,84)

        if isinstance(info, dict) and "final_observation" in infos:
            final_obs = infos["final_observation"]
            final_mask = infos.get("_final_observation", done) 
            idx = np.where(final_mask & done)[0]
            if idx.size > 0:
                term_frame = preprocess_rgb_batch_torch(final_obs[idx], out_size=84, device="cpu")
                next_state[idx] = torch.cat([state[idx, 1:], term_frame], dim=1).contiguous()

        # Ahora pusheamos todo el batch
        buffer.push_batch(
            states=state,
            actions=actions,
            rewards=rewards,
            next_states=next_state,
            dones=done
        )
        
        state = next_state # Actualizamos el estado actual al siguiente estado para la próxima iteración
        
        # episode_rewards += reward.astype(np.float32) # Acumulamos la recompensa del episodio actual para cada entorno

        if np.any(done):
            done_idx = np.where(done)[0]
            for i in done_idx:
                writer.add_scalar("episode_reward", episode_rewards[i], global_step)
                writer.add_scalar("episode_length", episode_lengths[i], global_step)
                n_episodes += 1
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0
                
        if global_step >= START_TRAINING and (global_step % TRAIN_FREQ) < NUM_ENVS:
            states, actions_t, reward_t, next_states, dones_t = buffer.sample(BATCH_SIZE)
            # buffer.sample ya devuelve tensores en DEVICE (según tu implementación)

            q_values = q_net(states).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # DQN estándar
                max_next_q = target_net(next_states).max(1)[0]
                target = reward_t + GAMMA * max_next_q * (1.0 - dones_t)

            loss = torch.nn.functional.mse_loss(q_values, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()
            updates_done += 1

            if global_step % LOG_EVERY < NUM_ENVS:
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("epsilon", eps, global_step)
                writer.add_scalar("updates_done", updates_done, global_step)

        if should_update_target(global_step, TARGET_UPDATE, NUM_ENVS):
            target_net.load_state_dict(q_net.state_dict())


        # Guardar checkpoints periódicos del modelo entrenado cada 100k pasos
        # if (global_step % 250_000 == 0) < NUM_ENVS and global_step > 0 or global_step == TOTAL_STEPS - 1:
        if (global_step % 250_000) < NUM_ENVS and global_step > 0 or global_step >= TOTAL_STEPS - NUM_ENVS:
            
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d_step{global_step}.pt")
            
            # Hacemos un pequeño test de evaluación del modelo guardado para verificar que se ha guardado correctamente (con 10 episodios de prueba)
            q_net.eval()
            test_rewards = []
            
            eval_env = gym.make(ENV_ID, render_mode="rgb_array")
            # eval_env = ForwardAliveSmoothReward(eval_env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
            eval_env = IgnoreAngleTerminationWrapper(eval_env)
            eval_env = DiscreteActionWrapper(eval_env)
            eval_env = RGBObsWrapper(eval_env)

            for ep in tqdm(range(10)):
                obs_eval, _ = eval_env.reset() # Semilla diferente para el test de evaluación para mayor diversidad
                
                obs_eval_np = np.ascontiguousarray(obs_eval)  # Aseguramos que la observación es contigua en memoria para evitar warnings de PyTorch
                frame_eval = preprocess_rgb_batch_torch(obs_eval_np[None, ...], out_size=84, device="cpu")
                state_eval = frame_eval.repeat(1, 4, 1, 1).contiguous()

                test_episode_reward = 0.0
                while True:
                    with torch.no_grad():
                        s = state_eval.to(DEVICE, non_blocking=True).float().div_(255.0)  # (1,4,84,84)
                        action = int(q_net(s).argmax(dim=1).item())

                    next_obs, reward, terminated, truncated, infos = eval_env.step(action) # era test_state antes de next_obs
                    test_episode_reward += float(reward)
                    
                    next_obs_eval_np = np.ascontiguousarray(next_obs)
                    # preprocess next frame + update stack
                    next_frame = preprocess_rgb_batch_torch(next_obs_eval_np[None, ...], out_size=84, device="cpu")
                    state_eval = torch.cat([state_eval[:, 1:], next_frame], dim=1).contiguous()
                    
                    if terminated or truncated:
                        break

                test_rewards.append(test_episode_reward)

            avg_test_reward = float(np.mean(test_rewards))
            eval_env.close()

            print(f"Checkpoint saved at step {global_step}, average test reward over 10 episodes: {avg_test_reward}")
            writer.add_scalar("avg_test_reward", avg_test_reward, global_step) # Registramos la recompensa media del test de evaluación en TensorBoard (ajustamos el paso para que coincida con el número total de pasos incluyendo los 3M iniciales)
            q_net.train() # Volvemos a poner la red en modo entrenamiento después del test de evaluación
    
        # step counters
        global_step += NUM_ENVS
        pbar.update(NUM_ENVS)

    row = {
        "model_dir": MODEL_DIR[4:],
        "seed": SEED,

        # hiperparámetros
        "total_steps": TOTAL_STEPS,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "lr": LR,
        "target_update": TARGET_UPDATE,
        "start_training": START_TRAINING,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay": EPS_DECAY,

        # métricas resumen
        f"avg_eval_reward": avg_test_reward,
        "n_episodes": n_episodes,
        "comments": "subiendo batch size",
    }
    
    print(f"avg_eval_reward: {avg_test_reward:.2f}, n_episodes: {n_episodes}")

    save_experiment_to_excel(row, EXPERIMENT_XLSX)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")

    pbar.close()
    env.close()
    writer.close()
    
    # Guardamos el modelo entrenado al finalizar el entrenamiento
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d.pt")


if __name__ == "__main__":
    main()
