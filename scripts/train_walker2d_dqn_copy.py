import os
from datetime import datetime
from turtle import done

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv

from src.dqn_copy import QNetwork, ReplayBuffer
from src.envs_copy import (
    DiscreteActionWrapper,
    Gray84ObsWrapper,
    ForwardAliveSmoothReward, 
    ReduceAngleTerminationWrapper,
    ProgressWithSafetyShaping,
    PixelStackWrapper,
    RGBObsWrapper
)
from .utils import (
    epsilon,
    preprocess_rgb_batch_torch,
    should_update_target,
    save_experiment_to_excel,
    to_uint8_stack
)

# -----------------------------
# Hiperparámetros
# -----------------------------

ENV_ID = "Walker2d-v5"

TOTAL_STEPS = 8_000_000 # Número total de pasos de interacción con el entorno (no episodios)
BUFFER_SIZE = 500_000 # Capacidad máxima del replay buffer (número de transiciones almacenadas)
BATCH_SIZE = 64 # Tamaño del batch para el entrenamiento de la red Q
GAMMA = 0.99 # Ponderación del valor futuro en la actualización de Q (factor de descuento)
LR = 1e-4
TARGET_UPDATE = 40_000 # Frecuencia de actualización de la red objetivo (en pasos de interacción)
START_TRAINING = 50_000 # Número de pasos de interacción antes de empezar a entrenar (para llenar el buffer con experiencias iniciales)

EPS_START = 1.0 # Valor inicial de epsilon para la política epsilon-greedy (probabilidad de acción aleatoria)
# EPS_START = 0.1
EPS_END = 0.1 # Valor final de epsilon después de la fase de decaimiento (probabilidad mínima de acción aleatoria)
EPS_DECAY = 4_000_000 # Número de pasos durante los cuales epsilon decae linealmente desde EPS_START hasta EPS_END
START_DECAY = 0 # Número de pasos antes de empezar a decaer epsilon 

SEED = 42 # Semilla para reproducibilidad
NUM_ENVS = 8 # Número de entornos paralelos para entrenamiento 

# Configuracion de reward
ALPHA_RW = 2.0 # 1.5
BETA_RW = 1.0
GAMMA_RW = 0.8
DELTA_RW = 1.0
LAM_RW = 0.05

# Train con saltos
TRAIN_FREQ = 1 # Como hay 4 envs, poner 4 es como hacer 1 update por iteración
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
        env = gym.make(ENV_ID, render_mode="rgb_array", width=480, height=480)
        # env = ForwardAliveSmoothReward(env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
        env = ReduceAngleTerminationWrapper(env)
        env = DiscreteActionWrapper(env)
        # env = RGBObsWrapper(env)
        # env = Gray84ObsWrapper(env, size=84) 
        env = ProgressWithSafetyShaping(env)
        env = PixelStackWrapper(env, k=4, size=84) # Apilamos 4 frames preprocesados (grayscale 84x84) para captar movimiento y convertir la observación a un formato adecuado para la red Q
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
    
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    net_param_ids = {id(p) for p in q_net.parameters()}
    
    # print("optimizer params", len(opt_param_ids), "net params", len(net_param_ids))
    # print("missing in optimizer", len(net_param_ids - opt_param_ids))
    
    buffer = ReplayBuffer(BUFFER_SIZE, obs_shape=(4,84,84), device=DEVICE) 

    seeds = [SEED + i for i in range(NUM_ENVS)] # Semillas diferentes para cada entorno paralelo para mayor diversidad de experiencias
    # state, _ = env.reset(seed=seeds) # Reiniciamos el entorno y obtenemos el estado inicial (stack de frames)
    # episode_reward = 0.0
    # n_episodes = 0

    obs, info = env.reset(seed=seeds) # Reiniciamos el entorno y obtenemos el estado inicial (batch de stacks de frames)
    state = to_uint8_stack(obs) # Convertimos la observación inicial a uint8 y al formato (B,4,84,84) para el buffer

    target_update_steps = max(1, TARGET_UPDATE // NUM_ENVS)
    # Frame inicial -> stack 4
    
    # frame = preprocess_rgb_batch_torch(obs, out_size=84, device="cpu") # (B,1,84,84) uint8
    # state = frame.repeat(1, 4, 1, 1).contiguous()
    
    # frame = torch.from_numpy(obs).unsqueeze(1)          # (B,1,84,84) uint8
    # state = frame.repeat(1, 4, 1, 1).contiguous()       # (B,4,84,84) uint8

    # obs_chw = transpose_obs_batch(obs) # Transponemos las observaciones al formato (B, C, H, W) 
    # state = np.repeat(obs_chw, 4, axis=1) # Creamos el stack inicial de 4 frames repitiendo la misma observación 4 veces

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    n_episodes = 0
    updates_done = 0  # Contador de updates
    avg_test_reward = np.nan  # para que exista incluso si no llegas a guardar checkpoint

    # Contador transiciones reales
    global_step = 0

    # pbar = tqdm(total=TOTAL_STEPS, desc="train_steps")
    # while global_step < TOTAL_STEPS:
    for step in tqdm(range(TOTAL_STEPS)):
    # for it in tqdm(range(total_iters)):
        # eps = epsilon(global_step, EPS_END, EPS_START, 
        #               START_DECAY, EPS_DECAY) # Calculamos el valor de epsilon para esta etapa del entrenamiento (decay lineal)
        eps = epsilon(step, EPS_END, EPS_START, 
                      START_DECAY, EPS_DECAY) # Calculamos el valor de epsilon para esta etapa del entrenamiento (decay lineal)

        # # epsilon-greedy (batch)
        # if np.random.rand() < eps:
        #     actions = np.array([env.single_action_space.sample() for _ in range(NUM_ENVS)], dtype=np.int64)
        # else:
        #     with torch.no_grad():
        #         s = state.to(DEVICE, non_blocking=True).float().div_(255.0)  # (B,4,84,84) float
        #         q = q_net(s)
        #         actions = q.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

        # epsilon-greedy PER-ENV (mezcla random/greedy por subentorno)
        actions = np.empty((NUM_ENVS,), dtype=np.int64)
        rand_mask = (np.random.rand(NUM_ENVS) < eps)

        # random donde toca
        n_rand = int(rand_mask.sum())
        if n_rand > 0:
            actions[rand_mask] = np.array(
                [env.single_action_space.sample() for _ in range(n_rand)],
                dtype=np.int64
            )

        # greedy donde toca
        if (~rand_mask).any():
            with torch.no_grad():
                s = state[~rand_mask].to(DEVICE, non_blocking=True).float().div_(255.0)
                q = q_net(s).argmax(dim=1).cpu().numpy()
            actions[~rand_mask] = q

        # Ejecutamos la acción en el entorno vectorizado
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        # next_frame = env.render() # Renderizamos el entorno para obtener los frames RGB (si el entorno lo soporta)
        episode_done = np.logical_or(terminated, truncated)
        done_boot = terminated

        # if global_step % 10_000 < NUM_ENVS:
            # m = next_obs.mean(axis=(1,2,3))
            # s = next_obs.std(axis=(1,2,3))
            # writer.add_scalar("debug/obs_mean_min", float(m.min()), global_step)
            # writer.add_scalar("debug/obs_mean_max", float(m.max()), global_step)
            # writer.add_scalar("debug/obs_std_min", float(s.min()), global_step)
            # writer.add_scalar("debug/obs_std_max", float(s.max()), global_step)
            # 
            # writer.add_scalar("debug/done_rate", float(done.mean()), global_step)


        episode_rewards+= rewards.astype(np.float32)
        episode_lengths += 1
        
        # preprocess next frames (batch)
        
        # next_frame = preprocess_rgb_batch_torch(next_obs, out_size=84, device="cpu") # (B,1,84,84) uint8
        # next_state = torch.cat([state[:, 1:], next_frame], dim=1).contiguous()        # (B,4,84,84
        
        # next_frame = torch.from_numpy(next_obs).unsqueeze(1)    # (B,1,84,84) uint8
        # next_state = torch.cat([state[:, 1:], next_frame], dim=1).contiguous()
        
        # if isinstance(infos, dict) and "final_observation" in infos:
        #     final_obs = infos["final_observation"]
        #     final_mask = infos.get("_final_observation", done) 
        #     
        #     idx = np.where(final_mask & done)[0]
        #     if idx.size > 0:
        #         term_frame = preprocess_rgb_batch_torch(final_obs[idx], out_size=84, device="cpu")
        #         next_state_buf[idx] = torch.cat([state[idx, 1:], term_frame], dim=1).contiguous()
        # if global_step % 10_000 < NUM_ENVS:
        #     m = next_frame.float().mean(axis=(1,2,3)).numpy()
        #     s = next_frame.float().std(axis=(1,2,3)).numpy()
        #     writer.add_scalar("debug/frame_mean_min", float(m.min()), global_step)
        #     writer.add_scalar("debug/frame_mean_max", float(m.max()), global_step)
        #     writer.add_scalar("debug/frame_std_min", float(s.min()), global_step)
        #     writer.add_scalar("debug/frame_std_max", float(s.max()), global_step)

        #     last = state[:, -1].to(torch.int16)
        #     new = next_frame[:, 0].to(torch.int16)
        #     writer.add_scalar("debug/mean abs (last-new)", (last-new).abs().float().mean().item(), global_step)
        
        next_state = to_uint8_stack(next_obs) # Convertimos el siguiente estado a uint8 y al formato (B,4,84,84) para el buffer

        # Ahora pusheamos todo el batch
        buffer.push_batch(
            states=state,
            actions=actions,
            rewards=rewards,
            next_states=next_state,
            dones=done_boot
        )
        
        if step % 10_000 < NUM_ENVS:
            has_final = int(isinstance(infos, dict) and ("final_observation" in infos))
            writer.add_scalar("debug/has_final_observation", has_final, step)
            writer.add_scalar("debug/buffer_size", len(buffer), step)
            if has_final:
                fm = infos.get("_final_observation", episode_done)
                writer.add_scalar("debug/final_obs_count", int(np.sum(fm)), step)
        
        # # --- reset SOLO de los entornos que han terminado ---
        # if np.any(episode_done):
        if episode_done.any():
            # Para los envs reseteados, reiniciamos el stack con su primer frame
            done_idx = np.where(episode_done)[0]
            for i in done_idx: 
                writer.add_scalar("episode_reward", float(episode_rewards[i]), step)
                writer.add_scalar("episode_length", int(episode_lengths[i]), step)
                n_episodes += 1
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

            obs, _ = env.reset(seed=seeds)
            state = to_uint8_stack(obs)

        if len(buffer) > START_TRAINING:
            states_b, actions_t, reward_t, next_states_b, dones_t = buffer.sample(BATCH_SIZE)
            # buffer.sample ya devuelve tensores en DEVICE (según tu implementación)
            q_values = q_net(states_b).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # DQN estándar
                max_next_q = target_net(next_states_b).max(1)[0]
                target = reward_t + GAMMA * max_next_q * (1.0 - dones_t)

            loss = torch.nn.functional.mse_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()

            # if global_step % 10_000 < NUM_ENVS:
            #     total_norm_sq = 0.0
            #     for p in q_net.parameters():
            #         if p.grad is not None:
            #             n = p.grad.data.norm(2).item()
            #             total_norm_sq += n * n
            #     grad_norm = total_norm_sq ** 0.5
            #     writer.add_scalar("debug/grad_norm", grad_norm, global_step)

            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            # if global_step % 10_000 < NUM_ENVS:
            #     with torch.no_grad():
            #         p = next(q_net.parameters())
            #         p_before = p.detach().clone()

            optimizer.step()

            # print("n_with_state:", n_with_state, "n_with_grad:", n_with_grad,
            #   "exp_avg_max_all:", ea_max, "exp_avg_sq_max_all:", eas_max)
            # with torch.no_grad():
            #     max_change = 0.0
            #     for k, v in q_net.state_dict().items():
            #         max_change = max(max_change, (v - before[k]).abs().max().item())
            # print("STATE_DICT max_change:", max_change)

            # p0 = next(q_net.parameters())
            # st = optimizer.state.get(p0, {})
            # ea = st.get("exp_avg", None)
            # eas = st.get("exp_avg_sq", None)
            # print("has_state:", bool(st), "exp_avg_max:", None if ea is None else ea.abs().max().item(),
            #     "exp_avg_sq_max:", None if eas is None else eas.abs().max().item())

            # # ---- CHECK CHANGE ----
            # with torch.no_grad():
            #     p0_after = next(q_net.parameters()).detach()
            #     diff_max = (p0_after - p0_before).abs().max().item()
            #     diff_mean = (p0_after - p0_before).abs().mean().item()

            # print("param change: max", diff_max, "mean", diff_mean)

            # if global_step % 10_000 < NUM_ENVS:
            #     with torch.no_grad():
            #         p = next(q_net.parameters())
            #         diff_max = (p - p0_before).abs().max().item()
            #         diff_mean = (p - p0_before).abs().mean().item()
            #     writer.add_scalar("debug/param_diff_max", diff_max, global_step)
            #     writer.add_scalar("debug/param_diff_mean", diff_mean, global_step)
            updates_done += 1

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("epsilon", eps, step)
            writer.add_scalar("updates_done", updates_done, step)

            # if global_step % 10_000 < NUM_ENVS:
            #     s, a, r, ns, d = buffer.sample(64)
            #     delta = (s[:, -1] - ns[:, -1]).abs().mean().item()
            #     writer.add_scalar("debug/sample_state_delta", delta, global_step)

            #     with torch.no_grad():
            #         writer.add_scalar("debug/updates_done", updates_done, global_step)

            #         writer.add_scalar("debug/q_mean", float(q_values.mean().item()), global_step)
            #         writer.add_scalar("debug/q_std", float(q_values.std().item()), global_step)
                    
            #         writer.add_scalar("debug/target_mean", float(target.mean().item()), global_step)
            #         writer.add_scalar("debug/target_std", float(target.std().item()), global_step)
                    
            #         writer.add_scalar("debug/reward_mean", float(reward_t.mean().item()), global_step)
            #         writer.add_scalar("debug/reward_std", float(reward_t.std().item()), global_step)
                    
            #         writer.add_scalar("debug/done_mean", float(dones_t.mean().item()), global_step)

        # if should_update_target(global_step, TARGET_UPDATE, NUM_ENVS):
        #     target_net.load_state_dict(q_net.state_dict())
        if step % target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Guardar checkpoints periódicos del modelo entrenado cada 100k pasos
        # if (global_step % 250_000 == 0) < NUM_ENVS and global_step > 0 or global_step == TOTAL_STEPS - 1:
        # if (global_step % 250_000) < NUM_ENVS and global_step > 0 or global_step >= TOTAL_STEPS - NUM_ENVS:
        if step % 250_000 == 0 and step > 0 or step >= TOTAL_STEPS - 1:  
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d_step{step}.pt")
            
            # Hacemos un pequeño test de evaluación del modelo guardado para verificar que se ha guardado correctamente (con 10 episodios de prueba)
            q_net.eval()
            test_rewards = []
            
            eval_env = gym.make(ENV_ID, render_mode="rgb_array", width=480, height=480)
            # eval_env = ForwardAliveSmoothReward(eval_env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
            eval_env = ReduceAngleTerminationWrapper(eval_env)
            eval_env = DiscreteActionWrapper(eval_env)
            eval_env = ProgressWithSafetyShaping(eval_env)
            eval_env = PixelStackWrapper(eval_env, k=4, size=84) # Mismo preprocesamiento que en el entrenamiento para que la red pueda procesar las observaciones correctamente
            # eval_env = RGBObsWrapper(eval_env)
            # eval_env = Gray84ObsWrapper(eval_env, size=84)

            for ep in tqdm(range(10)):
                obs_eval, _ = eval_env.reset(seed=SEED + 10_000 + ep) # Semilla diferente para el test de evaluación para mayor diversidad
                obs_eval_b = obs_eval[None, ...] # (1,H,W,3) uint8
                state_eval = to_uint8_stack(obs_eval_b) # (1,4,84,84) uint8, stack inicial con 4 frames iguales

                # frame_eval = preprocess_rgb_batch_torch(obs_eval_np[None, ...], out_size=84, device="cpu")
                # state_eval = frame_eval.repeat(1, 4, 1, 1).contiguous()

                # frame_eval = torch.from_numpy(obs_eval_b).unsqueeze(1)  # (1,1,84,84) uint8
                # state_eval = frame_eval.repeat(1, 4, 1, 1).contiguous()  # (1,4,84,84) uint8

                test_episode_reward = 0.0
                while True:
                    with torch.no_grad():
                        s = state_eval.to(DEVICE, non_blocking=True).float().div_(255.0)  # (1,4,84,84)
                        action = int(q_net(s).argmax(dim=1).item())

                    next_obs_eval, reward, terminated, truncated, infos = eval_env.step(action) # era test_state antes de next_obs
                    test_episode_reward += float(reward)
                    
                    # next_obs_eval_np = np.ascontiguousarray(next_obs)
                    # # preprocess next frame + update stack
                    # # next_frame = preprocess_rgb_batch_torch(next_obs_eval_np[None, ...], out_size=84, device="cpu")
                    # # state_eval = torch.cat([state_eval[:, 1:], next_frame], dim=1).contiguous()
                    
                    # next_frame = torch.from_numpy(next_obs_eval_np).unsqueeze(0).unsqueeze(1)  # (1,1,84,84) uint8
                    # state_eval = torch.cat([state_eval[:, 1:], next_frame], dim=1).contiguous()  # (1,4,84,84) uint8
                    state_eval = to_uint8_stack(next_obs_eval[None, ...]) # (1,4,84,84) uint8, actualizamos el stack con el nuevo frame

                    if terminated or truncated:
                        break

                test_rewards.append(test_episode_reward)

            avg_test_reward = float(np.mean(test_rewards))
            eval_env.close()

            print(f"Checkpoint saved at step {step}, average test reward over 10 episodes: {avg_test_reward}")
            writer.add_scalar("avg_test_reward", avg_test_reward, step) # Registramos la recompensa media del test de evaluación en TensorBoard (ajustamos el paso para que coincida con el número total de pasos incluyendo los 3M iniciales)
            q_net.train() # Volvemos a poner la red en modo entrenamiento después del test de evaluación
    
        # step counters
        global_step += NUM_ENVS
        # pbar.update(NUM_ENVS)

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

    # pbar.close()
    env.close()
    writer.close()
    
    # Guardamos el modelo entrenado al finalizar el entrenamiento
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d.pt")


if __name__ == "__main__":
    main()
