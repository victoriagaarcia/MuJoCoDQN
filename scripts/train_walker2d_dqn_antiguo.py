import gymnasium as gym
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv
from src.dqn_antiguo import QNetwork, ReplayBuffer
from src.envs_antiguo import (
    DiscreteActionWrapper,
    ProgressWithSafetyShaping,
    PixelStackWrapper)

from datetime import datetime

# Hiperparámetros
ENV_ID = "Walker2d-v5"
TOTAL_STEPS = 10_000_000
BUFFER_SIZE = 200_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4 
TARGET_UPDATE = 40_000
START_TRAINING = 50_000

EPS_START = 1.0
# EPS_START = 0.1
EPS_END1 = 0.1 
EPS_END2 = 0.05
EPS_DECAY = 1_500_000
START_DECAY = 0 
SEED = 42
LAST_EPISODES = 100
EXPERIMENT_XLSX = "runs/experiments.xlsx"
NUM_ENVS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "runs/" + datetime.now().strftime("%b%d_%H_%M_%S") 

def epsilon(step):
    if step < EPS_DECAY:
       return max(EPS_END1, EPS_START - (max(0, step - START_DECAY) / EPS_DECAY))
    elif step < 2 * EPS_DECAY:
        return EPS_END1
    else:
        return EPS_END2

def save_experiment_to_excel(row_dict, filename="runs/experiments.xlsx"):
    # Convertimos el diccionario en un DataFrame de una sola fila
    new_df = pd.DataFrame([row_dict])
    
    # Comprobamos si el archivo ya existe
    if not os.path.isfile(filename):
        # Si no existe, creamos el archivo con cabeceras
        new_df.to_excel(filename, index=False, engine='openpyxl')
    else:
        # Si ya existe, abrimos el archivo y añadimos la fila al final
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # Cargamos la hoja actual para saber dónde escribir
            try:
                start_row = writer.book['Sheet1'].max_row
            except KeyError:
                start_row = 0
            
            # Escribimos los datos sin repetir la cabecera (header=False)
            new_df.to_excel(writer, index=False, header=False, startrow=start_row, sheet_name='Sheet1')

def make_env(rank:int):
    def _thunk():
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = DiscreteActionWrapper(env)
        env = ProgressWithSafetyShaping(env)
        env = PixelStackWrapper(env)
        return env
    return _thunk

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    writer = SummaryWriter(MODEL_DIR) 

    env = AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)]) 
    n_actions = env.single_action_space.n

    # Creamos la red Q y la red objetivo
    q_net = QNetwork(n_actions).to(DEVICE)

    target_net = QNetwork(n_actions).to(DEVICE)
    # Inicializamos la red objetivo con los mismos pesos que la red online
    target_net.load_state_dict(q_net.state_dict()) 

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    seeds = [SEED + i for i in range(NUM_ENVS)] 
    # Reiniciamos el entorno y obtenemos el estado inicial (stack de frames)
    state, _ = env.reset(seed=seeds) 

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    n_episodes = 0
    # para que exista incluso si no llegas a guardar checkpoint
    avg_test_reward = np.nan  

    # Escala target update por NUM_ENVS
    target_update_steps = max(1, TARGET_UPDATE // NUM_ENVS)
    
    for step in tqdm(range(TOTAL_STEPS)):
        eps = epsilon(step) 
        
        # acciones aleatorias por entorno
        actions = np.empty((NUM_ENVS,), dtype=np.int64)
        # Máscara booleana para decidir qué entornos toman acción aleatoria
        rand_mask = np.random.rand(NUM_ENVS) < eps 
        
        n_rand = int(rand_mask.sum())
        if n_rand > 0:
            # Acción aleatoria para los entornos seleccionados por la máscara
            actions[rand_mask] = np.array(
                [env.single_action_space.sample() for _ in range(n_rand)],
                dtype=np.int64
            )
        
        # Decisión exploración vs explotación según epsilon-greedy
        # Si hay algún entorno que no toma acción aleatoria, calculamos la acción con la red Q para esos entornos
        if (~rand_mask).any(): 
            with torch.no_grad():
                s = torch.tensor(state[~rand_mask], dtype=torch.float32).to(DEVICE)
                # Acciones con mayor valor Q según la red online para los entornos que no toman acción aleatoria
                greedy = q_net(s).argmax(dim=1).cpu().numpy() 
            actions[~rand_mask] = greedy
        
        # Ejecutamos la acción en el entorno y obtenemos la siguiente transición (s, a, r, s', done)
        next_state, reward, terminated, truncated, _ = env.step(actions)
        done = np.logical_or(terminated, truncated) 

        for i in range(NUM_ENVS):
            buffer.push(
                state[i],
                int(actions[i]),
                float(reward[i]),
                next_state[i],
                bool(done[i])
            )

        # Acumulamos la recompensa del episodio actual para cada entorno
        episode_rewards += reward.astype(np.float32) #

        if done.any(): 
            done_ids = np.where(done)[0]
            for i in done_ids:
                # Registramos la recompensa total del episodio en TensorBoard
                writer.add_scalar("episode_reward", float(episode_rewards[i]), step) 
                episode_rewards[i] = 0.0
                n_episodes += 1
            
            state, _ = env.reset(seed=seeds)
        else:
            # Actualizamos el estado actual al siguiente estado para la próxima iteración
            state = next_state 

        # Empezamos a entrenar la red Q solo después de haber llenado el buffer con suficientes experiencias iniciales
        if len(buffer) > START_TRAINING: 
            # Muestreamos un batch aleatorio de transiciones del buffer para entrenar la red Q
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)
            next_states = next_states.to(DEVICE)
            dones = dones.to(DEVICE)

            # Calculamos los valores Q actuales para las acciones tomadas usando la red online
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                # Objetivo de entrenamiento: r + gamma * max_a' Q_target(s', a') si no es terminal, solo r si es terminal
                target = rewards + GAMMA * max_next_q * (1 - dones) 

            # Calculamos la pérdida como el error cuadrático medio entre los valores Q actuales y los objetivos de entrenamiento
            loss = torch.nn.functional.mse_loss(q_values, target) 

            optimizer.zero_grad()
            loss.backward()
            
            # Clipping de gradientes para evitar explosión de gradientes
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0) 
            optimizer.step()

            # Registramos la pérdida en TensorBoard
            writer.add_scalar("loss", loss.item(), step) 
             # Registramos el valor de epsilon en TensorBoard
            writer.add_scalar("epsilon", eps, step)
            
        # Cada cierto número de pasos, actualizamos la red objetivo copiando los pesos de la red online
        if step % target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())
        
        # Guardar checkpoints periódicos del modelo entrenado cada 100k pasos
        if step % 250_000 == 0 and step > 0 or step == TOTAL_STEPS - 1:
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d_step{step}.pt")

            q_net.eval()
            test_rewards = []
            
            eval_env = gym.make(ENV_ID, render_mode="rgb_array")
            eval_env = DiscreteActionWrapper(eval_env)
            eval_env = ProgressWithSafetyShaping(eval_env)
            eval_env = PixelStackWrapper(eval_env)

            for ep in tqdm(range(10)):
                # Semilla diferente para el test de evaluación para mayor diversidad
                test_state, _ = eval_env.reset(seed=SEED + 10_000 + ep) 
                test_episode_reward = 0.0
                while True:
                    with torch.no_grad():
                        s = torch.tensor(test_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        action = q_net(s).argmax(dim=1).item()
                    test_state, reward, terminated, truncated, _ = eval_env.step(action)
                    test_episode_reward += reward
                    if terminated or truncated:
                        break
                test_rewards.append(test_episode_reward)
            avg_test_reward = np.mean(test_rewards)
            print(f"Checkpoint saved at step {step}, average test reward over 10 episodes: {avg_test_reward}")
            # Registramos la recompensa media del test de evaluación en TensorBoard
            writer.add_scalar("avg_test_reward", avg_test_reward, step)
            q_net.train() # Volvemos a poner la red en modo entrenamiento después del test de evaluación
    
    
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
        "eps_end1": EPS_END1,
        "eps_end2": EPS_END2,
        "eps_decay": EPS_DECAY,

        # métricas resumen
        f"avg_eval_reward": avg_test_reward,
        "n_episodes": n_episodes,
        "comments": "subiendo batch size",
    }
    
    print(f"avg_eval_reward: {avg_test_reward:.2f}, n_episodes: {n_episodes}")

    save_experiment_to_excel(row, EXPERIMENT_XLSX)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")

    env.close()
    writer.close()
    
    # Guardamos el modelo entrenado al finalizar el entrenamiento
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d.pt")


if __name__ == "__main__":
    main()
