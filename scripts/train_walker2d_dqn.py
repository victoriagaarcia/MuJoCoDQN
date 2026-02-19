import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv

from src.dqn import QNetwork, ReplayBuffer
from src.envs import DiscreteActionWrapper, PixelStackWrapper, ForwardAliveSmoothReward, IgnoreAngleTerminationWrapper

# -----------------------------
# Hiperparámetros
# -----------------------------


ENV_ID = "Walker2d-v5"
TOTAL_STEPS = 6_000_000 # Número total de pasos de interacción con el entorno (no episodios)
BUFFER_SIZE = 500_000 # Capacidad máxima del replay buffer (número de transiciones almacenadas)
BATCH_SIZE = 64 # Tamaño del batch para el entrenamiento de la red Q
GAMMA = 0.99 # Ponderación del valor futuro en la actualización de Q (factor de descuento)
LR = 1e-4
TARGET_UPDATE = 40_000 # Frecuencia de actualización de la red objetivo (en pasos de interacción)
START_TRAINING = 50_000 # Número de pasos de interacción antes de empezar a entrenar (para llenar el buffer con experiencias iniciales)

EPS_START = 1.0 # Valor inicial de epsilon para la política epsilon-greedy (probabilidad de acción aleatoria)
# EPS_START = 0.1
EPS_END = 0.1 # Valor final de epsilon después de la fase de decaimiento (probabilidad mínima de acción aleatoria)
EPS_DECAY = 3_000_000 # Número de pasos durante los cuales epsilon decae linealmente desde EPS_START hasta EPS_END
START_DECAY = 50_000 # Número de pasos antes de empezar a decaer epsilon 
SEED = 42 # Semilla para reproducibilidad
LAST_EPISODES = 100 # Número de episodios finales para calcular la recompensa media al finalizar el entrenamiento
EXPERIMENT_XLSX = "runs/experiments.xlsx" # Archivo Excel para guardar los resultados de los experimentos

NUM_ENVS = 4 # Número de entornos paralelos para entrenamiento 

# Configuracion de reward
ALPHA_RW = 1.5
BETA_RW = 1.0
GAMMA_RW = 0.8
DELTA_RW = 1.0
LAM_RW = 0.05


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "runs/" + datetime.now().strftime("%b%d_%H_%M_%S") # Directorio para guardar el modelo entrenado y los logs de TensorBoard
# MODEL_DIR = f"runs/Feb14_20_38_13" # Directorio para guardar el modelo entrenado y los logs de TensorBoard (ajusta esto)

# MODEL_DATE = "Feb14_20_38_13"
# MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d.pt"  # ← ajusta esto
# MODEL_PATH = f"runs/{MODEL_DATE}/dqn_walker2d_step3000000.pt"  # ← ajusta esto

def epsilon(step):
   # return max(EPS_END, EPS_START - (step  / EPS_DECAY))
   return max(EPS_END, EPS_START - (max(0, step - START_DECAY) / EPS_DECAY)) # Decay lineal con fase inicial de epsilon constante

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
        env = ForwardAliveSmoothReward(env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
        env = IgnoreAngleTerminationWrapper(env)
        env = DiscreteActionWrapper(env)
        env = PixelStackWrapper(env)
        return env
    return _thunk

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    writer = SummaryWriter(MODEL_DIR) # Creamos un escritor de TensorBoard para registrar métricas durante el entrenamiento

    # # Creamos el entorno con renderizado en modo "rgb_array" para obtener frames como imágenes
    # env = gym.make(ENV_ID, render_mode="rgb_array")
    # # Envolvemos el entorno para discretizar las acciones y apilar frames de píxeles
    # env = DiscreteActionWrapper(env)
    # # Envolvemos el entorno para convertir las observaciones en stacks de frames de píxeles preprocesados (grises y redimensionados) CONTINUOS
    # env = PixelStackWrapper(env)

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
    state, _ = env.reset(seed=seeds) # Reiniciamos el entorno y obtenemos el estado inicial (stack de frames)
    # episode_reward = 0.0
    # n_episodes = 0

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    n_episodes = 0
    avg_test_reward = np.nan  # para que exista incluso si no llegas a guardar checkpoint

    # Contador transiciones reales
    global_step = 0

    # Escala target update por NUM_ENVS (en términos de transiciones reales)
    def should_update_target(gs: int) -> bool:
        return (gs % TARGET_UPDATE) < NUM_ENVS
    
    total_iters = TOTAL_STEPS // NUM_ENVS

    for it in tqdm(range(total_iters)):
        eps = epsilon(global_step) # Calculamos el valor de epsilon para esta etapa del entrenamiento (decay lineal)

        # acciones aleatorias por entorno (vector)
        actions = np.empty((NUM_ENVS,), dtype=np.int64)
        rand_mask = np.random.rand(NUM_ENVS) < eps # Máscara booleana para decidir qué entornos toman acción aleatoria
        
        n_rand = int(rand_mask.sum())
        if n_rand > 0:
            actions[rand_mask] = np.array(
                [env.single_action_space.sample() for _ in range(n_rand)],
                dtype=np.int64
            ) # Acción aleatoria para los entornos seleccionados por la máscara
        
        # Decisión exploración vs explotación según epsilon-greedy
        if (~rand_mask).any(): # Si hay algún entorno que no toma acción aleatoria, calculamos la acción con la red Q para esos entornos
            with torch.no_grad():
                s = torch.tensor(state[~rand_mask], dtype=torch.float32).to(DEVICE)
                greedy = q_net(s).argmax(dim=1).cpu().numpy() # Acciones con mayor valor Q según la red online para los entornos que no toman acción aleatoria
            actions[~rand_mask] = greedy
        
        # Ejecutamos la acción en el entorno y obtenemos la siguiente transición (s, a, r, s', done)
        next_state, reward, terminated, truncated, info = env.step(actions)
        done = np.logical_or(terminated, truncated) 

        if isinstance(info, dict) and "final_observation" in info:
            final_obs = info["final_observation"]
            final_mask = info.get("_final_observation", None)
            if final_mask is None:
                final_mask = done
            for i in range(NUM_ENVS):
                if bool(final_mask[i]) and bool(done[i]):
                    # Reemplazamos el siguiente estado por la observación final para los entornos que han terminado el episodio 
                    # (esto es importante para que el agente aprenda correctamente a partir de la transición final)
                    next_state[i] = final_obs[i] 
        
        for i in range(NUM_ENVS):
            buffer.push(
                state[i],
                int(actions[i]),
                float(reward[i]),
                next_state[i],
                bool(done[i])
            )
        
        # # Guardamos la transición en el replay buffer
        # buffer.push(state, actions, reward, next_state, done)
        
        # # Actualizamos el estado actual al siguiente estado
        # state = next_state
        # # Acumulamos la recompensa del episodio actual
        # episode_rewards += reward
        
        episode_rewards += reward.astype(np.float32) # Acumulamos la recompensa del episodio actual para cada entorno

        if done.any(): # Si el episodio ha terminado, registramos la recompensa total del episodio en TensorBoard y reiniciamos el entorno
            done_ids = np.where(done)[0]
            for i in done_ids:
                writer.add_scalar("episode_reward", float(episode_rewards[i]), global_step) # Registramos la recompensa total del episodio en TensorBoard (ajustamos el paso para que coincida con el número total de pasos incluyendo los 3M iniciales)
                episode_rewards[i] = 0.0
                n_episodes += 1
            
        state = next_state # Actualizamos el estado actual al siguiente estado para la próxima iteración

        global_step += NUM_ENVS # Incrementamos el contador de pasos global por el número de entornos paralelos (cada iteración representa NUM_ENVS pasos reales)

        if len(buffer) > START_TRAINING: # Empezamos a entrenar la red Q solo después de haber llenado el buffer con suficientes experiencias iniciales
            # Muestreamos un batch aleatorio de transiciones del buffer para entrenar la red Q
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            # states = states.to(DEVICE)
            # actions = actions.to(DEVICE)
            # rewards = rewards.to(DEVICE)
            # next_states = next_states.to(DEVICE)
            # dones = dones.to(DEVICE)
            # El buffer ya devuelve los tensores en el dispositivo correcto, así que no es necesario moverlos aquí

            # Calculamos los valores Q actuales para las acciones tomadas usando la red online
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target = rewards + GAMMA * max_next_q * (1 - dones) # Objetivo de entrenamiento: r + gamma * max_a' Q_target(s', a') si no es terminal, solo r si es terminal

            loss = torch.nn.functional.mse_loss(q_values, target) # Calculamos la pérdida como el error cuadrático medio entre los valores Q actuales y los objetivos de entrenamiento

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0) # Clipping de gradientes para evitar explosión de gradientes
            optimizer.step()

            writer.add_scalar("loss", loss.item(), global_step) # Registramos la pérdida en TensorBoard (ajustamos el paso para que coincida con el número total de pasos incluyendo los 3M iniciales)
            writer.add_scalar("epsilon", eps, global_step) # Registramos el valor de epsilon en TensorBoard (ajustamos el paso para que coincida con el número total de pasos incluyendo los 3M iniciales)
            
        if should_update_target(global_step):
            target_net.load_state_dict(q_net.state_dict())

        # Guardar checkpoints periódicos del modelo entrenado cada 100k pasos
        # if (global_step % 250_000 == 0) < NUM_ENVS and global_step > 0 or global_step == TOTAL_STEPS - 1:
        if (global_step % 250_000) < NUM_ENVS and global_step > 0 or global_step >= TOTAL_STEPS - NUM_ENVS:
            
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d_step{global_step}.pt")
            # Hacemos un pequeño test de evaluación del modelo guardado para verificar que se ha guardado correctamente (con 10 episodios de prueba)
            q_net.eval()
            test_rewards = []
            
            eval_env = gym.make(ENV_ID, render_mode="rgb_array")
            eval_env = ForwardAliveSmoothReward(eval_env, alpha=ALPHA_RW, beta=BETA_RW, gamma=GAMMA_RW, delta=DELTA_RW, lam=LAM_RW)
            eval_env = IgnoreAngleTerminationWrapper(eval_env)
            eval_env = DiscreteActionWrapper(eval_env)
            eval_env = PixelStackWrapper(eval_env)

            for ep in tqdm(range(10)):
                test_state, _ = eval_env.reset(seed=SEED + 10_000 + ep) # Semilla diferente para el test de evaluación para mayor diversidad
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
    
            print(f"Checkpoint saved at step {global_step}, average test reward over 10 episodes: {avg_test_reward}")
            writer.add_scalar("avg_test_reward", avg_test_reward, global_step) # Registramos la recompensa media del test de evaluación en TensorBoard (ajustamos el paso para que coincida con el número total de pasos incluyendo los 3M iniciales)
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

    env.close()
    writer.close()
    
    # Guardamos el modelo entrenado al finalizar el entrenamiento
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d.pt")


if __name__ == "__main__":
    main()
