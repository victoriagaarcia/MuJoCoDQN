import gymnasium as gym
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.dqn import QNetwork, ReplayBuffer
from src.envs import DiscreteActionWrapper, PixelStackWrapper
from datetime import datetime

# -----------------------------
# Hiperparámetros
# -----------------------------
ENV_ID = "Walker2d-v5"
TOTAL_STEPS = 500_000 # Número total de pasos de interacción con el entorno (no episodios)
BUFFER_SIZE = 100_000 # Capacidad máxima del replay buffer (número de transiciones almacenadas)
BATCH_SIZE = 32 # Tamaño del batch para el entrenamiento de la red Q
GAMMA = 0.99 # Ponderación del valor futuro en la actualización de Q (factor de descuento)
LR = 1e-4
TARGET_UPDATE = 5_000 # Frecuencia de actualización de la red objetivo (en pasos de interacción)
START_TRAINING = 10_000 # Número de pasos de interacción antes de empezar a entrenar (para llenar el buffer con experiencias iniciales)

EPS_START = 1.0 # Valor inicial de epsilon para la política epsilon-greedy (probabilidad de acción aleatoria)
EPS_END = 0.1 # Valor final de epsilon después de la fase de decaimiento (probabilidad mínima de acción aleatoria)
EPS_DECAY = 300_000 # Número de pasos durante los cuales epsilon decae linealmente desde EPS_START hasta EPS_END

SEED = 42 # Semilla para reproducibilidad
LAST_EPISODES = 100 # Número de episodios finales para calcular la recompensa media al finalizar el entrenamiento
EXPERIMENT_XLSX = "runs/experiment_results.xlsx" # Archivo Excel para guardar los resultados de los experimentos

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "runs/" + datetime.now().strftime("%b%d_%H_%M_%S") # Directorio para guardar el modelo entrenado y los logs de TensorBoard

def epsilon(step):
    return max(EPS_END, EPS_START - step / EPS_DECAY)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    writer = SummaryWriter(MODEL_DIR) # Creamos un escritor de TensorBoard para registrar métricas durante el entrenamiento

    # Creamos el entorno con renderizado en modo "rgb_array" para obtener frames como imágenes
    env = gym.make(ENV_ID, render_mode="rgb_array")
    # Envolvemos el entorno para discretizar las acciones y apilar frames de píxeles
    env = DiscreteActionWrapper(env)
    # Envolvemos el entorno para convertir las observaciones en stacks de frames de píxeles preprocesados (grises y redimensionados) CONTINUOS
    env = PixelStackWrapper(env)

    n_actions = env.action_space.n

    # Creamos la red Q (online: para seleccionar acciones) y la red objetivo (target: para calcular los objetivos de entrenamiento)
    q_net = QNetwork(n_actions).to(DEVICE)
    target_net = QNetwork(n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict()) # Inicializamos la red objetivo con los mismos pesos que la red online

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    state, _ = env.reset(seed=SEED) # Reiniciamos el entorno y obtenemos el estado inicial (stack de frames)
    episode_reward = 0.0
    n_episodes = 0

    for step in tqdm(range(TOTAL_STEPS)):
        eps = epsilon(step) # Calculamos el valor de epsilon para esta etapa del entrenamiento (decay lineal)

        # Decisión exploración vs explotación según epsilon-greedy
        if np.random.rand() < eps:
            action = env.action_space.sample() # Acción aleatoria (exploración)
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action = q_net(s).argmax(dim=1).item() # Acción con mayor valor Q según la red online (explotación)

        # Ejecutamos la acción en el entorno y obtenemos la siguiente transición (s, a, r, s', done)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Guardamos la transición en el replay buffer
        buffer.push(state, action, reward, next_state, done)
        
        # Actualizamos el estado actual al siguiente estado
        state = next_state
        # Acumulamos la recompensa del episodio actual
        episode_reward += reward

        if done: # Si el episodio ha terminado, registramos la recompensa total del episodio en TensorBoard y reiniciamos el entorno
            writer.add_scalar("episode_reward", episode_reward, step)
            state, _ = env.reset()
            episode_reward = 0.0 
            n_episodes += 1

        if len(buffer) > START_TRAINING: # Empezamos a entrenar la red Q solo después de haber llenado el buffer con suficientes experiencias iniciales
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
                target = rewards + GAMMA * max_next_q * (1 - dones) # Objetivo de entrenamiento: r + gamma * max_a' Q_target(s', a') si no es terminal, solo r si es terminal

            loss = torch.nn.functional.mse_loss(q_values, target) # Calculamos la pérdida como el error cuadrático medio entre los valores Q actuales y los objetivos de entrenamiento

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("epsilon", eps, step)
            

        if step % TARGET_UPDATE == 0: # Cada cierto número de pasos, actualizamos la red objetivo copiando los pesos de la red online
            target_net.load_state_dict(q_net.state_dict())
        
        # Guardar checkpoints periódicos del modelo entrenado cada 100k pasos
        if step % 100_000 == 0 and step > 0 or step == TOTAL_STEPS - 1:
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d_step{step}.pt")
            # Hacemos un pequeño test de evaluación del modelo guardado para verificar que se ha guardado correctamente (con 10 episodios de prueba)
            q_net.eval()
            test_rewards = []
            for _ in tqdm(range(10)):
                test_state, _ = env.reset()
                test_episode_reward = 0.0
                while True:
                    with torch.no_grad():
                        s = torch.tensor(test_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        action = q_net(s).argmax(dim=1).item()
                    test_state, reward, terminated, truncated, _ = env.step(action)
                    test_episode_reward += reward
                    if terminated or truncated:
                        break
                test_rewards.append(test_episode_reward)
            avg_test_reward = np.mean(test_rewards)
            print(f"Checkpoint saved at step {step}, average test reward over 10 episodes: {avg_test_reward}")
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
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(EXPERIMENT_XLSX):
        df_old = pd.read_excel(EXPERIMENT_XLSX)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(EXPERIMENT_XLSX, index=False)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")

    env.close()
    writer.close()
    
    # Guardamos el modelo entrenado al finalizar el entrenamiento
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/dqn_walker2d.pt")


if __name__ == "__main__":
    main()
