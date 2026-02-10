import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.dqn import QNetwork, ReplayBuffer
from src.envs import DiscreteActionWrapper, PixelStackWrapper

# -----------------------------
# Hiperparámetros
# -----------------------------
ENV_ID = "Walker2d-v5"
TOTAL_STEPS = 500_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 5_000
START_TRAINING = 10_000

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 300_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def epsilon(step):
    return max(EPS_END, EPS_START - step / EPS_DECAY)


def main():
    writer = SummaryWriter("runs/dqn_walker2d")

    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DiscreteActionWrapper(env)
    env = PixelStackWrapper(env)

    n_actions = env.action_space.n

    q_net = QNetwork(n_actions).to(DEVICE)
    target_net = QNetwork(n_actions).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    state, _ = env.reset()
    episode_reward = 0.0

    for step in range(TOTAL_STEPS):
        eps = epsilon(step)

        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action = q_net(s).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            writer.add_scalar("episode_reward", episode_reward, step)
            state, _ = env.reset()
            episode_reward = 0.0

        if len(buffer) > START_TRAINING:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            rewards = rewards.to(DEVICE)
            next_states = next_states.to(DEVICE)
            dones = dones.to(DEVICE)

            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target = rewards + GAMMA * max_next_q * (1 - dones)

            loss = torch.nn.functional.mse_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("epsilon", eps, step)

        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
