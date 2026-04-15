# Ablation study part 2:
# Rainbow sin Prioritized Experience Replay (PER)
#
# Se mantiene:
# - Distributional RL (C51)
# - Double DQN
# - Dueling architecture
# - Noisy Nets
# - n-step returns
#
# Cambio principal respecto a train_walker2d_rainbow.py:
# - Se reemplaza PrioritizedReplayBuffer por un replay uniforme
# - Se elimina beta_schedule, los weights de importance sampling
# - Se elimina update_priorities(...)
# - La loss pasa a ser la media simple de la cross-entropy por muestra


import os
from datetime import datetime
from collections import deque
import random

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv

from src.envs_antiguo import (
    DiscreteActionWrapper,
    ProgressWithSafetyShaping,
    PixelStackWrapper,
)

from src.rainbow import (
    NStepAccumulator,
    RainbowDQN,
    projection_distribution,
)

from .utils import (
    save_experiment_to_excel,
    to_uint8_stack,
)

ENV_ID = "Walker2d-v5"

TOTAL_STEPS = 5_000_000
BUFFER_SIZE = 500_000
BATCH_SIZE = 64

GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 40_000
START_TRAINING = 50_000

N_STEP = 3

N_ATOMS = 51
V_MIN = -80.0
V_MAX = 500.0

SIGMA_INIT = 0.017

SEED = 42
NUM_ENVS = 8

LOG_EVERY = 5_000
CHECKPOINT_EVERY = 100_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "runs/" + datetime.now().strftime("%b%d_%H_%M_%S")
EXPERIMENT_XLSX = "runs/experiments.xlsx"


class UniformReplayBuffer:
    """
    Replay buffer uniforme para reemplazar PER.
    Guarda transiciones n-step: (state, action, reward_n, next_state, done).
    """
    def __init__(self, capacity: int, device: str, obs_shape=(4, 84, 84)):
        self.capacity = int(capacity)
        self.device = device
        self.obs_shape = obs_shape

        self.states = torch.empty((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.empty((capacity,), dtype=torch.long)
        self.rewards = torch.empty((capacity,), dtype=torch.float32)
        self.next_states = torch.empty((capacity, *obs_shape), dtype=torch.uint8)
        self.dones = torch.empty((capacity,), dtype=torch.float32)

        self.pos = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.pos

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos].copy_(state)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_states[self.pos].copy_(next_state)
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        size = len(self)
        idxs = np.random.randint(0, size, size=batch_size)

        states_b = self.states[idxs].to(self.device, non_blocking=True).float().div_(255.0)
        actions_t = self.actions[idxs].to(self.device, non_blocking=True)
        rewards_t = self.rewards[idxs].to(self.device, non_blocking=True)
        next_states_b = self.next_states[idxs].to(self.device, non_blocking=True).float().div_(255.0)
        dones_t = self.dones[idxs].to(self.device, non_blocking=True)

        return states_b, actions_t, rewards_t, next_states_b, dones_t


def make_env(rank: int):
    def _thunk():
        env = gym.make(ENV_ID, render_mode="rgb_array", width=480, height=480)
        env = DiscreteActionWrapper(env)
        env = ProgressWithSafetyShaping(env)
        env = PixelStackWrapper(env, k=4, size=84)
        return env
    return _thunk


def main():
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    writer = SummaryWriter(MODEL_DIR)

    env = AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])
    n_actions = env.single_action_space.n

    q_net = RainbowDQN(
        num_actions=n_actions,
        n_atoms=N_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        sigma_init=SIGMA_INIT
    ).to(DEVICE)

    target_net = RainbowDQN(
        num_actions=n_actions,
        n_atoms=N_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        sigma_init=SIGMA_INIT
    ).to(DEVICE)

    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

    # Buffer uniforme en lugar de PER.
    buffer = UniformReplayBuffer(
        capacity=BUFFER_SIZE,
        device=DEVICE,
        obs_shape=(4, 84, 84),
    )

    nstep = NStepAccumulator(n=N_STEP, gamma=GAMMA, n_envs=NUM_ENVS)

    seeds = [SEED + i for i in range(NUM_ENVS)]
    obs, _ = env.reset(seed=seeds)
    state = to_uint8_stack(obs)

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    n_episodes = 0
    updates_done = 0
    avg_test_reward = np.nan

    target_update_steps = max(1, TARGET_UPDATE // NUM_ENVS)

    for step in tqdm(range(TOTAL_STEPS), desc="train_steps(ablation2_without_per)"):
        # Warmup aleatorio; luego greedy sobre NoisyNet.
        actions = np.empty((NUM_ENVS,), dtype=np.int64)

        if step < START_TRAINING:
            actions[:] = np.array(
                [env.single_action_space.sample() for _ in range(NUM_ENVS)],
                dtype=np.int64
            )
        else:
            with torch.no_grad():
                q_net.train()
                q_net.reset_noise()
                s = state.to(DEVICE, non_blocking=True).float().div_(255.0)
                q_vals = q_net.get_q_values(s)
                actions[:] = q_vals.argmax(dim=1).cpu().numpy().astype(np.int64)

        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        episode_done = np.logical_or(terminated, truncated)
        done_boot = terminated

        writer.add_scalar("rewards/base", float(np.mean(infos.get("debug/base", np.nan))), step)
        writer.add_scalar("rewards/speed_bonus", float(np.mean(infos.get("debug/speed_bonus", np.nan))), step)
        writer.add_scalar("rewards/height_pen", float(np.mean(infos.get("debug/height_pen", np.nan))), step)
        writer.add_scalar("rewards/angle_pen", float(np.mean(infos.get("debug/angle_pen", np.nan))), step)
        writer.add_scalar("rewards/alive_bonus", float(np.mean(infos.get("debug/alive_bonus", np.nan))), step)

        episode_rewards += rewards.astype(np.float32)
        episode_lengths += 1

        next_state = to_uint8_stack(next_obs)

        # Acumula n-step y escribe en replay uniforme.
        for i in range(NUM_ENVS):
            outs = nstep.add(
                env_id=i,
                state=state[i].cpu(),
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_state=next_state[i].cpu(),
                done=bool(done_boot[i]),
            )
            for (s0, a0, Rn, sn, dn) in outs:
                buffer.push(s0, a0, Rn, sn, float(dn))

        state = next_state

        if episode_done.any():
            done_idx = np.where(episode_done)[0]
            for i in done_idx:
                writer.add_scalar("episode_reward", float(episode_rewards[i]), step)
                writer.add_scalar("episode_length", int(episode_lengths[i]), step)
                n_episodes += 1
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

            obs, _ = env.reset(seed=seeds)
            state = to_uint8_stack(obs)

        # Update C51 + Double DQN con sample uniforme (sin PER).
        if len(buffer) > START_TRAINING:
            states_b, actions_t, rewards_t, next_states_b, dones_t = buffer.sample(BATCH_SIZE)

            q_net.reset_noise()
            logits_all = q_net(states_b, return_probs=False)
            log_probs_all = torch.log_softmax(logits_all, dim=-1)
            log_probs_sa = log_probs_all[torch.arange(BATCH_SIZE, device=DEVICE), actions_t]
            
            with torch.no_grad():
                # Double DQN: la accion se selecciona con la online.
                next_q_online = q_net.get_q_values(next_states_b)
                a_star = next_q_online.argmax(dim=1)

                target_net.reset_noise()
                next_probs_all = target_net(next_states_b, return_probs=True)
                next_dist = next_probs_all[torch.arange(BATCH_SIZE, device=DEVICE), a_star]

                # Proyeccion C51.
                target_dist = projection_distribution(
                    next_dist=next_dist.unsqueeze(1),
                    rewards=rewards_t,
                    dones=dones_t,
                    gamma=GAMMA,
                    n_step=N_STEP,
                    support=q_net.support,
                    v_min=V_MIN,
                    v_max=V_MAX
                )

            per_sample_loss = -(target_dist * log_probs_sa).sum(dim=1)

            # Media simple: no hay IS weights de PER.
            loss = per_sample_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            updates_done += 1

            if step % LOG_EVERY == 0:
                writer.add_scalar("loss", float(loss.item()), step)
                writer.add_scalar("uniform_replay/mean_sample_loss", float(per_sample_loss.mean().item()), step)
                writer.add_scalar("updates_done", updates_done, step)
                writer.add_scalar("buffer_size", len(buffer), step)

        if step % target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (step % CHECKPOINT_EVERY == 0 and step > 0) or (step >= TOTAL_STEPS - 1):
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/ablation2_woper_walker2d_step{step}.pt")

            q_net.eval()
            q_net.disable_noise()
            test_rewards = []

            eval_env = gym.make(ENV_ID, render_mode="rgb_array", width=480, height=480)
            eval_env = DiscreteActionWrapper(eval_env)
            eval_env = ProgressWithSafetyShaping(eval_env)
            eval_env = PixelStackWrapper(eval_env, k=4, size=84)

            for ep in range(10):
                obs_eval, _ = eval_env.reset(seed=SEED + 10_000 + ep)
                state_eval = to_uint8_stack(obs_eval[None, ...])

                ep_ret = 0.0
                while True:
                    with torch.no_grad():
                        s = state_eval.to(DEVICE, non_blocking=True).float().div_(255.0)
                        q = q_net.get_q_values(s)
                        action = int(q.argmax(dim=1).item())

                    next_obs_eval, reward, terminated, truncated, _ = eval_env.step(action)
                    ep_ret += float(reward)
                    state_eval = to_uint8_stack(next_obs_eval[None, ...])

                    if terminated or truncated:
                        break

                test_rewards.append(ep_ret)

            avg_test_reward = float(np.mean(test_rewards))
            eval_env.close()

            writer.add_scalar("avg_test_reward", avg_test_reward, step)
            print(f"[ckpt step {step}] avg_test_reward(10eps) = {avg_test_reward:.2f}")

            q_net.train()
            q_net.enable_noise()

    torch.save(q_net.state_dict(), f"{MODEL_DIR}/ablation2_woper_walker2d.pt")

    row = {
        "model_dir": MODEL_DIR[5:],
        "seed": SEED,
        "algo": "Rainbow ablation 2 (WITH Distributional RL, WITHOUT PER)",
        "total_steps": TOTAL_STEPS,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "n_step": N_STEP,
        "lr": LR,
        "target_update": TARGET_UPDATE,
        "start_training": START_TRAINING,
        "n_atoms": N_ATOMS,
        "v_min": V_MIN,
        "v_max": V_MAX,
        "sigma_init": SIGMA_INIT,
        "avg_eval_reward": avg_test_reward,
        "n_episodes": n_episodes,
        "comments": "ablation study part 2: Rainbow keeps C51/Double/Dueling/Noisy/n-step and removes PER",
    }

    save_experiment_to_excel(row, EXPERIMENT_XLSX)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()