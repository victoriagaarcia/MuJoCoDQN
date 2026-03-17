# Ablation study part 1:
# Rainbow SIN parte distribucional (sin C51)
#
# Se mantiene:
# - Double DQN
# - Prioritized Experience Replay
# - n-step returns
# - Noisy Nets
# - Dueling architecture
#
# Cambio principal respecto a train_walker2d_rainbow.py:
# - La red ya no es RainbowDQN (C51), sino NoisyandDuelingDQN
# - La loss deja de ser distribución-proyectada + cross-entropy
# - Pasamos a TD escalar con Double DQN + n-step + PER

import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import AsyncVectorEnv

from src.envs_antiguo import (
    DiscreteActionWrapper,
    ProgressWithSafetyShaping,
    PixelStackWrapper,
)

from src.rainbow import (
    PrioritizedReplayBuffer,
    NStepAccumulator,
    NoisyandDuelingDQN,   # <-- red no distribucional
)

from .utils import (
    should_update_target,
    save_experiment_to_excel,
    to_uint8_stack,
)

# -----------------------------
# Hiperpar�metros
# -----------------------------
ENV_ID = "Walker2d-v5"

TOTAL_STEPS = 15_000_000
BUFFER_SIZE = 500_000
BATCH_SIZE = 64

GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 40_000
START_TRAINING = 50_000

# Rainbow extras que sí mantenemos
N_STEP = 3
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0

# Noisy
SIGMA_INIT = 0.017

SEED = 42
NUM_ENVS = 8

LOG_EVERY = 5_000
CHECKPOINT_EVERY = 250_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "runs/" + datetime.now().strftime("%b%d_%H_%M_%S")
EXPERIMENT_XLSX = "runs/experiments.xlsx"


def make_env(rank: int):
    def _thunk():
        env = gym.make(ENV_ID, render_mode="rgb_array", width=480, height=480)
        env = DiscreteActionWrapper(env)
        env = ProgressWithSafetyShaping(env)
        env = PixelStackWrapper(env, k=4, size=84)
        return env
    return _thunk


def beta_schedule(step: int) -> float:
    # lineal PER beta -> 1.0 a lo largo de TOTAL_STEPS
    t = min(1.0, float(step) / float(TOTAL_STEPS))
    return PER_BETA_START + t * (PER_BETA_END - PER_BETA_START)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    writer = SummaryWriter(MODEL_DIR)

    env = AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])
    n_actions = env.single_action_space.n

    # ---------------------------------------------------------
    # CAMBIO RESPECTO A RAINBOW ORIGINAL:
    # usamos red no distribucional (Q escalar) pero mantenemos
    # Noisy + Dueling
    # ---------------------------------------------------------
    q_net = NoisyandDuelingDQN(
        num_actions=n_actions,
        sigma_init=SIGMA_INIT
    ).to(DEVICE)

    target_net = NoisyandDuelingDQN(
        num_actions=n_actions,
        sigma_init=SIGMA_INIT
    ).to(DEVICE)

    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

    # PER buffer + n-step accumulator (igual que Rainbow original)
    buffer = PrioritizedReplayBuffer(
        capacity=BUFFER_SIZE,
        device=DEVICE,
        obs_shape=(4, 84, 84),
        alpha=PER_ALPHA
    )

    nstep = NStepAccumulator(n=N_STEP, gamma=GAMMA, n_envs=NUM_ENVS)

    seeds = [SEED + i for i in range(NUM_ENVS)]
    obs, _ = env.reset(seed=seeds)
    state = to_uint8_stack(obs)

    # stats
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    n_episodes = 0
    updates_done = 0
    avg_test_reward = np.nan

    # target update en unidades "steps del loop"
    target_update_steps = max(1, TARGET_UPDATE // NUM_ENVS)

    global_step = 0

    for step in tqdm(range(TOTAL_STEPS), desc="train_steps(ablation1_no_distributional)"):
        # ---------------------------------------------------------
        # ACT (Noisy -> greedy; warmup random al principio)
        # Igual que Rainbow original
        # ---------------------------------------------------------
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
                q_vals = q_net(s)  # <-- CAMBIO: la red ya devuelve Q-values
                actions[:] = q_vals.argmax(dim=1).cpu().numpy().astype(np.int64)

        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        episode_done = np.logical_or(terminated, truncated)
        done_boot = terminated  # mismo criterio que tu Rainbow original

        # logs reward shaping
        writer.add_scalar("rewards/base", float(np.mean(infos.get("debug/base", np.nan))), step)
        writer.add_scalar("rewards/speed_bonus", float(np.mean(infos.get("debug/speed_bonus", np.nan))), step)
        writer.add_scalar("rewards/height_pen", float(np.mean(infos.get("debug/height_pen", np.nan))), step)
        writer.add_scalar("rewards/angle_pen", float(np.mean(infos.get("debug/angle_pen", np.nan))), step)
        writer.add_scalar("rewards/alive_bonus", float(np.mean(infos.get("debug/alive_bonus", np.nan))), step)

        episode_rewards += rewards.astype(np.float32)
        episode_lengths += 1

        next_state = to_uint8_stack(next_obs)

        # ---------------------------------------------------------
        # PUSH en n-step -> PER
        # Igual que Rainbow original
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # EPISODE LOG + reset vector envs
        # Igual que Rainbow original
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # UPDATE
        #
        # CAMBIO CR�TICO respecto a Rainbow original:
        # antes: C51 + projection_distribution + cross-entropy
        # ahora: Double DQN escalar + n-step + PER
        # ---------------------------------------------------------
        if len(buffer) > START_TRAINING:
            beta = beta_schedule(step)

            states_b, actions_t, rewards_t, next_states_b, dones_t, weights, idxs = buffer.sample(
                BATCH_SIZE,
                beta=beta
            )

            # Q(s,a) online
            q_net.reset_noise()
            q_values = q_net(states_b)  # (B, A)
            q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

            with torch.no_grad():
                # Double DQN:
                # a* = argmax_a Q_online(s', a)
                # q_net.reset_noise()
                next_q_online = q_net(next_states_b)  # (B, A)
                a_star = next_q_online.argmax(dim=1, keepdim=True)  # (B,1)

                # Q_target(s', a*)
                target_net.reset_noise()
                next_q_target = target_net(next_states_b)  # (B, A)
                next_q = next_q_target.gather(1, a_star).squeeze(1)  # (B,)

                td_target = rewards_t + (GAMMA ** N_STEP) * (1.0 - dones_t) * next_q  # (B,)

            td_error = td_target - q_sa
            per_sample_loss = F.smooth_l1_loss(q_sa, td_target, reduction="none")  # (B,)
            loss = (weights * per_sample_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            # PER priorities: mejor usar |TD error|
            new_prios = td_error.detach().abs() + 1e-6
            buffer.update_priorities(idxs, new_prios)

            updates_done += 1

            if step % LOG_EVERY == 0:
                writer.add_scalar("loss", float(loss.item()), step)
                writer.add_scalar("per/beta", float(beta), step)
                writer.add_scalar("per/mean_weight", float(weights.mean().item()), step)
                writer.add_scalar("per/mean_sample_loss", float(per_sample_loss.mean().item()), step)
                writer.add_scalar("td/mean_abs_error", float(td_error.abs().mean().item()), step)
                writer.add_scalar("td/mean_q_sa", float(q_sa.mean().item()), step)
                writer.add_scalar("td/mean_target", float(td_target.mean().item()), step)
                writer.add_scalar("updates_done", updates_done, step)
                writer.add_scalar("buffer_size", len(buffer), step)

        # ---------------------------------------------------------
        # TARGET UPDATE
        # Igual que Rainbow original
        # ---------------------------------------------------------
        if step % target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ---------------------------------------------------------
        # CHECKPOINT + EVAL
        # Igual estructura que Rainbow original
        # ---------------------------------------------------------
        if (step % CHECKPOINT_EVERY == 0 and step > 0) or (step >= TOTAL_STEPS - 1):
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/ablation1_walker2d_step{step}.pt")

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
                        q = q_net(s)
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

        global_step += NUM_ENVS

    # save final
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/ablation1_walker2d.pt")

    row = {
        "model_dir": MODEL_DIR[5:],
        "seed": SEED,
        "algo": "Rainbow ablation 1 (NO Distributional RL; keep Double+PER+Nstep+Noisy+Dueling)",
        "total_steps": TOTAL_STEPS,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "n_step": N_STEP,
        "lr": LR,
        "target_update": TARGET_UPDATE,
        "start_training": START_TRAINING,
        "per_alpha": PER_ALPHA,
        "per_beta_start": PER_BETA_START,
        "per_beta_end": PER_BETA_END,
        "sigma_init": SIGMA_INIT,
        "avg_eval_reward": avg_test_reward,
        "n_episodes": n_episodes,
        "comments": "ablation study part 1: Rainbow without C51/distributional component",
    }

    save_experiment_to_excel(row, EXPERIMENT_XLSX)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()