# train_walker2d_rainbow.py
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
    RainbowDQN,
    projection_distribution,
)

from .utils import (
    should_update_target,
    save_experiment_to_excel,
    to_uint8_stack,
)

# -----------------------------
# Hiperparámetros
# -----------------------------
ENV_ID = "Walker2d-v5"

TOTAL_STEPS = 5_000_000
BUFFER_SIZE = 500_000
BATCH_SIZE = 64

GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 40_000
START_TRAINING = 50_000

# Rainbow extras
N_STEP = 3
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0

# C51
N_ATOMS = 51
V_MIN = -50.0
V_MAX = 200.0

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

    # Online y target (C51 + Dueling + Noisy)
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

    # PER buffer + n-step accumulator
    buffer = PrioritizedReplayBuffer(
        capacity=BUFFER_SIZE,
        device=DEVICE,
        obs_shape=(4, 84, 84),
        alpha=PER_ALPHA
    )

    nstep = NStepAccumulator(n=N_STEP, gamma=GAMMA, n_envs=NUM_ENVS)

    seeds = [SEED + i for i in range(NUM_ENVS)]
    obs, _ = env.reset(seed=seeds)
    state = to_uint8_stack(obs)  # (B,4,84,84) uint8/torch (como vuestro DQN) :contentReference[oaicite:3]{index=3}

    # stats
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    n_episodes = 0
    updates_done = 0
    avg_test_reward = np.nan

    # target update en unidades "steps del loop"
    target_update_steps = max(1, TARGET_UPDATE // NUM_ENVS)

    global_step = 0

    for step in tqdm(range(TOTAL_STEPS), desc="train_steps(rainbow)"):
        # ---------------------------------------------------------
        # ACT (Noisy -> greedy; warmup random al principio)
        # ---------------------------------------------------------
        actions = np.empty((NUM_ENVS,), dtype=np.int64)

        if step < START_TRAINING:
            actions[:] = np.array([env.single_action_space.sample() for _ in range(NUM_ENVS)], dtype=np.int64)
        else:
            with torch.no_grad():
                q_net.train()
                q_net.reset_noise()
                s = state.to(DEVICE, non_blocking=True).float().div_(255.0)  # (B,4,84,84)
                q_vals = q_net.get_q_values(s)  # (B,A)
                actions[:] = q_vals.argmax(dim=1).cpu().numpy().astype(np.int64)

        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        episode_done = np.logical_or(terminated, truncated)
        done_boot = terminated  # consistente con vuestro DQN :contentReference[oaicite:4]{index=4}

        # logs reward shaping (si existen en infos)
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
        # ---------------------------------------------------------
        for i in range(NUM_ENVS):
            outs = nstep.add(
                env_id=i,
                state=state[i].cpu(),
                action=int(actions[i]),
                reward=float(rewards[i]),
                next_state=next_state[i].cpu(),
                done=bool(done_boot[i]),  # ojo: aquí usas done_boot como en tu clase actual
            )
            for (s0, a0, Rn, sn, dn) in outs:
                buffer.push(s0, a0, Rn, sn, float(dn))

        state = next_state

        # ---------------------------------------------------------
        # EPISODE LOG + reset vector envs (como vuestro flujo)
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
        # UPDATE (PER + Double + C51 + n-step + Noisy)
        # ---------------------------------------------------------
        if len(buffer) > START_TRAINING:
            beta = beta_schedule(step)

            # tu buffer devuelve: s,a,r,ns,d,weights,idxs
            states_b, actions_t, rewards_t, next_states_b, dones_t, weights, idxs = buffer.sample(BATCH_SIZE, beta=beta)

            # logits/probs online para (s,a) actuales
            q_net.reset_noise()
            logits = q_net(states_b, return_probs=False)  # (B,A,N)
            log_probs = F.log_softmax(logits, dim=-1).clamp(min=-30.0)  # estabilidad

            # seleccionar acción del batch
            act_idx = actions_t.view(-1, 1, 1).expand(-1, 1, N_ATOMS)
            log_probs_sa = log_probs.gather(1, act_idx).squeeze(1)  # (B,N)

            with torch.no_grad():
                # Double DQN para C51:
                # a* = argmax_a E[Z] usando online
                # q_net.reset_noise() # a ver si así se arregla el inplace
                next_q_online = q_net.get_q_values(next_states_b)  # (B,A)
                a_star = next_q_online.argmax(dim=1)              # (B,)

                # next_dist = target(s',a*) (B,N)
                target_net.reset_noise()
                next_probs_all = target_net(next_states_b, return_probs=True)  # (B,A,N)
                next_dist = next_probs_all[torch.arange(BATCH_SIZE, device=DEVICE), a_star]  # (B,N)

                # proyección C51 -> target_dist (B,N)
                # Tu projection_distribution espera (B,A,N), así que le pasamos (B,1,N)
                target_dist = projection_distribution(
                    next_dist=next_dist.unsqueeze(1),
                    rewards=rewards_t,
                    dones=dones_t,
                    gamma=GAMMA,
                    n_step=N_STEP,
                    support=q_net.support,  # buffer (N,)
                    v_min=V_MIN,
                    v_max=V_MAX
                )  # (B,N)

            # loss cross-entropy: - sum target * log p
            per_sample_loss = -(target_dist * log_probs_sa).sum(dim=1)  # (B,)
            loss = (weights * per_sample_loss).mean()

            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

            # PER priorities: usa la loss por muestra como “error”
            new_prios = per_sample_loss.detach() + 1e-6
            buffer.update_priorities(idxs, new_prios)

            updates_done += 1

            if step % LOG_EVERY == 0:
                writer.add_scalar("loss", float(loss.item()), step)
                writer.add_scalar("per/beta", float(beta), step)
                writer.add_scalar("per/mean_weight", float(weights.mean().item()), step)
                writer.add_scalar("per/mean_sample_loss", float(per_sample_loss.mean().item()), step)
                writer.add_scalar("updates_done", updates_done, step)
                writer.add_scalar("buffer_size", len(buffer), step)

        # ---------------------------------------------------------
        # TARGET UPDATE
        # ---------------------------------------------------------
        if step % target_update_steps == 0:
            target_net.load_state_dict(q_net.state_dict())

        # ---------------------------------------------------------
        # CHECKPOINT + EVAL
        # ---------------------------------------------------------
        if (step % CHECKPOINT_EVERY == 0 and step > 0) or (step >= TOTAL_STEPS - 1):
            torch.save(q_net.state_dict(), f"{MODEL_DIR}/rainbow_walker2d_step{step}.pt")

            # mini-eval (10 episodios)
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

        global_step += NUM_ENVS

    # save final
    torch.save(q_net.state_dict(), f"{MODEL_DIR}/rainbow_walker2d.pt")

    row = {
        "model_dir": MODEL_DIR[5:],
        "seed": SEED,
        "algo": "Rainbow(C51+Double+PER+Nstep+Noisy+Dueling)",
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
        "n_atoms": N_ATOMS,
        "v_min": V_MIN,
        "v_max": V_MAX,
        "sigma_init": SIGMA_INIT,
        "avg_eval_reward": avg_test_reward,
        "n_episodes": n_episodes,
        "comments": "rainbow C51 training script",
    }

    save_experiment_to_excel(row, EXPERIMENT_XLSX)
    print(f"[Excel] Appended results to {EXPERIMENT_XLSX}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()