import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from src.dqn_antiguo import QNetwork
from src.rainbow import RainbowDQN as QNetwork

from src.envs import (
    DiscreteActionWrapper,
    ProgressWithSafetyShaping,
    PixelStackWrapper,
)

ProgressWithSafetyShapingNew = None
HAS_NEW_SHAPING = False


ENV_ID = "Walker2d-v5"


def extract_step(path: Path) -> int:
    """
    Extrae el n�mero de step de un checkpoint tipo:
    dqn_walker2d_step15000000.pt
    """
    m = re.search(r"step(\d+)\.pt$", path.name)
    if m is None:
        return -1
    return int(m.group(1))


def build_env(use_new_shaping: bool):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = DiscreteActionWrapper(env)

    if use_new_shaping and HAS_NEW_SHAPING:
        env = ProgressWithSafetyShapingNew(env)
    else:
        env = ProgressWithSafetyShaping(env)

    env = PixelStackWrapper(env)
    return env


def summarize(values):
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0))


def evaluate_checkpoint(
    model_path: Path,
    n_episodes: int,
    seed: int,
    device: str,
    use_new_shaping: bool,
):
    env = build_env(use_new_shaping=use_new_shaping)

    q_net = QNetwork(env.action_space.n).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    metrics = defaultdict(list)

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + 10_000 + ep)

        done = False
        ep_return = 0.0
        ep_len = 0

        base_sum = 0.0
        speed_bonus_sum = 0.0
        height_pen_sum = 0.0
        angle_pen_sum = 0.0
        alive_bonus_sum = 0.0

        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                # action = q_net(s).argmax(dim=1).item()
                q_values = q_net.get_q_values(s)
                action = q_values.argmax(dim=1).item()

            state, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)

            base_sum += float(info.get("debug/base", 0.0))
            speed_bonus_sum += float(info.get("debug/speed_bonus", 0.0))
            height_pen_sum += float(info.get("debug/height_pen", 0.0))
            angle_pen_sum += float(info.get("debug/angle_pen", 0.0))
            alive_bonus_sum += float(info.get("debug/alive_bonus", 0.0))

        # Estado final fisico del torso
        data = env.unwrapped.data
        final_z = float(data.qpos[1])
        final_angle = float(data.qpos[2])
        final_abs_angle = abs(final_angle)
        final_healthy = float(env.unwrapped.is_healthy)

        metrics["return"].append(ep_return)
        metrics["episode_length"].append(ep_len)
        metrics["terminated"].append(float(terminated))
        metrics["truncated"].append(float(truncated))
        metrics["final_torso_height"].append(final_z)
        metrics["final_torso_angle"].append(final_angle)
        metrics["final_abs_torso_angle"].append(final_abs_angle)
        metrics["final_healthy"].append(final_healthy)

        metrics["base_reward_sum"].append(base_sum)
        metrics["speed_bonus_sum"].append(speed_bonus_sum)
        metrics["height_pen_sum"].append(height_pen_sum)
        metrics["angle_pen_sum"].append(angle_pen_sum)
        metrics["alive_bonus_sum"].append(alive_bonus_sum)

    env.close()

    summary = {}
    for key, values in metrics.items():
        mean, std = summarize(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std

    # Guardamos tambien distribuciones brutas utiles para histogramas
    summary["_raw_returns"] = np.asarray(metrics["return"], dtype=np.float32)
    summary["_raw_lengths"] = np.asarray(metrics["episode_length"], dtype=np.float32)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directorio de la corrida, p.ej. runs/Feb26_10_30_09")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--every", type=int, default=1,
                        help="Evalua 1 de cada N checkpoints (util si hay muchos)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximo numero de checkpoints a evaluar; 0 = todos")
    parser.add_argument("--use_old_shaping", action="store_true",
                        help="Fuerza ProgressWithSafetyShaping en lugar de ProgressWithSafetyShapingNew")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"No existe el directorio: {run_dir}"

    ckpts = sorted(
        run_dir.glob("rainbow_walker2d_step*.pt"),
        key=extract_step
    )

    ckpts = [p for p in ckpts if extract_step(p) >= 0]

    if args.every > 1:
        ckpts = ckpts[::args.every]

    if args.limit > 0:
        ckpts = ckpts[:args.limit]

    if len(ckpts) == 0:
        raise RuntimeError(f"No se han encontrado checkpoints en {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = run_dir / f"eval_checkpoints_tb_{timestamp}"
    csv_path = run_dir / f"checkpoint_eval_summary_{timestamp}.csv"

    writer = SummaryWriter(str(tb_dir))
    rows = []

    use_new_shaping = (not args.use_old_shaping)

    print(f"Evaluando {len(ckpts)} checkpoints...")
    print(f"TensorBoard logs: {tb_dir}")
    print(f"CSV resumen: {csv_path}")

    best_step = None
    best_return = -np.inf

    for ckpt in tqdm(ckpts):
        step = extract_step(ckpt)

        summary = evaluate_checkpoint(
            model_path=ckpt,
            n_episodes=args.n_episodes,
            seed=args.seed,
            device=args.device,
            use_new_shaping=use_new_shaping,
        )

        # Scalars principales
        writer.add_scalar("eval/return_mean", summary["return_mean"], step)
        writer.add_scalar("eval/return_std", summary["return_std"], step)

        writer.add_scalar("eval/episode_length_mean", summary["episode_length_mean"], step)
        writer.add_scalar("eval/episode_length_std", summary["episode_length_std"], step)

        writer.add_scalar("eval/terminated_rate", summary["terminated_mean"], step)
        writer.add_scalar("eval/truncated_rate", summary["truncated_mean"], step)

        writer.add_scalar("eval/final_torso_height_mean", summary["final_torso_height_mean"], step)
        writer.add_scalar("eval/final_abs_torso_angle_mean", summary["final_abs_torso_angle_mean"], step)
        writer.add_scalar("eval/final_healthy_rate", summary["final_healthy_mean"], step)

        # Componentes de reward
        writer.add_scalar("eval_reward_terms/base_reward_sum_mean", summary["base_reward_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/speed_bonus_sum_mean", summary["speed_bonus_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/height_pen_sum_mean", summary["height_pen_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/angle_pen_sum_mean", summary["angle_pen_sum_mean"], step)
        writer.add_scalar("eval_reward_terms/alive_bonus_sum_mean", summary["alive_bonus_sum_mean"], step)

        # Histogramas utiles
        writer.add_histogram("eval_distributions/returns", summary["_raw_returns"], step)
        writer.add_histogram("eval_distributions/episode_lengths", summary["_raw_lengths"], step)

        row = {
            "checkpoint": ckpt.name,
            "step": step,
            "return_mean": summary["return_mean"],
            "return_std": summary["return_std"],
            "episode_length_mean": summary["episode_length_mean"],
            "episode_length_std": summary["episode_length_std"],
            "terminated_rate": summary["terminated_mean"],
            "truncated_rate": summary["truncated_mean"],
            "final_torso_height_mean": summary["final_torso_height_mean"],
            "final_torso_height_std": summary["final_torso_height_std"],
            "final_abs_torso_angle_mean": summary["final_abs_torso_angle_mean"],
            "final_abs_torso_angle_std": summary["final_abs_torso_angle_std"],
            "final_healthy_rate": summary["final_healthy_mean"],
            "base_reward_sum_mean": summary["base_reward_sum_mean"],
            "speed_bonus_sum_mean": summary["speed_bonus_sum_mean"],
            "height_pen_sum_mean": summary["height_pen_sum_mean"],
            "angle_pen_sum_mean": summary["angle_pen_sum_mean"],
            "alive_bonus_sum_mean": summary["alive_bonus_sum_mean"],
        }
        rows.append(row)

        if summary["return_mean"] > best_return:
            best_return = summary["return_mean"]
            best_step = step

        print(
            f"[step={step}] "
            f"return={summary['return_mean']:.2f}�{summary['return_std']:.2f} | "
            f"len={summary['episode_length_mean']:.1f}�{summary['episode_length_std']:.1f} | "
            f"healthy_end={summary['final_healthy_mean']:.2f}"
        )

    writer.close()

    df = pd.DataFrame(rows).sort_values("step")
    df.to_csv(csv_path, index=False)

    print("\nResumen final")
    print(f"Mejor checkpoint por return_mean: step={best_step} | return_mean={best_return:.2f}")
    print(f"CSV guardado en: {csv_path}")
    print(f"TensorBoard logdir: {tb_dir}")


if __name__ == "__main__":
    main()